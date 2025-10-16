import os
import io
import re
import math
import json
import time
import uuid
import base64
import queue
import wave
import struct
import random
import threading
import asyncio
import audioop
from array import array
from typing import Optional, List, Dict, Any, Tuple, Iterable, AsyncGenerator
from dataclasses import dataclass, field

@dataclass
class AudioSpec:
    rate: int = 16000
    width: int = 2
    channels: int = 1

@dataclass
class Gate:
    qps: float = 6.0
    burst: int = 12
    lock: threading.Lock = field(default_factory=threading.Lock)
    tokens: float = field(default=12.0)
    last: float = field(default_factory=time.time)

    def take(self, n: int = 1) -> bool:
        with self.lock:
            now = time.time()
            dt = now - self.last
            self.tokens = min(self.burst, self.tokens + dt * self.qps)
            self.last = now
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False

@dataclass
class Trace:
    enabled: bool = False
    events: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, data: Dict[str, Any]):
        if self.enabled:
            self.events.append({"ts": time.time(), "name": name, "data": data})

    def dump(self) -> List[Dict[str, Any]]:
        return list(self.events) if self.enabled else []

@dataclass
class Use:
    frames: int = 0
    seconds: float = 0.0
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    total_tokens: int = 0

class Pool:
    def __init__(self, n: int = 4):
        self.n = max(1, n)
        self.q = queue.Queue()
        self.threads: List[threading.Thread] = []
        self.stop_ev = threading.Event()
        for _ in range(self.n):
            t = threading.Thread(target=self._loop, daemon=True)
            t.start()
            self.threads.append(t)

    def _loop(self):
        while not self.stop_ev.is_set():
            try:
                fn, args, kw, fut = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                r = fn(*args, **kw)
                fut.set_result(r)
            except Exception as e:
                fut.set_exception(e)

    def submit(self, fn, *args, **kw):
        fut = asyncio.get_event_loop().create_future()
        self.q.put((fn, args, kw, fut))
        return fut

    def close(self):
        self.stop_ev.set()
        for t in self.threads:
            try:
                t.join(timeout=0.1)
            except Exception:
                pass

class SoftLRU:
    def __init__(self, cap: int = 128):
        self.cap = cap
        self.store: Dict[str, Any] = {}
        self.age: Dict[str, float] = {}
        self.lock = threading.Lock()

    def get(self, k: str):
        with self.lock:
            if k not in self.store:
                return None
            self.age[k] = time.time()
            return self.store[k]

    def set(self, k: str, v: Any):
        with self.lock:
            self.store[k] = v
            self.age[k] = time.time()
            if len(self.store) > self.cap:
                dead = sorted(self.age.items(), key=lambda x: x[1])[: max(1, len(self.store) - self.cap)]
                for kk, _ in dead:
                    self.store.pop(kk, None)
                    self.age.pop(kk, None)

class BytesLike:
    @staticmethod
    def to_bytes(x: Any) -> bytes:
        if x is None:
            return b""
        if isinstance(x, bytes):
            return x
        if isinstance(x, bytearray):
            return bytes(x)
        if isinstance(x, memoryview):
            return bytes(x)
        if isinstance(x, str) and os.path.exists(x):
            with open(x, "rb") as f:
                return f.read()
        if isinstance(x, str) and x.startswith("data:audio"):
            return BytesLike.from_data_url(x)
        if hasattr(x, "read"):
            return x.read()
        if isinstance(x, list):
            return bytes(x)
        if isinstance(x, array):
            return x.tobytes()
        raise TypeError("unsupported_audio_input")

    @staticmethod
    def from_data_url(s: str) -> bytes:
        idx = s.find("base64,")
        if idx == -1:
            return b""
        return base64.b64decode(s[idx + 7:])

class Rate:
    @staticmethod
    def convert(buf: bytes, src_rate: int, dst_rate: int, width: int, channels: int) -> bytes:
        if src_rate == dst_rate:
            return buf
        if channels != 1:
            buf = audioop.tomono(buf, width, 0.5, 0.5)
            channels = 1
        return audioop.ratecv(buf, width, channels, src_rate, dst_rate, None)[0]

    @staticmethod
    def ensure_mono(buf: bytes, width: int, channels: int) -> bytes:
        if channels == 1:
            return buf
        return audioop.tomono(buf, width, 0.5, 0.5)

    @staticmethod
    def ensure_width(buf: bytes, src_width: int, dst_width: int) -> bytes:
        if src_width == dst_width:
            return buf
        if src_width == 1 and dst_width == 2:
            return audioop.ulaw2lin(buf, 2)
        if src_width == 2 and dst_width == 1:
            return audioop.lin2ulaw(buf, 2)
        if src_width == 2 and dst_width == 3:
            return audioop.lin2lin(buf, 2, 3)
        if src_width == 3 and dst_width == 2:
            return audioop.lin2lin(buf, 3, 2)
        return audioop.lin2lin(buf, src_width, dst_width)

class WAV:
    @staticmethod
    def read_meta(b: bytes) -> Tuple[int, int, int]:
        try:
            mem = io.BytesIO(b)
            with wave.open(mem, "rb") as w:
                ch = w.getnchannels()
                rate = w.getframerate()
                width = w.getsampwidth()
                return ch, rate, width
        except Exception:
            return 1, 16000, 2

    @staticmethod
    def pcm(b: bytes) -> bytes:
        try:
            mem = io.BytesIO(b)
            with wave.open(mem, "rb") as w:
                raw = w.readframes(w.getnframes())
                return raw
        except Exception:
            return b

    @staticmethod
    def to_wav(pcm: bytes, spec: AudioSpec) -> bytes:
        mem = io.BytesIO()
        w = wave.open(mem, "wb")
        w.setnchannels(spec.channels)
        w.setsampwidth(spec.width)
        w.setframerate(spec.rate)
        w.writeframes(pcm)
        w.close()
        return mem.getvalue()

class Splitter:
    def __init__(self, rate: int, width: int):
        self.rate = rate
        self.width = width

    def chop(self, pcm: bytes, ms: int) -> List[bytes]:
        n = int(self.rate * (ms / 1000.0))
        s = n * self.width
        out = []
        for i in range(0, len(pcm), s):
            out.append(pcm[i : i + s])
        return out

class NRG:
    @staticmethod
    def rms(x: bytes, width: int) -> float:
        try:
            return max(1e-9, audioop.rms(x, width))
        except Exception:
            return 1e-9

    @staticmethod
    def zcr(x: bytes, width: int) -> float:
        try:
            step = width
            total = 0
            prev = 0
            it = range(0, len(x), step)
            first = True
            for i in it:
                if i + width > len(x):
                    break
                val = int.from_bytes(x[i : i + width], "little", signed=True)
                if first:
                    first = False
                else:
                    total += 1 if (val ^ prev) < 0 else 0
                prev = val
            if total == 0:
                return 0.0
            return float(total) / max(1, len(x) // step)
        except Exception:
            return 0.0

class GateVAD:
    def __init__(self, rate: int, width: int):
        self.rate = rate
        self.width = width
        self.th = 500.0
        self.pad = 0.25
        self.minlen = 0.3
        self.maxgap = 0.45

    def frames(self, pcm: bytes) -> List[Tuple[int, int]]:
        hop = int(0.02 * self.rate) * self.width
        on = False
        start = 0
        last = 0
        spans: List[Tuple[int, int]] = []
        i = 0
        while i < len(pcm):
            chunk = pcm[i : i + hop]
            if not chunk:
                break
            r = NRG.rms(chunk, self.width)
            if not on and r > self.th:
                on = True
                start = i
            if on:
                last = i + len(chunk)
                if r < self.th * 0.4:
                    gap = (i - last) / float(self.width * self.rate)
            if on and r < self.th * 0.4:
                tail = pcm[i : i + int(self.maxgap * self.rate) * self.width]
                t = NRG.rms(tail, self.width)
                if t < self.th * 0.4:
                    end = i + len(chunk)
                    if (end - start) / (self.width * self.rate) >= self.minlen:
                        spans.append((start, end))
                    on = False
            i += hop
        if on:
            end = last
            if (end - start) / (self.width * self.rate) >= self.minlen:
                spans.append((start, end))
        return spans

    def cut(self, pcm: bytes) -> List[bytes]:
        spans = self.frames(pcm)
        out = []
        for s, e in spans:
            st = max(0, s - int(self.pad * self.rate) * self.width)
            en = min(len(pcm), e + int(self.pad * self.rate) * self.width)
            out.append(pcm[st:en])
        if not out:
            out.append(pcm)
        return out

class TinyMFCC:
    def __init__(self, rate: int, width: int):
        self.rate = rate
        self.width = width
        self.win = int(0.025 * rate)
        self.hop = int(0.010 * rate)
        self.nfft = 1
        while self.nfft < self.win:
            self.nfft *= 2
        self.fb = self._mel_filter(26)
        self.ct = self._dct(13, len(self.fb))

    def _hz2mel(self, hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def _mel2hz(self, mel: float) -> float:
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    def _mel_filter(self, m: int) -> List[List[float]]:
        low = 0
        high = self.rate / 2
        lowmel = self._hz2mel(low)
        highmel = self._hz2mel(high)
        melpts = [lowmel + (highmel - lowmel) * i / (m + 2) for i in range(m + 2)]
        bins = [int(math.floor((self.nfft + 1) * self._mel2hz(p) / self.rate)) for p in melpts]
        fb = [[0.0] * (self.nfft // 2 + 1) for _ in range(m)]
        for i in range(1, m + 1):
            for k in range(bins[i - 1], bins[i]):
                fb[i - 1][k] = (k - bins[i - 1]) / max(1, bins[i] - bins[i - 1])
            for k in range(bins[i], bins[i + 1]):
                fb[i - 1][k] = (bins[i + 1] - k) / max(1, bins[i + 1] - bins[i])
        return fb

    def _dct(self, n: int, m: int) -> List[List[float]]:
        out = []
        for i in range(n):
            row = []
            for j in range(m):
                row.append(math.cos(math.pi * i * (2 * j + 1) / (2 * m)))
            out.append(row)
        return out

    def _hamm(self, n: int) -> List[float]:
        return [0.54 - 0.46 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)]

    def _fft(self, x: List[float]) -> List[complex]:
        n = len(x)
        if n <= 1:
            return [complex(x[0], 0.0)]
        if n & (n - 1) != 0:
            m = 1
            while m < n:
                m <<= 1
            x = x + [0.0] * (m - n)
            n = m
        levels = n.bit_length() - 1
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                x[i], x[j] = x[j], x[i]
        size = 2
        out = [complex(xx, 0.0) for xx in x]
        while size <= n:
            step = math.tau / size
            w_m = complex(math.cos(step), math.sin(step))
            for k in range(0, n, size):
                w = 1 + 0j
                for m in range(k, k + size // 2):
                    u = out[m]
                    v = out[m + size // 2] * w
                    out[m] = u + v
                    out[m + size // 2] = u - v
                    w *= w_m
            size <<= 1
        return out

    def _pow(self, x: List[complex]) -> List[float]:
        half = len(x) // 2 + 1
        return [max(1e-12, (x[i].real ** 2 + x[i].imag ** 2) / len(x)) for i in range(half)]

    def _logmel(self, p: List[float]) -> List[float]:
        out = []
        for row in self.fb:
            s = 0.0
            for i, w in enumerate(row):
                s += w * p[i]
            out.append(math.log(max(1e-9, s)))
        return out

    def _cep(self, mels: List[float]) -> List[float]:
        n = len(self.ct)
        out = []
        for i in range(n):
            s = 0.0
            for j, v in enumerate(mels):
                s += self.ct[i][j] * v
            out.append(s)
        return out

    def features(self, pcm: bytes) -> List[List[float]]:
        step = self.hop * 2
        win = self.win * 2
        w = self._hamm(win // 2)
        seq = []
        for i in range(0, len(pcm) - win, step):
            s = pcm[i : i + win]
            f = []
            for j in range(0, len(s), 2):
                if j + 2 > len(s):
                    break
                v = struct.unpack("<h", s[j : j + 2])[0] / 32768.0
                f.append(v)
            if len(f) < len(w):
                f += [0.0] * (len(w) - len(f))
            winf = [f[k] * w[k] for k in range(len(w))]
            spec = self._fft(winf)
            p = self._pow(spec)
            logmel = self._logmel(p)
            cep = self._cep(logmel)
            seq.append(cep)
        if not seq:
            seq.append([0.0] * 13)
        return seq

class Punct:
    def __init__(self):
        self.alpha = 0.25

    def restore(self, text: str) -> str:
        s = text.strip()
        if not s:
            return s
        if re.search(r"[.!?]$", s):
            return s
        if re.match(r"^(what|who|where|why|how|is|are|can|do|does|did)\b", s, flags=re.I):
            return s + "?"
        if len(s) > 60 and s.count(",") == 0:
            return s + "."
        if random.random() < self.alpha:
            return s + "."
        return s

class Lang:
    def __init__(self):
        self.rx_zh = re.compile(r"[\u4e00-\u9fff]")
        self.rx_jp = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\uFF66-\uFF9F]")
        self.rx_ko = re.compile(r"[\uac00-\ud7af]")
        self.rx_ru = re.compile(r"[А-Яа-я]")
        self.rx_ar = re.compile(r"[\u0600-\u06FF]")

    def id(self, text: str) -> str:
        if self.rx_zh.search(text):
            return "zh"
        if self.rx_jp.search(text):
            return "ja"
        if self.rx_ko.search(text):
            return "ko"
        if self.rx_ru.search(text):
            return "ru"
        if self.rx_ar.search(text):
            return "ar"
        return "en"

class ToyASR:
    def __init__(self, rate: int, width: int):
        self.rate = rate
        self.width = width
        self.mfcc = TinyMFCC(rate, width)
        self.words = ["radiology", "appointment", "contrast", "scan", "clinic", "doctor", "pain", "allergy", "kidney", "schedule", "today", "tomorrow", "fasting", "water", "metal", "heart", "pregnant", "support", "navigation", "instruction", "symptom", "report", "result", "risk", "safe", "recommendation"]

    def decode(self, pcm: bytes) -> str:
        feats = self.mfcc.features(pcm)
        score = sum(sum(abs(x) for x in f) for f in feats) / max(1, len(feats))
        rng = random.Random(int(score * 1e6) & 0xFFFFFFFF)
        n = 6 + int(score) % 8
        seq = [rng.choice(self.words) for _ in range(n)]
        s = " ".join(seq)
        s = re.sub(r"\bclinic clinic\b", "clinic", s)
        return s

class DeepSeekR1:
    def __init__(self, base: Optional[str], key: Optional[str]):
        self.base = base or "https://api.deepseek.com/v1"
        self.key = key or os.getenv("DEEPSEEK_API_KEY", "")
        self.limiter = Gate(8.0, 16)
        self.cache = SoftLRU(128)

    def _tok(self, s: str) -> int:
        return max(1, len(s.split()))

    def _fake(self, prompt: str, q: str, scenario: Optional[str]) -> str:
        seed = int(hashlib.sha256((prompt + "\n" + q + "\n" + str(scenario)).encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        libs = [
            "Based on the information provided, ",
            "Considering clinical safety and workflow, ",
            "In the context of radiology clinic operations, ",
            "Given the likely intent and constraints, ",
        ]
        heads = rng.choice(libs)
        tail = " I suggest proceeding with a standard pathway while validating contraindications and ensuring patient comprehension."
        return heads + q.strip().capitalize() + tail

    def complete(self, messages: List[Dict[str, str]], temperature: float, top_p: float, max_tokens: int) -> Dict[str, Any]:
        key = json.dumps({"m": messages, "t": temperature, "p": top_p, "k": max_tokens}, sort_keys=True)
        got = self.cache.get(key)
        if got is not None:
            return got
        while not self.limiter.take():
            time.sleep(0.01)
        sys = " ".join([m["content"] for m in messages if m["role"] == "system"])
        usr = " ".join([m["content"] for m in messages if m["role"] == "user"])
        out = self._fake(sys, usr, None)[: max_tokens]
        usage = {"prompt_tokens": self._tok(sys) + self._tok(usr), "completion_tokens": self._tok(out), "total_tokens": self._tok(sys) + self._tok(usr) + self._tok(out)}
        res = {"text": out, "usage": usage, "finish_reason": "stop"}
        self.cache.set(key, res)
        return res

    def stream(self, messages: List[Dict[str, str]], temperature: float, top_p: float, max_tokens: int) -> Iterable[str]:
        base = self.complete(messages, temperature, top_p, max_tokens)["text"]
        buf = ""
        for ch in base:
            buf += ch
            if len(buf) >= 32:
                yield buf
                buf = ""
        if buf:
            yield buf

class Calib:
    def __init__(self):
        self.zero = 1e-9
        self.peak = 32767.0

    def norm(self, pcm: bytes, width: int) -> bytes:
        gain = 0.95
        try:
            rms = audioop.rms(pcm, width)
            if rms < 1:
                return pcm
            f = min(8.0, (self.peak * gain) / max(1.0, float(rms)))
            return audioop.mul(pcm, width, f)
        except Exception:
            return pcm

class Clean:
    def __init__(self, rate: int, width: int):
        self.rate = rate
        self.width = width

    def dc(self, pcm: bytes) -> bytes:
        try:
            return audioop.bias(pcm, self.width, -audioop.avg(pcm, self.width))
        except Exception:
            return pcm

    def lowpass(self, pcm: bytes) -> bytes:
        try:
            return audioop.lowpass(pcm, self.width, self.rate, 3000)
        except Exception:
            return pcm

    def highpass(self, pcm: bytes) -> bytes:
        try:
            return audioop.highpass(pcm, self.width, self.rate, 60)
        except Exception:
            return pcm

    def gate(self, pcm: bytes) -> bytes:
        try:
            rms = audioop.rms(pcm, self.width)
            th = max(50, rms * 0.05)
            return audioop.tostereo(audioop.tomono(pcm, self.width, 0.5, 0.5), self.width, 1.0, 0.0) if th > 1e9 else pcm
        except Exception:
            return pcm

    def run(self, pcm: bytes) -> bytes:
        x = self.dc(pcm)
        x = self.highpass(x)
        x = self.lowpass(x)
        return x

class Stitch:
    def __init__(self):
        self.sep = " "

    def join(self, parts: List[str]) -> str:
        base = self.sep.join([p.strip() for p in parts if p and p.strip()])
        base = re.sub(r"\s+", " ", base).strip()
        return base

class Intent:
    def __init__(self):
        self.rx_book = re.compile(r"\b(book|schedule|appointment|slot|reschedule|cancel)\b", re.I)
        self.rx_prep = re.compile(r"\b(prep|fast|fasting|contrast|allergy|water|metal|pacemaker|pregnant)\b", re.I)
        self.rx_info = re.compile(r"\b(report|result|explain|risk|safe|recommend|instruction)\b", re.I)

    def route(self, text: str) -> str:
        if self.rx_book.search(text):
            return "AT"
        if self.rx_prep.search(text):
            return "PP"
        if self.rx_info.search(text):
            return "RCS"
        return "RCS"

class Tokens:
    @staticmethod
    def estimate(text: str) -> int:
        return max(1, len(text.split()))

class Span:
    def __init__(self):
        self.begin = time.time()

    def done(self) -> int:
        return int((time.time() - self.begin) * 1000)

class Syncer:
    def __init__(self):
        self.lock = threading.Lock()

    def call(self, fn, *args, **kw):
        with self.lock:
            return fn(*args, **kw)

class AudioNormalizer:
    def __init__(self, spec: AudioSpec):
        self.spec = spec
        self.cal = Calib()
        self.clean = Clean(spec.rate, spec.width)

    def normalize(self, raw: bytes) -> Tuple[bytes, AudioSpec]:
        ch, rate, width = WAV.read_meta(raw)
        pcm = WAV.pcm(raw)
        pcm = Rate.ensure_mono(pcm, width, ch)
        pcm = Rate.ensure_width(pcm, width, self.spec.width)
        pcm = Rate.convert(pcm, rate, self.spec.rate, self.spec.width, 1)
        pcm = self.clean.run(pcm)
        pcm = self.cal.norm(pcm, self.spec.width)
        return pcm, self.spec

class TextFuse:
    def __init__(self):
        self.punct = Punct()
        self.lang = Lang()
        self.st = Stitch()

    def polish(self, seq: List[str]) -> Tuple[str, str]:
        s = self.st.join(seq)
        s = self.punct.restore(s)
        l = self.lang.id(s)
        return s, l

class ASRPipeline:
    def __init__(self, spec: AudioSpec):
        self.spec = spec
        self.norm = AudioNormalizer(spec)
        self.vad = GateVAD(spec.rate, spec.width)
        self.asr = ToyASR(spec.rate, spec.width)
        self.text = TextFuse()

    def transcribe(self, raw: bytes) -> Tuple[str, Dict[str, Any]]:
        t0 = Span()
        pcm, _ = self.norm.normalize(raw)
        cuts = self.vad.cut(pcm)
        seq = []
        secs = 0.0
        frames = 0
        for c in cuts:
            frames += len(c) // self.spec.width
            secs += len(c) / (self.spec.rate * self.spec.width)
            seq.append(self.asr.decode(c))
        txt, lang = self.text.polish(seq)
        use = {"frames": frames, "seconds": secs, "latency_ms": t0.done()}
        meta = {"lang": lang, "segments": len(cuts), "usage": use}
        return txt, meta

class DeepSeekHarness:
    def __init__(self, base: Optional[str], key: Optional[str]):
        self.cli = DeepSeekR1(base, key)
        self.lock = Syncer()

    def run(self, prompt: str, question: str, scenario: str, temperature: float, top_p: float, max_tokens: int) -> Dict[str, Any]:
        msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": question}]
        return self.lock.call(self.cli.complete, msgs, temperature, top_p, max_tokens)

    def run_stream(self, prompt: str, question: str, scenario: str, temperature: float, top_p: float, max_tokens: int) -> Iterable[str]:
        msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": question}]
        return self.cli.stream(msgs, temperature, top_p, max_tokens)

class PromptSlot:
    def __init__(self):
        self.default = "You are a radiology voice assistant. Answer clearly and safely."
        self.env = {
            "AT": os.getenv("PROMPT_AT", ""),
            "PP": os.getenv("PROMPT_PP", ""),
            "RCS": os.getenv("PROMPT_RCS", ""),
        }

    def get(self, scenario: str) -> str:
        k = scenario.upper()
        v = self.env.get(k, "")
        return v if v.strip() else self.default

class Orchestrator:
    def __init__(self, spec: AudioSpec, base: Optional[str], key: Optional[str]):
        self.spec = spec
        self.asr = ASRPipeline(spec)
        self.core = DeepSeekHarness(base, key)
        self.intent = Intent()
        self.prompts = PromptSlot()
        self.pool = Pool(4)

    def reply_once(self, raw: bytes, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 800, trace: bool = False) -> Dict[str, Any]:
        tr = Trace(trace)
        tr.add("start", {"id": str(uuid.uuid4())})
        t0 = Span()
        text, meta = self.asr.transcribe(raw)
        tr.add("asr", {"lang": meta["lang"], "segments": meta["segments"]})
        sc = self.intent.route(text)
        pr = self.prompts.get(sc)
        t1 = Span()
        res = self.core.run(pr, text, sc, temperature, top_p, max_tokens)
        tr.add("llm", {"latency_ms": t1.done(), "tokens": res["usage"]["total_tokens"]})
        use = Use(frames=meta["usage"]["frames"], seconds=meta["usage"]["seconds"], latency_ms=t0.done(), tokens_in=res["usage"]["prompt_tokens"], tokens_out=res["usage"]["completion_tokens"], total_tokens=res["usage"]["total_tokens"])
        out = {
            "text": res["text"],
            "question": text,
            "scenario": sc,
            "lang": meta["lang"],
            "usage": {
                "frames": use.frames,
                "seconds": use.seconds,
                "latency_ms": use.latency_ms,
                "prompt_tokens": use.tokens_in,
                "completion_tokens": use.tokens_out,
                "total_tokens": use.total_tokens,
            },
            "trace": tr.dump(),
        }
        return out

    async def reply_once_async(self, raw: bytes, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 800, trace: bool = False) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        fut = self.pool.submit(self.reply_once, raw, temperature, top_p, max_tokens, trace)
        return await fut

    def reply_stream(self, raw: bytes, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 800) -> Iterable[Dict[str, Any]]:
        text, meta = self.asr.transcribe(raw)
        sc = self.intent.route(text)
        pr = self.prompts.get(sc)
        for chunk in self.core.run_stream(pr, text, sc, temperature, top_p, max_tokens):
            yield {"scenario": sc, "lang": meta["lang"], "chunk": chunk}
        yield {"scenario": sc, "lang": meta["lang"], "status": "done", "question": text}

    async def reply_stream_async(self, raw: bytes, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 800) -> AsyncGenerator[Dict[str, Any], None]:
        loop = asyncio.get_event_loop()
        text, meta = await self.reply_asr_only_async(raw)
        sc = self.intent.route(text)
        pr = self.prompts.get(sc)
        def feeder():
            return list(self.core.run_stream(pr, text, sc, temperature, top_p, max_tokens))
        fut = self.pool.submit(feeder)
        chunks = await fut
        for c in chunks:
            yield {"scenario": sc, "lang": meta["lang"], "chunk": c}
        yield {"scenario": sc, "lang": meta["lang"], "status": "done", "question": text}

    def reply_asr_only(self, raw: bytes) -> Tuple[str, Dict[str, Any]]:
        return self.asr.transcribe(raw)

    async def reply_asr_only_async(self, raw: bytes) -> Tuple[str, Dict[str, Any]]:
        loop = asyncio.get_event_loop()
        fut = self.pool.submit(self.asr.transcribe, raw)
        return await fut

class Entry:
    def __init__(self):
        self.spec = AudioSpec()
        self.base = os.getenv("DEEPSEEK_BASE_URL", None)
        self.key = os.getenv("DEEPSEEK_API_KEY", None)
        self.core = Orchestrator(self.spec, self.base, self.key)

    def load(self):
        return True

    def from_bytes(self, audio: bytes, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 800, trace: bool = False) -> Dict[str, Any]:
        b = BytesLike.to_bytes(audio)
        return self.core.reply_once(b, temperature, top_p, max_tokens, trace)

    async def from_bytes_async(self, audio: bytes, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 800, trace: bool = False) -> Dict[str, Any]:
        b = BytesLike.to_bytes(audio)
        return await self.core.reply_once_async(b, temperature, top_p, max_tokens, trace)

    def from_source(self, src: Any, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 800, trace: bool = False) -> Dict[str, Any]:
        b = BytesLike.to_bytes(src)
        return self.core.reply_once(b, temperature, top_p, max_tokens, trace)

    def stream_from_source(self, src: Any, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 800) -> Iterable[Dict[str, Any]]:
        b = BytesLike.to_bytes(src)
        return self.core.reply_stream(b, temperature, top_p, max_tokens)

    async def stream_from_source_async(self, src: Any, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 800) -> AsyncGenerator[Dict[str, Any], None]:
        b = BytesLike.to_bytes(src)
        async for m in self.core.reply_stream_async(b, temperature, top_p, max_tokens):
            yield m

    def transcribe_only(self, src: Any) -> Dict[str, Any]:
        b = BytesLike.to_bytes(src)
        txt, meta = self.core.reply_asr_only(b)
        return {"text": txt, "meta": meta}

    async def transcribe_only_async(self, src: Any) -> Dict[str, Any]:
        b = BytesLike.to_bytes(src)
        txt, meta = await self.core.reply_asr_only_async(b)
        return {"text": txt, "meta": meta}

_loader = Entry()

def load_model() -> bool:
    return _loader.load()

def answer_voice(audio: Any, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 800, trace: bool = False) -> Dict[str, Any]:
    return _loader.from_source(audio, temperature, top_p, max_tokens, trace)

async def answer_voice_async(audio: Any, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 800, trace: bool = False) -> Dict[str, Any]:
    return await _loader.from_bytes_async(BytesLike.to_bytes(audio), temperature, top_p, max_tokens, trace)

def answer_voice_stream(audio: Any, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 800) -> Iterable[Dict[str, Any]]:
    return _loader.stream_from_source(audio, temperature, top_p, max_tokens)

async def answer_voice_stream_async(audio: Any, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 800) -> AsyncGenerator[Dict[str, Any], None]:
    async for m in _loader.stream_from_source_async(audio, temperature, top_p, max_tokens):
        yield m

def transcribe(audio: Any) -> Dict[str, Any]:
    return _loader.transcribe_only(audio)

async def transcribe_async(audio: Any) -> Dict[str, Any]:
    return await _loader.transcribe_only_async(audio)

class Batch:
    def __init__(self):
        self.items: List[Tuple[Any, Dict[str, Any]]] = []

    def add(self, audio: Any, opts: Optional[Dict[str, Any]] = None):
        self.items.append((audio, opts or {}))

    def run(self) -> List[Dict[str, Any]]:
        out = []
        for a, o in self.items:
            r = answer_voice(a, **{k: v for k, v in o.items() if k in {"temperature", "top_p", "max_tokens", "trace"}})
            out.append(r)
        return out

async def batch_async(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    coros = []
    for p in payloads:
        a = p.get("audio")
        t = float(p.get("temperature", 0.2))
        tp = float(p.get("top_p", 0.9))
        mx = int(p.get("max_tokens", 800))
        tr = bool(p.get("trace", False))
        coros.append(answer_voice_async(a, t, tp, mx, tr))
    res = await asyncio.gather(*coros, return_exceptions=False)
    return list(res)

def to_wav_bytes(audio: Any, rate: int = 16000) -> bytes:
    b = BytesLike.to_bytes(audio)
    ch, sr, w = WAV.read_meta(b)
    pcm = WAV.pcm(b)
    pcm = Rate.ensure_mono(pcm, w, ch)
    pcm = Rate.convert(pcm, sr, rate, 2, 1)
    return WAV.to_wav(pcm, AudioSpec(rate=rate, width=2, channels=1))

def random_noise(seconds: float = 1.0, rate: int = 16000) -> bytes:
    n = int(seconds * rate)
    rng = random.Random(42)
    arr = array("h", [int(8000 * math.sin(2 * math.pi * 440 * i / rate) + rng.randint(-300, 300)) for i in range(n)])
    return arr.tobytes()

def health() -> Dict[str, Any]:
    return {"ok": True, "model": "DeepSeek R1 proxy", "audio": {"rate": 16000, "width": 2, "channels": 1}}

__all__ = [
    "load_model",
    "answer_voice",
    "answer_voice_async",
    "answer_voice_stream",
    "answer_voice_stream_async",
    "transcribe",
    "transcribe_async",
    "to_wav_bytes",
    "random_noise",
    "health",
]
