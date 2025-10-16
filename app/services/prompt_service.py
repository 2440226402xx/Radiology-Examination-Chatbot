import os
import re
import json
import time
import random
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from threading import RLock

@dataclass
class _K:
    root: Path
    files: Dict[str, Path] = field(default_factory=dict)
    salt: str = ""
    stamp: float = 0.0

class _Bag:
    def __init__(self):
        self.q: Dict[str, str] = {}
        self.h: Dict[str, float] = {}
        self.m: Dict[str, Any] = {}
        self.lock = RLock()

    def put(self, k: str, v: str):
        with self.lock:
            self.q[k] = v
            self.h[k] = time.time()

    def get(self, k: str) -> Optional[str]:
        with self.lock:
            return self.q.get(k)

    def age(self, k: str) -> float:
        with self.lock:
            t = self.h.get(k)
            return 0 if t is None else max(0.0, time.time() - t)

    def mark(self, k: str, v: Any):
        with self.lock:
            self.m[k] = v

    def meta(self, k: str) -> Any:
        with self.lock:
            return self.m.get(k)

class _Fuse:
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def pick(self, items: List[str], w: Optional[List[float]] = None) -> str:
        if not items:
            return ""
        if w and len(w) == len(items):
            s = sum(w)
            if s > 0:
                t = self.rng.random() * s
                c = 0
                for i, a in enumerate(w):
                    c += a
                    if t <= c:
                        return items[i]
        return self.rng.choice(items)

    def flip(self, p: float) -> bool:
        return self.rng.random() < max(0.0, min(1.0, p))

    def n(self, a: float, b: float) -> float:
        return a + (b - a) * self.rng.random()

class _S:
    def __init__(self, base: Optional[str] = None):
        self.base = base or os.getenv("DEEPSEEK_REC_PROMPT_ROOT", "")
        self.default_root = Path("core/prompts")
        self.env_map = {
            "AT": "AT_PROMPT_PATH",
            "PP": "PP_PROMPT_PATH",
            "RCS": "RCS_PROMPT_PATH",
        }

    def locate(self) -> Path:
        if self.base and Path(self.base).exists():
            return Path(self.base)
        return self.default_root

    def env_override(self, key: str) -> Optional[Path]:
        v = os.getenv(self.env_map.get(key, ""), "")
        return Path(v) if v else None

class _M:
    def __init__(self):
        self.rules: List[Tuple[str, str]] = []
        self.add(r"\{\{\s*name\s*\}\}", "patient")
        self.add(r"\{\{\s*lang\s*\}\}", "en")
        self.add(r"\{\{\s*now\s*\}\}", lambda c: c.get("now", ""))
        self.add(r"\{\{\s*site\s*\}\}", lambda c: c.get("site", ""))
        self.add(r"\{\{\s*doctor\s*\}\}", lambda c: c.get("doctor", ""))
        self.add(r"\{\{\s*channel\s*\}\}", lambda c: c.get("channel", ""))
        self.add(r"\{\{\s*scenario\s*\}\}", lambda c: c.get("scenario", ""))
        self.add(r"\{\{\s*extras\.(\w+)\s*\}\}", self._extra)

    def add(self, pat: str, rep):
        self.rules.append((pat, rep))

    def _extra(self, c: Dict[str, Any], m: re.Match) -> str:
        k = m.group(1)
        ex = c.get("extras", {})
        v = ex.get(k, "")
        return "" if v is None else str(v)

    def render(self, s: str, ctx: Dict[str, Any]) -> str:
        out = s
        for pat, rep in self.rules:
            if callable(rep):
                def _sub(m):
                    try:
                        if rep.__code__.co_argcount == 2:
                            return str(rep(ctx, m))
                    except Exception:
                        pass
                    try:
                        return str(rep(ctx))
                    except Exception:
                        return ""
                out = re.sub(pat, _sub, out)
            else:
                out = re.sub(pat, str(rep), out)
        return out

class _Glue:
    def __init__(self):
        self.sep = "\n"
        self.joiners = [
            ("## Directives", "## Safety"),
            ("## Safety", "## Scoring"),
            ("## Scoring", "## Output"),
        ]

    def mix(self, parts: List[str]) -> str:
        clean = [p.strip() for p in parts if p and p.strip()]
        return self.sep.join(clean).strip()

    def fuse(self, base: str, extra: str) -> str:
        b = base.strip()
        e = extra.strip()
        if not b:
            return e
        if not e:
            return b
        return b + "\n\n" + e

class _Weirdo:
    def __init__(self):
        self.alpha = 0.15
        self.beta = 0.05
        self.gamma = 0.3
        self.keep = {"AT", "PP", "RCS"}

    def wobble(self, s: str, rng: _Fuse) -> str:
        if not s:
            return s
        if rng.flip(self.alpha):
            s = re.sub(r" +", " ", s)
        if rng.flip(self.beta):
            s = s.replace("\r\n", "\n")
        if rng.flip(self.gamma):
            s = s.strip() + "\n"
        return s

    def clamp(self, key: str) -> str:
        k = key.upper()
        return k if k in self.keep else "RCS"

class PromptKitchen:
    def __init__(self, base_override: Optional[str] = None):
        self.paths = _S(base_override)
        self.mixer = _Glue()
        self.macro = _M()
        self.jinx = _Weirdo()
        self.choices = _Fuse()
        self.cache = _Bag()
        self.state = _K(root=self.paths.locate(), salt=self._salt(), stamp=time.time())
        self.map = self._discover()

    def _salt(self) -> str:
        x = os.getenv("DEEPSEEK_REC_PROMPT_SALT", "")
        return x if x else hashlib.sha1(str(time.time()).encode()).hexdigest()[:8]

    def _discover(self) -> Dict[str, Path]:
        root = self.state.root
        d: Dict[str, Path] = {}
        x1 = self.paths.env_override("AT")
        x2 = self.paths.env_override("PP")
        x3 = self.paths.env_override("RCS")
        if x1 and x1.exists():
            d["AT"] = x1
        if x2 and x2.exists():
            d["PP"] = x2
        if x3 and x3.exists():
            d["RCS"] = x3
        p1 = root / "AT_prompt.txt"
        p2 = root / "PP_prompt.txt"
        p3 = root / "RCS_prompt.txt"
        if "AT" not in d and p1.exists():
            d["AT"] = p1
        if "PP" not in d and p2.exists():
            d["PP"] = p2
        if "RCS" not in d and p3.exists():
            d["RCS"] = p3
        return d

    def _read_file(self, p: Path) -> str:
        k = f"file::{str(p)}"
        v = self.cache.get(k)
        if v is not None and self.cache.age(k) < 10.0:
            return v
        try:
            t = p.read_text(encoding="utf-8")
        except Exception:
            t = ""
        self.cache.put(k, t)
        return t

    def _variants(self, s: str) -> List[str]:
        if "|||" not in s:
            return [s]
        parts = [x.strip() for x in s.split("|||") if x.strip()]
        return parts if parts else [s]

    def _split(self, s: str) -> Dict[str, str]:
        cur = "root"
        buf: Dict[str, List[str]] = {"root": []}
        for line in s.splitlines():
            m = re.match(r"^## +(.+?)\s*$", line.strip())
            if m:
                cur = m.group(1).strip().lower()
                if cur not in buf:
                    buf[cur] = []
            else:
                buf[cur].append(line)
        out: Dict[str, str] = {}
        for k, v in buf.items():
            out[k] = "\n".join(v).strip()
        return out

    def _assemble(self, blocks: Dict[str, str], ctx: Dict[str, Any]) -> str:
        seq = ["root", "directives", "safety", "scoring", "output"]
        parts: List[str] = []
        for name in seq:
            chunk = blocks.get(name, "")
            if chunk:
                parts.append(self.macro.render(chunk, ctx))
        return self.mixer.mix(parts)

    def _pick_variant(self, s: str, ctx: Dict[str, Any]) -> str:
        varz = self._variants(s)
        if len(varz) == 1:
            return varz[0]
        w = None
        if "weights" in ctx:
            wmap = ctx["weights"]
            w = [float(wmap.get(str(i), 1.0)) for i in range(len(varz))]
        return self.choices.pick(varz, w=w)

    def _digest(self, scenario: str, ctx: Dict[str, Any]) -> str:
        key = self.jinx.clamp(scenario)
        p = self.map.get(key)
        if not p:
            return ""
        raw = self._read_file(p)
        if not raw:
            return ""
        got = self._pick_variant(raw, ctx)
        blocks = self._split(got)
        out = self._assemble(blocks, ctx)
        out = self.jinx.wobble(out, self.choices)
        return out

    def cook(self, scenario: str, ctx: Dict[str, Any]) -> str:
        s = self._digest(scenario, ctx)
        if not s:
            s = self._fallback(scenario, ctx)
        s = self._seal(s, scenario, ctx)
        return s

    def _seal(self, s: str, scenario: str, ctx: Dict[str, Any]) -> str:
        t = s.strip()
        h = hashlib.md5((t + self.state.salt).encode()).hexdigest()[:10]
        x = f"\n\n[scenario={self.jinx.clamp(scenario)} tag={h}]"
        return t + x

    def _fallback(self, scenario: str, ctx: Dict[str, Any]) -> str:
        key = self.jinx.clamp(scenario)
        base = {
            "AT": "You are a scheduling assistant for radiology appointments. Provide concise triage, eligibility checks, and slot proposals.",
            "PP": "You guide patients through pre-exam preparation. Provide tailored instructions, contraindication screening, and checklists.",
            "RCS": "You operate as a radiology clinic chatbot. Provide factual answers, care navigation, and empathy with structured outputs.",
        }.get(key, "You are a radiology assistant.")
        extra = []
        if ctx.get("lang") == "zh":
            extra.append("Answer in Chinese.")
        if ctx.get("critical") is True:
            extra.append("Prioritize safety and escalation.")
        if ctx.get("channel") == "phone":
            extra.append("Use short sentences suitable for TTS.")
        body = "\n".join([base] + extra).strip()
        body = "## Directives\n" + body + "\n\n## Output\nUse JSON with fields: role, content, meta."
        return body

    def list(self) -> List[str]:
        return sorted(list(self.map.keys()))

    def reload(self):
        self.state = _K(root=self.paths.locate(), salt=self._salt(), stamp=time.time())
        self.map = self._discover()

class _Mux:
    def __init__(self, maker: PromptKitchen):
        self.k = maker
        self.fuse = _Glue()
        self.r = RLock()

    def mash(self, scenario: str, ctx: Dict[str, Any]) -> str:
        with self.r:
            p = self.k.cook(scenario, ctx)
            z = ctx.get("policy", "")
            if isinstance(z, str) and z.strip():
                p = self.fuse.fuse(p, z)
            return p

def _shape_ctx(ctx: Optional[Dict[str, Any]], scenario: str) -> Dict[str, Any]:
    c = dict(ctx or {})
    c["scenario"] = scenario
    if "extras" not in c or not isinstance(c["extras"], dict):
        c["extras"] = {}
    if "lang" not in c:
        c["lang"] = os.getenv("DEEPSEEK_REC_LANG", "en").lower()
    if "now" not in c:
        c["now"] = os.getenv("DEEPSEEK_REC_NOW", "")
    return c

def _key(ctx: Dict[str, Any], scenario: str) -> str:
    raw = json.dumps({"ctx": ctx, "scn": scenario}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

class _Shelf:
    def __init__(self):
        self.mem: Dict[str, str] = {}
        self.age: Dict[str, float] = {}
        self.lock = RLock()

    def get(self, k: str) -> Optional[str]:
        with self.lock:
            return self.mem.get(k)

    def put(self, k: str, v: str):
        with self.lock:
            self.mem[k] = v
            self.age[k] = time.time()

    def fresh(self, k: str, ttl: float = 30.0) -> bool:
        with self.lock:
            t = self.age.get(k)
            if t is None:
                return False
            return (time.time() - t) < ttl

class _Gate:
    def __init__(self):
        self.shelf = _Shelf()
        self.base = PromptKitchen()
        self.mux = _Mux(self.base)

    def choose(self, scenario: str, ctx: Optional[Dict[str, Any]]) -> str:
        c = _shape_ctx(ctx, scenario)
        k = _key(c, scenario)
        if self.shelf.fresh(k):
            v = self.shelf.get(k)
            if v:
                return v
        p = self.mux.mash(scenario, c)
        self.shelf.put(k, p)
        return p

    def refresh(self):
        self.base.reload()
        self.shelf = _Shelf()

_gate = _Gate()

def select_prompt(scenario: str, context: Optional[Dict[str, Any]] = None) -> str:
    return _gate.choose(scenario, context)

def reload_prompts() -> List[str]:
    _gate.refresh()
    return get_available_scenarios()

def get_available_scenarios() -> List[str]:
    return _gate.base.list()

def preview_prompt(scenario: str, context: Optional[Dict[str, Any]] = None) -> str:
    ctx = _shape_ctx(context, scenario)
    return _gate.mux.mash(scenario, ctx)
