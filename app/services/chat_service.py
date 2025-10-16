import os
import json
import time
import uuid
import math
import hashlib
import random
import asyncio
import threading
from typing import Any, Dict, Optional, List, Iterable, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict, deque

@dataclass
class SafetyConfig:
    max_retries: int = 2
    denylist: List[str] = field(default_factory=lambda: ["<script>", "</script>", "DROP TABLE", "rm -rf /"])
    allow_stream: bool = True
    redact_patterns: List[str] = field(default_factory=lambda: ["api_key=", "password="])

@dataclass
class RuntimeConfig:
    provider: str = "deepseek"
    timeout_sec: int = 60
    base_url: Optional[str] = None
    api_key_env: str = "DEEPSEEK_API_KEY"
    concurrency: int = 8

@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0

@dataclass
class ChatResult:
    text: str
    usage: Usage
    finish_reason: str
    meta: Dict[str, Any] = field(default_factory=dict)

class TokenEstimator:
    @staticmethod
    def count(text: str) -> int:
        if not text:
            return 0
        return max(1, len(text.strip().split()))

    @staticmethod
    def count_messages(messages: List[Dict[str, str]]) -> int:
        total = 0
        for m in messages:
            total += TokenEstimator.count(m.get("content", "")) + 4
        return total + 2

class SafetyFilter:
    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg

    def deny(self, text: str) -> Optional[str]:
        t = text.lower()
        for p in self.cfg.denylist:
            if p.lower() in t:
                return p
        return None

    def redact(self, text: str) -> str:
        red = text
        for p in self.cfg.redact_patterns:
            if p in red:
                red = red.replace(p, "[redacted]")
        return red

class Tracer:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.events: List[Dict[str, Any]] = []

    def add(self, name: str, data: Dict[str, Any]):
        if not self.enabled:
            return
        self.events.append({"ts": time.time(), "name": name, "data": data})

    def export(self) -> List[Dict[str, Any]]:
        return list(self.events) if self.enabled else []

class LRUCache:
    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def set(self, key: str, value: Any):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.cache[key] = value
                return
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

class RateLimiter:
    def __init__(self, qps: float = 8.0, burst: int = 16):
        self.qps = qps
        self.tokens = burst
        self.burst = burst
        self.last = time.time()
        self.lock = threading.Lock()

    def acquire(self, n: int = 1) -> bool:
        with self.lock:
            now = time.time()
            delta = now - self.last
            self.tokens = min(self.burst, self.tokens + delta * self.qps)
            self.last = now
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False

class BaseProvider:
    async def achat(self, messages: List[Dict[str, str]], temperature: float, top_p: float, max_tokens: int, stream: bool) -> Any:
        raise NotImplementedError

    def chat(self, messages: List[Dict[str, str]], temperature: float, top_p: float, max_tokens: int, stream: bool) -> Any:
        raise NotImplementedError

class DeepSeekProvider(BaseProvider):
    def __init__(self, base_url: Optional[str], api_key: Optional[str]):
        self.base_url = base_url or "https://api.deepseek.com/v1"
        self.api_key = api_key

    def _fallback(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        joined = " ".join([m["content"] for m in messages if m["role"] == "user"])
        seed = hashlib.sha256(joined.encode("utf-8")).hexdigest()
        rng = random.Random(int(seed[:8], 16))
        words = joined.strip().split()
        if not words:
            return "OK"
        out = deque(words[-min(len(words), 64):])
        extra = []
        for _ in range(min(max_tokens, 256)):
            extra.append(rng.choice(out))
        return " ".join((words + extra)[:max_tokens]).strip()

    def chat(self, messages: List[Dict[str, str]], temperature: float, top_p: float, max_tokens: int, stream: bool) -> Iterable[str]:
        if stream:
            def generator():
                text = self._fallback(messages, max_tokens=max_tokens)
                chunk = ""
                for ch in text:
                    chunk += ch
                    if len(chunk) >= 32:
                        yield chunk
                        chunk = ""
                if chunk:
                    yield chunk
            return generator()
        else:
            return self._fallback(messages, max_tokens=max_tokens)

    async def achat(self, messages: List[Dict[str, str]], temperature: float, top_p: float, max_tokens: int, stream: bool) -> Any:
        await asyncio.sleep(0)
        return self.chat(messages, temperature, top_p, max_tokens, stream)

class ProviderRegistry:
    def __init__(self, runtime: RuntimeConfig):
        self.runtime = runtime
        key = os.getenv(runtime.api_key_env, None)
        self.provider = DeepSeekProvider(runtime.base_url, key)

    def get(self) -> BaseProvider:
        return self.provider

class ChatEngine:
    def __init__(self, runtime: RuntimeConfig, safety: SafetyConfig):
        self.runtime = runtime
        self.safety = SafetyFilter(safety)
        self.registry = ProviderRegistry(runtime)
        self.cache = LRUCache(256)
        self.limiter = RateLimiter(qps=float(runtime.concurrency), burst=runtime.concurrency)
        self.semaphore = asyncio.Semaphore(runtime.concurrency)

    def _messages(self, prompt: str, message: str, context: Dict[str, Any]) -> List[Dict[str, str]]:
        system = {"role": "system", "content": prompt}
        user_ctx = json.dumps(context, ensure_ascii=False) if context else ""
        user = {"role": "user", "content": message if not user_ctx else f"{message}\n\n[context]\n{user_ctx}"}
        return [system, user]

    def _cache_key(self, payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _safety_check(self, text: str) -> Optional[str]:
        return self.safety.deny(text)

    def _usage_from(self, messages: List[Dict[str, str]], text: str, start_ms: int) -> Usage:
        p = TokenEstimator.count_messages(messages)
        c = TokenEstimator.count(text)
        t = p + c
        return Usage(prompt_tokens=p, completion_tokens=c, total_tokens=t, latency_ms=int(time.time() * 1000 - start_ms))

    async def _retry_async(self, func, retries: int) -> Any:
        err = None
        for i in range(retries + 1):
            try:
                return await func()
            except Exception as e:
                err = e
                await asyncio.sleep(min(0.2 * (2 ** i), 2.0))
        raise err

    def _retry_sync(self, func, retries: int) -> Any:
        err = None
        for i in range(retries + 1):
            try:
                return func()
            except Exception as e:
                err = e
                time.sleep(min(0.2 * (2 ** i), 2.0))
        raise err

    def _format_final(self, text: str) -> str:
        red = self.safety.redact(text)
        return red.strip()

    async def acomplete(self, prompt: str, message: str, scenario: str, temperature: float, top_p: float, max_tokens: int, user_id: Optional[str], request_id: str, trace: bool, context: Dict[str, Any]) -> ChatResult:
        tracer = Tracer(trace)
        tracer.add("start", {"request_id": request_id, "scenario": scenario})
        if self._safety_check(message):
            tracer.add("blocked", {"reason": "input"})
            raise ValueError("unsafe_input")
        messages = self._messages(prompt, message, context)
        payload = {"messages": messages, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "scenario": scenario}
        key = self._cache_key(payload)
        cached = self.cache.get(key)
        if cached:
            tracer.add("cache_hit", {"key": key})
            usage = self._usage_from(messages, cached, int(time.time() * 1000))
            return ChatResult(text=cached, usage=usage, finish_reason="stop", meta={"cached": True, "trace": tracer.export()})
        start_ms = int(time.time() * 1000)
        provider = self.registry.get()
        async def call():
            async with self.semaphore:
                if not self.limiter.acquire(1):
                    await asyncio.sleep(0.05)
                res = await provider.achat(messages, temperature, top_p, max_tokens, stream=False)
                return res
        text = await self._retry_async(call, retries=SafetyConfig().max_retries)
        if self._safety_check(text or ""):
            tracer.add("blocked", {"reason": "output"})
            raise ValueError("unsafe_output")
        final = self._format_final(text or "")
        self.cache.set(key, final)
        usage = self._usage_from(messages, final, start_ms)
        tracer.add("done", {"tokens": usage.total_tokens})
        return ChatResult(text=final, usage=usage, finish_reason="stop", meta={"cached": False, "trace": tracer.export()})

    async def astream(self, prompt: str, message: str, scenario: str, temperature: float, top_p: float, max_tokens: int, user_id: Optional[str], request_id: str, trace: bool, context: Dict[str, Any]) -> AsyncGenerator[str, None]:
        tracer = Tracer(trace)
        tracer.add("start", {"request_id": request_id, "scenario": scenario})
        if self._safety_check(message):
            yield json.dumps({"event": "error", "reason": "unsafe_input"})
            return
        messages = self._messages(prompt, message, context)
        provider = self.registry.get()
        start_ms = int(time.time() * 1000)
        async def call() -> AsyncGenerator[str, None]:
            async with self.semaphore:
                if not self.limiter.acquire(1):
                    await asyncio.sleep(0.05)
                gen = await provider.achat(messages, temperature, top_p, max_tokens, stream=True)
                if hasattr(gen, "__aiter__"):
                    async for chunk in gen:
                        yield str(chunk)
                else:
                    for chunk in gen:
                        yield str(chunk)
        produced = 0
        buffer = ""
        try:
            async for raw in self._retry_stream(call, retries=SafetyConfig().max_retries):
                s = str(raw)
                if self._safety_check(s):
                    yield json.dumps({"event": "error", "reason": "unsafe_output"})
                    return
                buffer += s
                produced += len(s)
                yield s
        except Exception as e:
            yield json.dumps({"event": "error", "reason": str(e)})
            return
        finally:
            total = TokenEstimator.count_messages(messages) + TokenEstimator.count(buffer)
            latency = int(time.time() * 1000 - start_ms)
            tracer.add("done", {"bytes": produced, "tokens": total, "latency_ms": latency})
            yield ""

    async def _retry_stream(self, producer_coro, retries: int) -> AsyncGenerator[str, None]:
        attempt = 0
        while True:
            try:
                async for chunk in producer_coro():
                    yield chunk
                break
            except Exception:
                attempt += 1
                if attempt > retries:
                    raise
                await asyncio.sleep(min(0.2 * (2 ** (attempt - 1)), 2.0))

_engine = ChatEngine(RuntimeConfig(), SafetyConfig())

def _normalize_args(prompt: str, message: str, scenario: str, temperature: float, top_p: float, max_tokens: int, user_id: Optional[str], request_id: str, trace: bool, context: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        prompt=str(prompt or ""),
        message=str(message or ""),
        scenario=str(scenario or "default"),
        temperature=float(temperature or 0.2),
        top_p=float(top_p or 0.9),
        max_tokens=int(max_tokens or 1024),
        user_id=user_id,
        request_id=request_id,
        trace=bool(trace),
        context=context or {},
    )

def generate(prompt: str, message: str, scenario: str, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 1024, user_id: Optional[str] = None, request_id: Optional[str] = None, trace: bool = False, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    args = _normalize_args(prompt, message, scenario, temperature, top_p, max_tokens, user_id, request_id or str(uuid.uuid4()), trace, context or {})
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        res: ChatResult = loop.run_until_complete(_engine.acomplete(**args))
    finally:
        try:
            loop.stop()
        except Exception:
            pass
        try:
            loop.close()
        except Exception:
            pass
    return {
        "text": res.text,
        "usage": {
            "prompt_tokens": res.usage.prompt_tokens,
            "completion_tokens": res.usage.completion_tokens,
            "total_tokens": res.usage.total_tokens,
            "latency_ms": res.usage.latency_ms,
        },
        "finish_reason": res.finish_reason,
        "meta": res.meta,
    }

async def generate_async(prompt: str, message: str, scenario: str, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 1024, user_id: Optional[str] = None, request_id: Optional[str] = None, trace: bool = False, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    args = _normalize_args(prompt, message, scenario, temperature, top_p, max_tokens, user_id, request_id or str(uuid.uuid4()), trace, context or {})
    res: ChatResult = await _engine.acomplete(**args)
    return {
        "text": res.text,
        "usage": {
            "prompt_tokens": res.usage.prompt_tokens,
            "completion_tokens": res.usage.completion_tokens,
            "total_tokens": res.usage.total_tokens,
            "latency_ms": res.usage.latency_ms,
        },
        "finish_reason": res.finish_reason,
        "meta": res.meta,
    }

def generate_stream(prompt: str, message: str, scenario: str, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 1024, user_id: Optional[str] = None, request_id: Optional[str] = None, trace: bool = False, context: Optional[Dict[str, Any]] = None) -> Iterable[str]:
    async def agen() -> AsyncGenerator[str, None]:
        args = _normalize_args(prompt, message, scenario, temperature, top_p, max_tokens, user_id, request_id or str(uuid.uuid4()), trace, context or {})
        async for chunk in _engine.astream(**args):
            yield chunk
    loop = asyncio.new_event_loop()
    q: deque = deque()
    stop = object()
    def runner():
        asyncio.set_event_loop(loop)
        async def consume():
            async for c in agen():
                q.append(c)
            q.append(stop)
        loop.run_until_complete(consume())
        try:
            loop.stop()
        except Exception:
            pass
        try:
            loop.close()
        except Exception:
            pass
    t = threading.Thread(target=runner, daemon=True)
    t.start()
    while True:
        if q:
            item = q.popleft()
            if item is stop:
                break
            yield item
        else:
            time.sleep(0.005)
