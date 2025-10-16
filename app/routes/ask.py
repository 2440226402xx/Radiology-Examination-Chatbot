from flask import Blueprint, request, jsonify, current_app, Response, g
from pydantic import BaseModel, Field, ValidationError, constr
from typing import Optional, Dict, Any, Iterable
from uuid import uuid4
from time import time
import json
import asyncio
import inspect
from app.core.utils.logger import get_logger
from app.core.utils.scenario_classifier import classify
from app.core.utils.response_formatter import format_response
from app.services.prompt_service import select_prompt
from app.services.chat_service import generate, generate_stream

bp = Blueprint("ask", __name__, url_prefix="/api/v1")
logger = get_logger(__name__)
_rate_bucket: Dict[str, list] = {}

class AskPayload(BaseModel):
    message: constr(min_length=1)
    context: Optional[Dict[str, Any]] = None
    scenario: Optional[constr(strip_whitespace=True)] = None
    stream: bool = False
    temperature: Optional[float] = Field(default=0.2, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=8192)
    timeout_sec: Optional[int] = Field(default=60, ge=1, le=300)
    user_id: Optional[str] = None
    trace: Optional[bool] = False

def _rate_limit(key: str, limit: int = 60, window: int = 60) -> bool:
    now = time()
    bucket = _rate_bucket.setdefault(key, [])
    while bucket and now - bucket[0] > window:
        bucket.pop(0)
    if len(bucket) >= limit:
        return False
    bucket.append(now)
    return True

def _request_id() -> str:
    return request.headers.get("X-Request-ID", str(uuid4()))

def _client_key() -> str:
    return request.headers.get("X-User-ID") or request.remote_addr or "anonymous"

def _json(obj: Any, status: int = 200, rid: Optional[str] = None):
    resp = jsonify(obj)
    resp.status_code = status
    if rid:
        resp.headers["X-Request-ID"] = rid
    return resp

@bp.route("/ask", methods=["POST"])
def ask_route():
    rid = _request_id()
    client = _client_key()
    if not _rate_limit(client):
        return _json({"error": "rate_limited"}, 429, rid)
    try:
        payload = AskPayload.parse_obj(request.get_json(force=True, silent=False))
    except ValidationError as e:
        return _json({"error": "invalid_payload", "details": e.errors()}, 422, rid)
    g.request_id = rid
    g.user_id = payload.user_id
    meta = {"request_id": rid, "client": client, "user_id": payload.user_id}
    try:
        scen = payload.scenario or classify(payload.message, payload.context or {})
        prompt = select_prompt(scen, payload.context or {})
    except Exception as e:
        logger.exception("prepare_failure")
        return _json({"error": "prepare_failure", "details": str(e), "meta": meta}, 500, rid)
    async def _nonstream_worker():
        args = dict(
            prompt=prompt,
            message=payload.message,
            scenario=scen,
            temperature=payload.temperature,
            top_p=payload.top_p,
            max_tokens=payload.max_tokens,
            user_id=payload.user_id,
            request_id=rid,
            trace=payload.trace,
            context=payload.context or {},
        )
        if inspect.iscoroutinefunction(generate):
            result = await asyncio.wait_for(generate(**args), timeout=payload.timeout_sec)
        else:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(loop.run_in_executor(None, lambda: generate(**args)), timeout=payload.timeout_sec)
        return result
    async def _stream_worker() -> Iterable[bytes]:
        args = dict(
            prompt=prompt,
            message=payload.message,
            scenario=scen,
            temperature=payload.temperature,
            top_p=payload.top_p,
            max_tokens=payload.max_tokens,
            user_id=payload.user_id,
            request_id=rid,
            trace=payload.trace,
            context=payload.context or {},
        )
        async def agen():
            if inspect.isasyncgenfunction(generate_stream):
                async for chunk in generate_stream(**args):
                    yield chunk
            else:
                for chunk in generate_stream(**args):
                    yield chunk
        formatter_state = {}
        started = False
        try:
            async for chunk in asyncio.wait_for(agen().__aiter__(), timeout=payload.timeout_sec):
                started = True
                formatted = format_response(chunk, scenario=scen, streaming=True, state=formatter_state)
                data = {"request_id": rid, "scenario": scen, "chunk": formatted}
                yield (json.dumps(data, ensure_ascii=False) + "\n").encode("utf-8")
        except Exception as e:
            err = {"error": "stream_failure", "details": str(e), "request_id": rid}
            yield (json.dumps(err, ensure_ascii=False) + "\n").encode("utf-8")
        finally:
            if not started:
                yield (json.dumps({"request_id": rid, "status": "no_chunks"}) + "\n").encode("utf-8")
            yield (json.dumps({"request_id": rid, "status": "done"}) + "\n").encode("utf-8")
    try:
        if payload.stream:
            async def _streaming_response():
                async for b in _stream_worker():
                    yield b
            headers = {"X-Request-ID": rid, "Content-Type": "application/jsonl"}
            return Response(_streaming_response(), headers=headers)
        else:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(_nonstream_worker())
            finally:
                try:
                    loop.stop()
                except Exception:
                    pass
                try:
                    loop.close()
                except Exception:
                    pass
            formatted = format_response(result, scenario=scen, streaming=False, state=None)
            return _json({"request_id": rid, "scenario": scen, "data": formatted}, 200, rid)
    except asyncio.TimeoutError:
        return _json({"error": "timeout", "request_id": rid}, 504, rid)
    except Exception as e:
        logger.exception("execution_failure")
        return _json({"error": "execution_failure", "details": str(e), "request_id": rid}, 500, rid)

@bp.route("/ask/health", methods=["GET"])
def ask_health():
    rid = _request_id()
    return _json({"status": "ok", "request_id": rid}, 200, rid)
