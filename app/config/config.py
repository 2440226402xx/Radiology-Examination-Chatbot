import os
import json
import pathlib
import secrets
import datetime

def _coerce_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}

def _coerce_int(v, d=0):
    try:
        return int(str(v).strip())
    except Exception:
        return d

def _coerce_float(v, d=0.0):
    try:
        return float(str(v).strip())
    except Exception:
        return d

def _coerce_json(v, d=None):
    if v is None:
        return d
    try:
        return json.loads(v)
    except Exception:
        return d

def _coerce_list(v, sep=","):
    if v is None or v == "":
        return []
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    return [s.strip() for s in str(v).split(sep) if s.strip()]

def _env(key, default=None):
    return os.getenv(key, default)

def _path(p):
    if not p:
        return ""
    return str(pathlib.Path(p).expanduser().resolve())

def _ensure_dir(p):
    if not p:
        return p
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)
    return p

class AppConfig:
    APP_NAME = _env("APP_NAME", "DeepSeek REC Advanced")
    ENV = _env("FLASK_ENV", _env("ENV", "production"))
    DEBUG = _coerce_bool(_env("DEBUG", "false"))
    TESTING = _coerce_bool(_env("TESTING", "false"))
    SECRET_KEY = _env("SECRET_KEY", secrets.token_hex(32))
    HOST = _env("HOST", "0.0.0.0")
    PORT = _coerce_int(_env("PORT", "5000"), 5000)
    JSON_AS_ASCII = False
    JSONIFY_PRETTYPRINT_REGULAR = _coerce_bool(_env("JSON_PRETTY", "false"))
    PROPAGATE_EXCEPTIONS = _coerce_bool(_env("PROPAGATE_EXCEPTIONS", "false"))
    MAX_CONTENT_LENGTH = _coerce_int(_env("MAX_CONTENT_LENGTH", str(32 * 1024 * 1024)))
    LOG_DIR = _ensure_dir(_path(_env("LOG_DIR", "./logs")))
    LOG_LEVEL = _env("LOG_LEVEL", "INFO").upper()
    ACCESS_LOG = _coerce_bool(_env("ACCESS_LOG", "true"))
    CORS_ORIGINS = _coerce_list(_env("CORS_ORIGINS", "*"))
    CORS_ALLOW_HEADERS = _coerce_list(_env("CORS_ALLOW_HEADERS", "Content-Type,Authorization,X-User-ID,X-Request-ID"))
    CORS_EXPOSE_HEADERS = _coerce_list(_env("CORS_EXPOSE_HEADERS", "X-Request-ID"))
    CORS_SUPPORTS_CREDENTIALS = _coerce_bool(_env("CORS_SUPPORTS_CREDENTIALS", "false"))
    REQUEST_ID_HEADER = _env("REQUEST_ID_HEADER", "X-Request-ID")
    USER_ID_HEADER = _env("USER_ID_HEADER", "X-User-ID")
    RATELIMIT_QPS = _coerce_float(_env("RATELIMIT_QPS", "8.0"), 8.0)
    RATELIMIT_BURST = _coerce_int(_env("RATELIMIT_BURST", "16"), 16)
    REQUEST_TIMEOUT_SEC = _coerce_int(_env("REQUEST_TIMEOUT_SEC", "60"), 60)
    GENERATE_TIMEOUT_SEC = _coerce_int(_env("GENERATE_TIMEOUT_SEC", "60"), 60)
    STREAM_TIMEOUT_SEC = _coerce_int(_env("STREAM_TIMEOUT_SEC", "120"), 120)
    ENABLE_STREAMING = _coerce_bool(_env("ENABLE_STREAMING", "true"))
    ENABLE_TRACE = _coerce_bool(_env("ENABLE_TRACE", "false"))
    ENABLE_HEALTH = _coerce_bool(_env("ENABLE_HEALTH", "true"))
    PROMPT_ROOT = _path(_env("DEEPSEEK_REC_PROMPT_ROOT", "core/prompts"))
    AT_PROMPT_PATH = _path(_env("AT_PROMPT_PATH", os.path.join(PROMPT_ROOT, "AT_prompt.txt")))
    PP_PROMPT_PATH = _path(_env("PP_PROMPT_PATH", os.path.join(PROMPT_ROOT, "PP_prompt.txt")))
    RCS_PROMPT_PATH = _path(_env("RCS_PROMPT_PATH", os.path.join(PROMPT_ROOT, "RCS_prompt.txt")))
    DEEPSEEK_BASE_URL = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    DEEPSEEK_API_KEY = _env("DEEPSEEK_API_KEY", "")
    PROVIDER = _env("LLM_PROVIDER", "deepseek")
    PROVIDER_CONCURRENCY = _coerce_int(_env("PROVIDER_CONCURRENCY", "8"), 8)
    SAFETY_DENYLIST = _coerce_list(_env("SAFETY_DENYLIST", "<script>,</script>,DROP TABLE,rm -rf /"))
    SAFETY_REDACT = _coerce_list(_env("SAFETY_REDACT", "api_key=,password="))
    STREAM_CHUNK_BYTES = _coerce_int(_env("STREAM_CHUNK_BYTES", "32768"), 32768)
    STREAM_HEADER = "application/jsonl"
    ALLOWED_ORIGINS_ALL = CORS_ORIGINS == ["*"] or CORS_ORIGINS == ["*",""]
    DEFAULT_TEMPERATURE = _coerce_float(_env("DEFAULT_TEMPERATURE", "0.2"), 0.2)
    DEFAULT_TOP_P = _coerce_float(_env("DEFAULT_TOP_P", "0.9"), 0.9)
    DEFAULT_MAX_TOKENS = _coerce_int(_env("DEFAULT_MAX_TOKENS", "1024"), 1024)
    AUDIO_RATE = _coerce_int(_env("AUDIO_RATE", "16000"), 16000)
    AUDIO_WIDTH = _coerce_int(_env("AUDIO_WIDTH", "2"), 2)
    AUDIO_CHANNELS = _coerce_int(_env("AUDIO_CHANNELS", "1"), 1)
    TMP_DIR = _ensure_dir(_path(_env("TMP_DIR", "./tmp")))
    STATIC_DIR = _path(_env("STATIC_DIR", "./static"))
    TRUSTED_PROXIES = _coerce_list(_env("TRUSTED_PROXIES", ""))
    SERVER_NAME = _env("SERVER_NAME", "")
    PREFERRED_URL_SCHEME = _env("PREFERRED_URL_SCHEME", "http")
    SESSION_COOKIE_SECURE = _coerce_bool(_env("SESSION_COOKIE_SECURE", "false"))
    SESSION_COOKIE_HTTPONLY = _coerce_bool(_env("SESSION_COOKIE_HTTPONLY", "true"))
    SESSION_COOKIE_SAMESITE = _env("SESSION_COOKIE_SAMESITE", "Lax")
    RATE_WINDOW_SEC = _coerce_int(_env("RATE_WINDOW_SEC", "60"), 60)
    RATE_LIMIT_PER_WINDOW = _coerce_int(_env("RATE_LIMIT_PER_WINDOW", "60"), 60)
    METRICS_ENABLED = _coerce_bool(_env("METRICS_ENABLED", "true"))
    METRICS_NAMESPACE = _env("METRICS_NAMESPACE", "deepseek_rec")
    START_TIME = datetime.datetime.utcnow().isoformat() + "Z"

    @classmethod
    def as_mapping(cls):
        out = {}
        for k in dir(cls):
            if k.isupper():
                out[k] = getattr(cls, k)
        return out

    @classmethod
    def update_flask(cls, app):
        for k, v in cls.as_mapping().items():
            app.config[k] = v

    @classmethod
    def origin_allowed(cls, origin: str) -> bool:
        if cls.ALLOWED_ORIGINS_ALL:
            return True
        if not origin:
            return False
        o = origin.strip().lower()
        return any(o == x.strip().lower() for x in cls.CORS_ORIGINS)

    @classmethod
    def llm_kwargs(cls):
        return dict(temperature=cls.DEFAULT_TEMPERATURE, top_p=cls.DEFAULT_TOP_P, max_tokens=cls.DEFAULT_MAX_TOKENS)

    @classmethod
    def prompt_paths(cls):
        return dict(root=cls.PROMPT_ROOT, AT=cls.AT_PROMPT_PATH, PP=cls.PP_PROMPT_PATH, RCS=cls.RCS_PROMPT_PATH)

    @classmethod
    def audio_spec(cls):
        return dict(rate=cls.AUDIO_RATE, width=cls.AUDIO_WIDTH, channels=cls.AUDIO_CHANNELS)
