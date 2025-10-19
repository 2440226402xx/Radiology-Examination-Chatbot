import os
import sys
import time
import signal
import argparse
import threading
import socket
from contextlib import closing
from pathlib import Path

from app.config.config import AppConfig
from app.api import create_app

def env_str(key, default):
    v = os.getenv(key)
    return v if v is not None and str(v).strip() != "" else default

def env_int(key, default):
    v = os.getenv(key)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def env_bool(key, default):
    v = os.getenv(key)
    if v is None:
        return default
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}

def pick_port(default_port):
    host = env_str("HOST", "0.0.0.0")
    port = env_int("PORT", default_port)
    if port > 0:
        return host, port
    for p in range(5000, 5100):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, p))
                return host, p
            except OSError:
                continue
    return host, default_port

def install_signals(stop_event, app):
    def handler(signum, frame):
        try:
            app.logger.info(f"signal={signum} shutting_down=true")
        except Exception:
            pass
        stop_event.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except Exception:
            pass

def wait_for_stop(stop_event):
    while not stop_event.is_set():
        time.sleep(0.2)

def parse_args():
    parser = argparse.ArgumentParser(prog="deepseek-rec-advanced")
    parser.add_argument("--host", type=str, default=env_str("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=env_int("PORT", 5000))
    parser.add_argument("--workers", type=int, default=env_int("WORKERS", 1))
    parser.add_argument("--reload", action="store_true", default=env_bool("RELOAD", False))
    parser.add_argument("--debug", action="store_true", default=env_bool("DEBUG", AppConfig.DEBUG))
    parser.add_argument("--no-access-log", action="store_true", default=not AppConfig.ACCESS_LOG)
    return parser.parse_args()

def run_single(app, host, port, debug, use_reloader):
    app.logger.info(f"server=werkzeug host={host} port={port} debug={debug} reload={use_reloader}")
    app.run(host=host, port=port, debug=debug, use_reloader=use_reloader)

def run_threaded(app, host, port, workers):
    from wsgiref.simple_server import make_server, WSGIServer, WSGIRequestHandler
    class SilentHandler(WSGIRequestHandler):
        def log_message(self, format, *args):
            try:
                if AppConfig.ACCESS_LOG:
                    super().log_message(format, *args)
            except Exception:
                pass
    threads = []
    stop_event = threading.Event()
    install_signals(stop_event, app)
    def serve(bind_port):
        httpd = make_server(host, bind_port, app, handler_class=SilentHandler)
        app.logger.info(f"server=wsgi worker=1 host={host} port={bind_port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                httpd.server_close()
            except Exception:
                pass
    for i in range(workers):
        h, p = host, port + i
        t = threading.Thread(target=serve, args=(p,), daemon=True)
        t.start()
        threads.append(t)
    try:
        wait_for_stop(stop_event)
    finally:
        for t in threads:
            try:
                t.join(timeout=0.2)
            except Exception:
                pass

def ensure_dirs():
    Path(AppConfig.LOG_DIR).mkdir(parents=True, exist_ok=True)
    tmp = Path(AppConfig.TMP_DIR) if isinstance(AppConfig.TMP_DIR, str) else Path(str(AppConfig.TMP_DIR))
    tmp.mkdir(parents=True, exist_ok=True)

def main():
    args = parse_args()
    ensure_dirs()
    app = create_app()
    host = args.host
    port = args.port
    if port <= 0:
        host, port = pick_port(5000)
    if args.workers <= 1:
        run_single(app, host, port, args.debug, args.reload)
    else:
        run_threaded(app, host, port, args.workers)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"fatal_error: {e}\n")
        sys.stderr.flush()
        os._exit(1)
