from flask import Flask, jsonify, request, g
from flask_cors import CORS
import time, traceback, logging, importlib
from pathlib import Path

def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.config.AppConfig')
    CORS(app, resources={r"/*": {"origins": "*"}})
    setup_logging(app)
    register_middlewares(app)
    register_routes(app)
    return app

def setup_logging(app):
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    handler = logging.FileHandler(log_path / "api.log", encoding="utf-8")
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)

def register_middlewares(app):
    @app.before_request
    def start_timer():
        g.start = time.time()

    @app.after_request
    def log_request(response):
        duration = round(time.time() - g.start, 3)
        app.logger.info(f"{request.method} {request.path} {response.status_code} {duration}s")
        return response

    @app.errorhandler(Exception)
    def handle_error(e):
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

def register_routes(app):
    routes_dir = Path(__file__).parent / "routes"
    for route_file in routes_dir.glob("*.py"):
        if route_file.name.startswith("__"):
            continue
        module_name = f"app.routes.{route_file.stem}"
        module = importlib.import_module(module_name)
        if hasattr(module, "bp"):
            app.register_blueprint(module.bp)

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
