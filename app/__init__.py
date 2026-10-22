from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

from app.intelligence import lbph


def create_app():
    app = Flask(__name__)
    lbph.load_model()          # muat model terakhir saat service menyala

    from app.api.routes import bp
    app.register_blueprint(bp)

    @app.errorhandler(Exception)
    def on_error(e):
        # Semua error dibalas JSON berisi "error", supaya Laravel bisa menampilkan penyebab aslinya
        if isinstance(e, HTTPException):
            return jsonify(error=e.description), e.code
        app.logger.exception("Unhandled error")
        return jsonify(error=f"Kesalahan di Python service: {e}"), 500

    return app