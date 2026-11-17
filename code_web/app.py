"""Fire-Vision web app.

Listens on 0.0.0.0:$PORT. Render's proxy terminates TLS and forwards
WebSocket upgrades; ProxyFix trusts that one hop so Socket.IO sees https.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from flask import Flask, current_app, jsonify, render_template, request, send_file
from flask_socketio import SocketIO
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

import config
from alerts import AlertGate, AlertSender, twilio_configured

logger = logging.getLogger("firevision")
socketio = SocketIO()


def _form_int(value, default, lo, hi) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        number = default
    return max(lo, min(hi, number))


def _form_bool(value) -> bool:
    return str(value).lower() in {"on", "true", "1", "yes"}


def _emit(event, data):
    socketio.emit(event, data)


def request_stop():
    current_app.extensions["runner"].request_stop()
    socketio.emit("status", "Stopping after the current frame.")


def create_app(testing: bool = False) -> Flask:
    app = Flask(__name__)
    app.config["TESTING"] = testing
    # A few KB of multipart overhead, then Flask rejects the body itself.
    app.config["MAX_CONTENT_LENGTH"] = config.max_upload_bytes() + 8 * 1024
    socketio.init_app(
        app,
        async_mode="threading",
        cors_allowed_origins="*",
        ping_timeout=60,
        ping_interval=25,
    )
    # Outermost, so Render's X-Forwarded-Proto is applied before Socket.IO.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    if not getattr(socketio, "_firevision_handlers", False):
        socketio.on_event("stop_processing", request_stop)
        socketio._firevision_handlers = True

    from jobs import Runner

    runner = Runner()
    gate = AlertGate(AlertSender())
    app.extensions["runner"] = runner
    app.extensions["gate"] = gate

    def enqueue(target, kwargs) -> bool:
        if not runner.try_start():
            return False

        def wrapped():
            try:
                gate.reset()
                target(**kwargs)
            except Exception:
                logger.exception("video processing failed")
                _emit("status", "Processing failed before the clip finished.")
            finally:
                runner.finish()

        if app.config.get("TESTING"):
            app.config.setdefault("JOBS", []).append(wrapped)
            return True
        socketio.start_background_task(wrapped)
        return True

    @app.errorhandler(413)
    def too_large(_error):
        limit_mb = config.max_upload_bytes() // (1024 * 1024)
        return jsonify(error=f"Video must be under {limit_mb} MB."), 413

    @app.get("/health")
    def health():
        return jsonify(
            status="ok",
            alerts="twilio" if twilio_configured() else "demo",
            model=config.MODEL_PATH.name,
            model_present=config.MODEL_PATH.is_file(),
            frame_stride=config.frame_stride(),
            yolo_imgsz=config.yolo_imgsz(),
            max_edge=config.max_edge(),
            max_upload_mb=config.max_upload_bytes() / (1024 * 1024),
            max_duration_sec=config.max_duration_sec(),
            max_process_sec=config.max_process_sec(),
            sample=config.SAMPLE_PATH.name,
        )

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/upload")
    def upload_page():
        return render_template("index.html")

    @app.get("/models")
    def models():
        return render_template("models.html")

    @app.get("/media/sample")
    def sample_media():
        path = config.SAMPLE_PATH
        if not path.is_file():
            return jsonify(error="Sample clip is not in this build."), 404
        return send_file(path, mimetype="video/mp4", conditional=True, max_age=3600)

    def _launch(video_path: Path, options: dict, delete_after: bool):
        from processing import process_video

        started = enqueue(
            process_video,
            {
                "video_path": str(video_path),
                "run_motion": options["run_motion"],
                "run_yolo_detection": options["run_yolo"],
                "size_factor": options["size_factor"],
                "m_frames": options["m_frames"],
                "n_frames": options["n_frames"],
                "emit": _emit,
                "runner": runner,
                "gate": gate,
                "delete_after": delete_after,
            },
        )
        if not started:
            return jsonify(
                error="A video is already processing. Stop it or wait for it to finish."
            ), 409
        return jsonify(
            message="Processing started. Frames will show up as they finish.", result="started"
        ), 202

    @app.post("/demo")
    def demo():
        path = config.SAMPLE_PATH
        if not path.is_file():
            return jsonify(error="Sample clip is missing."), 404
        # Short windows so the 6.7s clip can show a spread arrow and a smoke mean.
        return _launch(
            path,
            {
                "run_motion": True,
                "run_yolo": True,
                "size_factor": 1,
                "m_frames": 12,
                "n_frames": 8,
            },
            delete_after=False,
        )

    @app.post("/upload")
    def handle_upload():
        from processing import VideoRejected, validate_upload

        if "video" not in request.files:
            return jsonify(error="No video file uploaded"), 400
        upload = request.files["video"]
        original = secure_filename(upload.filename or "")
        suffix = Path(original).suffix.lower()
        if suffix not in config.ALLOWED_EXTENSIONS:
            return jsonify(error="Upload an MP4, WebM, MOV, AVI, or MKV file."), 400

        run_motion = _form_bool(request.form.get("cameraStable"))
        run_yolo = _form_bool(request.form.get("runYolo"))
        if not run_motion and not run_yolo:
            return jsonify(error="Turn on camera-stable analysis, YOLO, or both."), 400

        config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
        upload.save(dest)
        try:
            validate_upload(dest)
        except VideoRejected as exc:
            dest.unlink(missing_ok=True)
            return jsonify(error=str(exc)), 400

        options = {
            "run_motion": run_motion,
            "run_yolo": run_yolo,
            "size_factor": request.form.get("sizeFactor", "1"),
            "m_frames": _form_int(request.form.get("mFrames"), 180, 0, 300),
            "n_frames": _form_int(request.form.get("nFrames"), 30, 0, 300),
        }
        try:
            response = _launch(dest, options, delete_after=True)
        except Exception:
            dest.unlink(missing_ok=True)
            raise
        if response[1] != 202:
            dest.unlink(missing_ok=True)
        return response

    return app


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app = create_app()
    port = int(os.environ.get("PORT", "5000"))
    mode = "twilio" if twilio_configured() else "demo"
    logger.info("listening on 0.0.0.0:%s alerts=%s", port, mode)
    socketio.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
