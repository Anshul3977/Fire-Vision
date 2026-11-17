import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from werkzeug.middleware.proxy_fix import ProxyFix

ROOT = Path(__file__).resolve().parents[1]


def write_clip(path, frames, fps, size=(80, 60)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    frame[:] = (40, 120, 255)
    for _ in range(frames):
        writer.write(frame)
    writer.release()


def test_socket_stop_is_delivered():
    from app import create_app, socketio

    app = create_app(testing=True)
    sock = socketio.test_client(app)
    assert sock.is_connected()
    sock.emit("stop_processing")
    assert app.extensions["runner"].stopped()
    names = [event["name"] for event in sock.get_received()]
    assert "status" in names


def test_health_and_pages(client):
    health = client.get("/health")
    assert health.status_code == 200
    body = health.get_json()
    assert body["status"] == "ok"
    assert body["alerts"] == "demo"
    assert body["max_duration_sec"] == 15
    assert isinstance(client.application.wsgi_app, ProxyFix)
    assert client.get("/").status_code == 200
    assert b"Run bundled sample" in client.get("/").data
    assert client.get("/upload").status_code == 200
    models = client.get("/models")
    assert models.status_code == 200
    assert b"model_data.json" in models.data
    sample = client.get("/media/sample")
    assert sample.status_code == 200
    assert sample.mimetype == "video/mp4"


def test_recorded_model_metrics_are_unchanged():
    recorded = json.loads((ROOT / "code_web" / "static" / "model_data.json").read_text())
    assert [row["model"] for row in recorded] == [
        "YOLO-v8-obb PT v0",
        "YOLO-v8 PT",
        "YOLO-v5-NU",
        "YOLO-v6",
        "D2Go Model",
    ]
    assert recorded[0]["precision"] == 0.9252
    assert recorded[0]["recall"] == 0.8919
    assert recorded[0]["map50"] == 0.9447
    assert recorded[0]["map50_95"] == 0.6321
    assert recorded[0]["iou"] == 0
    assert recorded[0]["epochs"] == 50
    assert recorded[1]["precision"] == 0.9002
    assert recorded[1]["recall"] == 0.8297
    assert recorded[1]["map50"] == 0.8902
    assert recorded[1]["map50_95"] == 0.5532
    assert recorded[2]["precision"] == 0.8990
    assert recorded[2]["recall"] == 0.8106
    assert recorded[2]["map50"] == 0.8723
    assert recorded[2]["map50_95"] == 0.5515
    assert recorded[3]["precision"] == 0.8791
    assert recorded[3]["recall"] == 0.7906
    assert recorded[3]["map50"] == 0.8468
    assert recorded[3]["map50_95"] == 0.5289
    assert recorded[4]["precision"] == 0.90
    assert recorded[4]["recall"] == 0.85
    assert recorded[4]["map50"] == 0.80
    assert recorded[4]["map50_95"] == 0.75
    assert recorded[4]["iou"] == 0.70
    assert recorded[4]["epochs"] == 1000


def test_health_reports_twilio_when_env_is_complete(client, monkeypatch):
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC_TEST")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token")
    monkeypatch.setenv("TWILIO_PHONE", "+10000000000")
    monkeypatch.setenv("ALERT_PHONE", "+10000000001")
    assert client.get("/health").get_json()["alerts"] == "twilio"


def test_upload_validation(client, tmp_path, monkeypatch):
    missing = client.post("/upload", data={})
    assert missing.status_code == 400

    bad = client.post("/upload", data={"video": (io.BytesIO(b"hi"), "notes.txt")})
    assert bad.status_code == 400

    neither = client.post(
        "/upload",
        data={"video": (io.BytesIO(b"hi"), "clip.mp4"), "sizeFactor": "1.5", "mFrames": "10.0"},
    )
    assert neither.status_code == 400
    assert "YOLO" in neither.get_json()["error"]

    from processing import VideoRejected, scaled_size, validate_upload

    assert scaled_size(640, 360, "1.5", 480) == (427, 240)
    assert scaled_size(640, 360, "0", 480)[0] <= 480
    long_clip = tmp_path / "long.avi"
    write_clip(long_clip, frames=30, fps=10)
    monkeypatch.setenv("MAX_DURATION_SEC", "2")
    with pytest.raises(VideoRejected):
        validate_upload(long_clip)

    clip = tmp_path / "ok.avi"
    write_clip(clip, frames=4, fps=10)
    captured = {}

    def fake_process(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("processing.process_video", fake_process)
    with clip.open("rb") as handle:
        started = client.post(
            "/upload",
            data={
                "video": (handle, "ok.avi"),
                "runYolo": "on",
                "sizeFactor": "1.5",
                "mFrames": "10.0",
                "nFrames": "5",
            },
            content_type="multipart/form-data",
        )
    assert started.status_code == 202
    job = client.application.config["JOBS"][0]
    job()
    assert captured["size_factor"] == "1.5"
    assert captured["m_frames"] == 10
    assert captured["run_yolo_detection"] is True
    assert captured["delete_after"] is True
    Path(captured["video_path"]).unlink(missing_ok=True)


def test_demo_queues_once_and_stop_flag(client, monkeypatch):
    seen = {}

    def fake_process(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr("processing.process_video", fake_process)
    first = client.post("/demo")
    assert first.status_code == 202
    second = client.post("/demo")
    assert second.status_code == 409
    from app import request_stop

    with client.application.app_context():
        request_stop()
    assert client.application.extensions["runner"].stopped()
    client.application.config["JOBS"][0]()
    assert seen["delete_after"] is False
    assert seen["run_motion"] is True
    assert seen["run_yolo_detection"] is True
    assert Path(seen["video_path"]).name == "sample_fire.mp4"


def test_oversized_body_is_rejected(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "200")
    from app import create_app

    app = create_app(testing=True)
    client = app.test_client()
    response = client.post("/upload", data={"video": (io.BytesIO(b"x" * 20_000), "clip.mp4")})
    assert response.status_code == 413
    assert "MB" in response.get_json()["error"]


def test_scaled_edge_cap():
    from processing import scaled_size

    assert scaled_size(1280, 720, 1, 480) == (480, 270)
