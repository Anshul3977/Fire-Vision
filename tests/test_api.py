import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import SAMPLES
from firevision.app import app

client = TestClient(app)


def test_health_and_models():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    models = client.get("/api/models").json()
    assert models["default"] in models["models"]


def test_detect_sample_reports_fire_and_never_sends_alerts():
    r = client.post("/api/detect", files={"file": ("f.jpg", (SAMPLES / "forest-burn.jpg").read_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["fire_detected"] is True
    assert body["alert"]["would_trigger"] is True and body["alert"]["sent"] is False
    assert set(body["timings"]) >= {"preprocess_ms", "inference_ms", "postprocess_ms"}


@pytest.mark.parametrize("model", ["obb-v1", "dfire"])
def test_model_switch(model):
    if model not in client.get("/api/models").json()["models"]:
        pytest.skip(f"{model} not bundled")
    r = client.post(f"/api/detect?model={model}", files={"file": ("f.jpg", (SAMPLES / "forest-burn.jpg").read_bytes(), "image/jpeg")})
    assert r.status_code == 200 and r.json()["model"] == model


def test_rejects_non_images_and_huge_files():
    assert client.post("/api/detect", files={"file": ("a.txt", b"hello", "text/plain")}).status_code == 415
    big = b"0" * (4 * 1024 * 1024 + 10)
    assert client.post("/api/detect", files={"file": ("a.jpg", big, "image/jpeg")}).status_code == 413


def test_clean_image_no_alert():
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), (30, 80, 160)).save(buf, "JPEG")
    body = client.post("/api/detect", files={"file": ("b.jpg", buf.getvalue(), "image/jpeg")}).json()
    assert body["detections"] == [] and body["alert"]["would_trigger"] is False


def test_video_endpoint_debounces_alert():
    clip = (SAMPLES / "forest-burn-pan.mp4").read_bytes()
    r = client.post("/api/detect/video?sample_fps=3&alert_after=2", files={"file": ("c.mp4", clip, "video/mp4")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["video"]["fps"] == pytest.approx(30, abs=1)
    assert body["frames_analysed"] == 12 and body["frame_step"] == 10
    assert body["positive_frames"] >= 2
    assert body["alert"]["would_trigger"] is True and body["alert"]["sent"] is False
    assert body["timings"]["processing_fps"] > 0


def test_video_rejects_bad_type():
    assert client.post("/api/detect/video", files={"file": ("x.txt", b"zz", "text/plain")}).status_code == 415
