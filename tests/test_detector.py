import numpy as np
import pytest
from PIL import Image

from conftest import SAMPLES
from firevision.detector import MODELS_DIR, FireDetector, box_nms, letterbox, probiou, rotated_nms
from firevision.pixels import fire_mask, fire_pixel_ratio, smoke_pixel_ratio


def test_letterbox_keeps_aspect_and_pads():
    t, scale, (px, py) = letterbox(Image.new("RGB", (1280, 640), (255, 0, 0)), 640)
    assert t.shape == (1, 3, 640, 640) and t.dtype == np.float32
    assert scale == 0.5 and px == 0 and py == 160
    assert t[0, 0, 0, 0] == pytest.approx(114 / 255) and t[0, 0, 320, 320] == pytest.approx(1.0)


def test_probiou_identical_and_disjoint():
    b = np.array([100, 100, 40, 20, 0.3], np.float32)
    others = np.array([[100, 100, 40, 20, 0.3], [500, 500, 40, 20, 0.3]], np.float32)
    iou = probiou(b, others)
    assert iou[0] == pytest.approx(1.0, abs=1e-3) and iou[1] < 1e-3


def test_rotated_nms_suppresses_overlaps():
    boxes = np.array([[100, 100, 40, 20, 0], [101, 100, 40, 20, 0], [300, 300, 40, 20, 0]], np.float32)
    keep = rotated_nms(boxes, np.array([0.9, 0.8, 0.7]), 0.7)
    assert sorted(keep) == [0, 2]


def test_box_nms():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], np.float32)
    assert sorted(box_nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)) == [0, 2]


@pytest.mark.parametrize("model", sorted(p.name for p in MODELS_DIR.glob("*.onnx")))
def test_models_find_fire_in_sample_and_nothing_in_blank(model):
    det = FireDetector(MODELS_DIR / model)
    blank, timings = det.detect(Image.new("RGB", (640, 480), (40, 90, 40)))
    assert blank == [] and timings["inference_ms"] > 0
    dets, _ = det.detect(Image.open(SAMPLES / "forest-burn.jpg"))
    assert any(d.label == "fire" for d in dets)
    W, H = Image.open(SAMPLES / "forest-burn.jpg").size
    for d in dets:
        assert 0 <= d.cx <= W and 0 <= d.cy <= H and 0 < d.confidence <= 1 and len(d.polygon) == 4


def test_colour_rules():
    flame = Image.new("RGB", (10, 10), (255, 160, 20))
    grass = Image.new("RGB", (10, 10), (60, 140, 50))
    img = Image.new("RGB", (20, 10))
    img.paste(flame, (0, 0))
    img.paste(grass, (10, 0))
    m = fire_mask(img)
    assert m[:, :10].all() and not m[:, 10:].any()
    assert fire_pixel_ratio(img) == pytest.approx(0.5)
    assert smoke_pixel_ratio(Image.new("RGB", (4, 4), (200, 200, 200))) == 1.0
