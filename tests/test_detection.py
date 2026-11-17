import numpy as np
import pytest

from detection import decode_yolo_obb, is_fire_class, letterbox, scale_xywhr


def test_letterbox_640x360_to_320():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    out, ratio, pad = letterbox(image, 320)
    assert out.shape == (320, 320, 3)
    assert ratio == pytest.approx(0.5)
    assert pad == (0, 70)


def test_scale_xywhr_maps_letterbox_center_to_image_center():
    x, y, width, height, angle = scale_xywhr((160, 160, 20, 10, 0.2), 0.5, (0, 70))
    assert (x, y) == pytest.approx((320, 180))
    assert width == pytest.approx(40)
    assert height == pytest.approx(20)
    assert angle == pytest.approx(0.2)


def test_is_fire_class_matches_the_checkpoint_names():
    assert is_fire_class(0)
    assert not is_fire_class(1)
    assert is_fire_class(1, {1: "Fire"})
    assert not is_fire_class("nope")


def test_decode_keeps_one_fire_box_and_suppresses_overlap():
    pred = np.zeros((1, 6, 3), dtype=np.float32)
    pred[0, :, 0] = [10, 10, 4, 4, 0.90, 0.0]
    pred[0, :, 1] = [80, 80, 4, 4, 0.10, 0.0]
    pred[0, :, 2] = [10.2, 10.2, 4, 4, 0.50, 0.0]
    kept = decode_yolo_obb(pred, conf_thres=0.35, iou_thres=0.45)
    assert len(kept) == 1
    assert kept[0]["score"] == pytest.approx(0.9)
    assert kept[0]["cls"] == 0
    assert kept[0]["xyxy"][0] == pytest.approx(8)


def test_decode_rejects_a_short_channel_axis():
    with pytest.raises(ValueError):
        decode_yolo_obb(np.zeros((1, 4, 8), dtype=np.float32), 0.35, 0.45)
