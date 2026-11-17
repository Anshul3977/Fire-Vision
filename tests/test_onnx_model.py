from pathlib import Path

import numpy as np
import pytest

from config import MODEL_PATH
from detection import is_fire_class

pytest.importorskip("onnxruntime")


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason="export models/yoloNano.onnx first")
def test_bundled_onnx_runs_on_a_blank_frame():
    from yolo_detection import load_model, run_yolo

    session = load_model()
    assert session.get_inputs()[0].shape[2] == 320
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    out, boxes, classes = run_yolo(frame, session=session)
    assert out.shape == frame.shape
    assert len(boxes) == len(classes)
    assert all(is_fire_class(class_id) for class_id in classes)
    assert Path(MODEL_PATH).stat().st_size > 100_000
