"""ONNX Runtime inference for the bundled YOLOv8-OBB nano weights.

PyTorch stays out of the Render image. The .pt file is still in models/;
models/yoloNano.onnx is that same checkpoint exported at imgsz 320.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

import config
from detection import decode_yolo_obb, letterbox, polygon_from_xywhr, scale_xywhr

_session = None
_session_lock = threading.Lock()


def load_model():
    global _session
    if _session is not None:
        return _session
    with _session_lock:
        if _session is None:
            if not config.MODEL_PATH.is_file():
                raise FileNotFoundError(
                    f"Missing {config.MODEL_PATH.name}. Export it with scripts/export_onnx.py."
                )
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.enable_cpu_mem_arena = False
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            _session = ort.InferenceSession(
                str(config.MODEL_PATH),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
    return _session


def session_imgsz(session) -> int:
    shape = session.get_inputs()[0].shape
    if len(shape) >= 4 and isinstance(shape[2], int):
        return int(shape[2])
    return config.yolo_imgsz()


def run_yolo(frame, session=None, conf=None, iou=None):
    global_model = session or load_model()
    imgsz = session_imgsz(global_model)
    letter, ratio, pad = letterbox(frame, imgsz)
    rgb = cv2.cvtColor(letter, cv2.COLOR_BGR2RGB)
    blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None].astype(np.float32) / 255.0)
    input_name = global_model.get_inputs()[0].name
    pred = global_model.run(None, {input_name: blob})[0]
    detections = decode_yolo_obb(
        pred,
        config.yolo_conf() if conf is None else conf,
        config.yolo_iou() if iou is None else iou,
    )

    pred_boxes = []
    pred_classes = []
    for det in detections:
        x, y, width, height, angle = scale_xywhr(det["xywhr"], ratio, pad)
        if width <= 1 or height <= 1:
            continue
        box = polygon_from_xywhr(x, y, width, height, angle)
        cv2.polylines(frame, [box], True, (0, 0, 255), 3)
        label_x = int(max(box[:, 0].min(), 0))
        label_y = int(max(box[:, 1].min() - 6, 12))
        cv2.putText(
            frame,
            f"fire {det['score']:.2f}",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        pred_boxes.append([x - width / 2, y - height / 2, x + width / 2, y + height / 2])
        pred_classes.append(int(det["cls"]))

    cv2.putText(
        frame,
        str(len(pred_classes)),
        (100, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 255),
        4,
        cv2.LINE_AA,
    )
    return frame, pred_boxes, pred_classes
