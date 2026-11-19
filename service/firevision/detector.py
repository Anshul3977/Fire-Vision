"""Fire / smoke detectors running on ONNX Runtime (CPU, no PyTorch at serve time).

Two YOLOv8 head layouts are supported and picked from the ONNX output shape:

* axis-aligned detect head, (1, 4 + nc, N): cx, cy, w, h, class scores.
  Used by `fire_smoke_dfire.onnx`, YOLOv8n fine-tuned on D-Fire (smoke, fire).
* oriented-box (OBB) head, (1, 4 + nc + 1, N): ... plus an angle in radians.
  Used by `fire_obb.onnx`, the original `models/yoloNano.pt` (one class, fire).
"""

from __future__ import annotations

import ast
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL = MODELS_DIR / "fire_obb.onnx"


@dataclass
class Detection:
    label: str
    confidence: float
    cx: float
    cy: float
    w: float
    h: float
    angle_deg: float
    polygon: list[list[float]]  # 4 corner points in original image pixels

    def to_dict(self) -> dict:
        return asdict(self)


def letterbox(img: Image.Image, size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to size x size with grey (114)."""
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad = ((size - nw) // 2, (size - nh) // 2)
    canvas.paste(resized, pad)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None], scale, pad


def _covariance(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w2, h2 = boxes[:, 2] ** 2 / 12, boxes[:, 3] ** 2 / 12
    cos, sin = np.cos(boxes[:, 4]), np.sin(boxes[:, 4])
    a = w2 * cos**2 + h2 * sin**2
    b = w2 * sin**2 + h2 * cos**2
    c = (w2 - h2) * cos * sin
    return a, b, c


def probiou(box: np.ndarray, others: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """ProbIoU between one rotated box and many (same metric Ultralytics uses for OBB NMS)."""
    a1, b1, c1 = (v[0] for v in _covariance(box[None]))
    a2, b2, c2 = _covariance(others)
    x1, y1 = box[0], box[1]
    x2, y2 = others[:, 0], others[:, 1]
    denom = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps
    t1 = ((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2) / denom * 0.25
    t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / denom * 0.5
    det1 = np.clip(a1 * b1 - c1**2, 0, None)
    det2 = np.clip(a2 * b2 - c2**2, 0, None)
    t3 = np.log(((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2) / (4 * np.sqrt(det1 * det2) + eps) + eps) * 0.5
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1 - hd


def rotated_nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """Matrix ("fast") NMS with ProbIoU, matching ultralytics.utils.ops.nms_rotated:
    a box is dropped if any higher-scoring box overlaps it above the threshold."""
    order = scores.argsort()[::-1]
    b = boxes[order]
    n = len(b)
    ious = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        ious[i, i + 1 :] = probiou(b[i], b[i + 1 :])
    keep = ious.max(axis=0) < iou_thres if n > 1 else np.ones(1, bool)
    return [int(i) for i in order[keep]]


def obb_polygon(cx: float, cy: float, w: float, h: float, angle: float) -> list[list[float]]:
    cos, sin = math.cos(angle), math.sin(angle)
    pts = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        pts.append([round(cx + dx * cos - dy * sin, 1), round(cy + dx * sin + dy * cos, 1)])
    return pts


def box_nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """Greedy axis-aligned NMS on xyxy boxes."""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        order = rest[inter / (area(boxes[i]) + area(boxes[rest]) - inter + 1e-9) <= iou_thres]
    return keep


class FireDetector:
    def __init__(self, model_path: str | os.PathLike = DEFAULT_MODEL, threads: int | None = None):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.size = int(self.session.get_inputs()[0].shape[-1])
        self.max_candidates = 300
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names: dict[int, str] = ast.literal_eval(meta["names"]) if "names" in meta else {0: "fire"}
        self.task = meta.get("task", "obb" if self.session.get_outputs()[0].shape[1] == len(self.names) + 5 else "detect")
        self.name = Path(model_path).stem

    def detect(self, img: Image.Image, conf: float = 0.25, iou: float = 0.7) -> tuple[list[Detection], dict]:
        img = img.convert("RGB")
        t0 = time.perf_counter()
        tensor, scale, (px, py) = letterbox(img, self.size)
        t1 = time.perf_counter()
        out = self.session.run(None, {self.input_name: tensor})[0][0].T  # (anchors, 4 + nc [+ 1])
        t2 = time.perf_counter()
        nc = len(self.names)
        scores_all = out[:, 4 : 4 + nc]
        cls = scores_all.argmax(1)
        scores = scores_all[np.arange(len(cls)), cls]
        m = scores > conf
        out, scores, cls = out[m], scores[m], cls[m]
        if len(out) > self.max_candidates:  # bound the O(n^2) NMS work
            top = np.argsort(scores)[::-1][: self.max_candidates]
            out, scores, cls = out[top], scores[top], cls[top]
        dets: list[Detection] = []
        if len(out):
            if self.task == "obb":
                rboxes = np.concatenate([out[:, :4], out[:, 4 + nc : 5 + nc]], axis=1)
                rboxes[:, :2] += cls[:, None] * 7680.0  # class-aware
                keep = rotated_nms(rboxes, scores, iou)
                rboxes[:, :2] -= cls[:, None] * 7680.0
            else:
                xyxy = np.concatenate([out[:, :2] - out[:, 2:4] / 2, out[:, :2] + out[:, 2:4] / 2], axis=1)
                keep = box_nms(xyxy + cls[:, None] * 7680.0, scores, iou)
                rboxes = np.concatenate([out[:, :4], np.zeros((len(out), 1), np.float32)], axis=1)
            W, H = img.size
            for i in keep:
                cx, cy, w, h, ang = (float(v) for v in rboxes[i])
                cx, cy = float(np.clip((cx - px) / scale, 0, W)), float(np.clip((cy - py) / scale, 0, H))
                w, h = w / scale, h / scale
                dets.append(
                    Detection(
                        label=self.names.get(int(cls[i]), str(int(cls[i]))),
                        confidence=round(float(scores[i]), 4),
                        cx=round(cx, 1),
                        cy=round(cy, 1),
                        w=round(w, 1),
                        h=round(h, 1),
                        angle_deg=round(math.degrees(ang), 1),
                        polygon=obb_polygon(cx, cy, w, h, ang),
                    )
                )
        t3 = time.perf_counter()
        timings = {
            "preprocess_ms": round((t1 - t0) * 1000, 2),
            "inference_ms": round((t2 - t1) * 1000, 2),
            "postprocess_ms": round((t3 - t2) * 1000, 2),
            "total_ms": round((t3 - t0) * 1000, 2),
        }
        return dets, timings
