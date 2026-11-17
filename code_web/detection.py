"""YOLO-OBB post-processing that does not need PyTorch.

The nano checkpoint is a one-class YOLOv8-OBB model (class 0 = fire).
Raw ONNX output is xywh + class scores + a final angle channel, in the
letterboxed network input. Rotated boxes are drawn; overlap uses the
rotated polygons.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import CLASS_NAMES


def is_fire_class(class_id, names=None) -> bool:
    table = CLASS_NAMES if names is None else names
    try:
        key = int(class_id)
    except (TypeError, ValueError):
        return False
    label = table.get(key, table.get(str(key), ""))
    return str(label).strip().lower() == "fire"


def letterbox(image, new_shape, color=(114, 114, 114)):
    """Resize with the Ultralytics letterbox padding (auto=False, stride-free)."""
    if isinstance(new_shape, int):
        new_shape = (int(new_shape), int(new_shape))
    height, width = image.shape[:2]
    ratio = min(new_shape[0] / height, new_shape[1] / width)
    new_unpad = (int(round(width * ratio)), int(round(height * ratio)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    resized = image
    if (width, height) != new_unpad:
        resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, ratio, (left, top)


def scale_xywhr(xywhr, ratio, pad):
    x, y, width, height, angle = xywhr
    left, top = pad
    return (
        (float(x) - left) / ratio,
        (float(y) - top) / ratio,
        float(width) / ratio,
        float(height) / ratio,
        float(angle),
    )


def polygon_from_xywhr(x, y, width, height, angle):
    rect = ((float(x), float(y)), (float(abs(width)), float(abs(height))), float(np.degrees(angle)))
    return np.intp(cv2.boxPoints(rect))


def rotated_iou(xywhr_a, xywhr_b) -> float:
    poly_a = cv2.boxPoints(
        (
            (float(xywhr_a[0]), float(xywhr_a[1])),
            (float(abs(xywhr_a[2])), float(abs(xywhr_a[3]))),
            float(np.degrees(xywhr_a[4])),
        )
    ).astype(np.float32)
    poly_b = cv2.boxPoints(
        (
            (float(xywhr_b[0]), float(xywhr_b[1])),
            (float(abs(xywhr_b[2])), float(abs(xywhr_b[3]))),
            float(np.degrees(xywhr_b[4])),
        )
    ).astype(np.float32)
    area, _ = cv2.intersectConvexConvex(poly_a, poly_b)
    if area <= 0:
        return 0.0
    union = abs(cv2.contourArea(poly_a)) + abs(cv2.contourArea(poly_b)) - area
    if union <= 0:
        return 0.0
    return float(area / union)


def _channels_first(pred) -> np.ndarray:
    array = np.asarray(pred, dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"expected a 2D prediction, got shape {getattr(pred, 'shape', None)}")
    channels, anchors = array.shape
    # The anchor axis is the long one. A tiny test tensor can be (6, 3).
    if anchors < 6 and channels >= 6:
        return array
    if channels > anchors and anchors >= 6 and anchors <= 64:
        return array.T
    if channels < 6:
        raise ValueError(f"expected at least 6 channels (xywh, class, angle), got {channels}")
    return array


def decode_yolo_obb(pred, conf_thres: float, iou_thres: float, max_det: int = 20) -> list[dict]:
    """Filter one OBB head output down to kept fire boxes in network-input pixels."""
    array = _channels_first(pred)
    channels, _anchors = array.shape
    class_count = channels - 5
    xywh = array[0:4]
    class_scores = array[4 : 4 + class_count]
    angles = array[-1]
    if class_count == 1:
        class_ids = np.zeros(angles.shape[0], dtype=np.int32)
        scores = class_scores[0]
    else:
        class_ids = np.argmax(class_scores, axis=0).astype(np.int32)
        scores = class_scores[class_ids, np.arange(angles.shape[0])]

    keep = scores >= conf_thres
    if not np.any(keep):
        return []
    scores = scores[keep]
    class_ids = class_ids[keep]
    xywh = xywh[:, keep]
    angles = angles[keep]

    if scores.shape[0] > 100:
        top = np.argpartition(scores, -100)[-100:]
        scores = scores[top]
        class_ids = class_ids[top]
        xywh = xywh[:, top]
        angles = angles[top]

    order = np.argsort(scores)[::-1]
    scores = scores[order]
    class_ids = class_ids[order]
    xywh = xywh[:, order]
    angles = angles[order]
    boxes = [
        (
            float(xywh[0, i]),
            float(xywh[1, i]),
            float(xywh[2, i]),
            float(xywh[3, i]),
            float(angles[i]),
        )
        for i in range(scores.shape[0])
    ]

    suppressed = [False] * len(boxes)
    selected: list[int] = []
    for index, box in enumerate(boxes):
        if suppressed[index]:
            continue
        selected.append(index)
        if len(selected) >= max_det:
            break
        for other in range(index + 1, len(boxes)):
            if suppressed[other] or class_ids[other] != class_ids[index]:
                continue
            if rotated_iou(box, boxes[other]) > iou_thres:
                suppressed[other] = True

    detections = []
    for index in selected:
        x, y, width, height, angle = boxes[index]
        detections.append(
            {
                "xywhr": (x, y, width, height, angle),
                "xyxy": (x - width / 2, y - height / 2, x + width / 2, y + height / 2),
                "score": float(scores[index]),
                "cls": int(class_ids[index]),
            }
        )
    return detections
