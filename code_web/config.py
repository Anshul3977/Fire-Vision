"""Runtime settings for the free Render instance.

Read on each call so tests and the process environment can change a limit
without restarting the interpreter.
"""

from __future__ import annotations

import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
MODEL_PATH = ROOT / "models" / "yoloNano.onnx"
WEIGHTS_PATH = ROOT / "models" / "yoloNano.pt"
SAMPLE_PATH = ROOT / "samples" / "sample_fire.mp4"
UPLOAD_DIR = CODE_DIR / "static" / "uploads"

ALLOWED_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}

TWILIO_ENV_KEYS = (
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_PHONE",
    "ALERT_PHONE",
)

# The bundled checkpoint's names dict is {0: "fire"} (YOLOv8-OBB, one class).
CLASS_NAMES = {0: "fire"}


def _raw(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def env_int(name: str, default: int) -> int:
    raw = _raw(name)
    if raw is None:
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    raw = _raw(name)
    if raw is None:
        return default
    return float(raw)


def frame_stride() -> int:
    return max(1, env_int("FRAME_STRIDE", 4))


def yolo_imgsz() -> int:
    return max(32, env_int("YOLO_IMGSZ", 320))


def max_edge() -> int:
    return max(64, env_int("MAX_EDGE", 480))


def max_upload_bytes() -> int:
    return max(1, env_int("MAX_UPLOAD_BYTES", 8 * 1024 * 1024))


def max_duration_sec() -> float:
    return max(1.0, env_float("MAX_DURATION_SEC", 15.0))


def max_process_sec() -> float:
    return max(5.0, env_float("MAX_PROCESS_SEC", 150.0))


def max_video_frames() -> int:
    return max(1, env_int("MAX_VIDEO_FRAMES", 450))


def yolo_conf() -> float:
    return min(0.99, max(0.01, env_float("YOLO_CONF", 0.35)))


def yolo_iou() -> float:
    return min(0.95, max(0.05, env_float("YOLO_IOU", 0.45)))


def graph_dpi() -> int:
    return max(40, env_int("GRAPH_DPI", 80))


def jpeg_quality() -> int:
    return min(95, max(30, env_int("JPEG_QUALITY", 60)))
