"""FastAPI app for the hosted Fire-Vision demo (Vercel function, Docker or plain uvicorn).

Twilio alerts are never sent from here: responses only say whether the full
pipeline *would* alert, using the same debounce rule the video endpoint applies.
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from .detector import MODELS_DIR, FireDetector
from .pixels import fire_pixel_ratio, smoke_pixel_ratio

MAX_BYTES = 4 * 1024 * 1024  # Vercel caps request bodies at 4.5 MB
MAX_SIDE = 1280
PUBLIC = Path(__file__).resolve().parent.parent / "public"
VIDEO_TYPES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}

MODELS = {
    # key: (file, human description)
    "dfire": ("fire_smoke_dfire.onnx", "YOLOv8n fine-tuned on D-Fire: fire + smoke boxes"),
    "obb-v1": ("fire_obb.onnx", "Original YOLOv8n-OBB: rotated fire boxes"),
}
DEFAULT_MODEL = os.getenv("FIREVISION_MODEL", "dfire")


class JsonLog(logging.Formatter):
    def format(self, r: logging.LogRecord) -> str:
        return json.dumps({"level": r.levelname, "msg": r.getMessage(), **getattr(r, "fields", {})})


_h = logging.StreamHandler()
_h.setFormatter(JsonLog())
log = logging.getLogger("firevision")
log.handlers[:] = [_h]
log.setLevel(logging.INFO)
log.propagate = False

app = FastAPI(
    title="Fire-Vision API",
    version="2.0.0",
    description="Wildfire fire/smoke detection on ONNX Runtime (CPU). Images, or short clips with alert debouncing.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
if PUBLIC.exists():
    app.mount("/samples", StaticFiles(directory=PUBLIC / "samples"), name="samples")


def available_models() -> dict[str, str]:
    return {k: d for k, (f, d) in MODELS.items() if (MODELS_DIR / f).exists()}


@lru_cache(maxsize=4)
def detector(key: str) -> FireDetector:
    if key not in available_models():
        raise HTTPException(404, f"unknown model {key!r}; available: {sorted(available_models())}")
    return FireDetector(MODELS_DIR / MODELS[key][0], threads=int(os.getenv("ORT_THREADS", "0")) or None)


def _model_key(model: str | None) -> str:
    key = model or DEFAULT_MODEL
    if key not in available_models():
        key = next(iter(available_models()))
    return key


def _decode_image(data: bytes) -> Image.Image:
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (max 4 MB).")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "Could not read that file as an image.")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def analyse(img: Image.Image, det: FireDetector, conf: float) -> dict:
    dets, timings = det.detect(img, conf=conf)
    t0 = time.perf_counter()
    fire_px, smoke_px = fire_pixel_ratio(img), smoke_pixel_ratio(img)
    timings["heuristics_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    labels = {d.label for d in dets}
    return {
        "width": img.width,
        "height": img.height,
        "detections": [d.to_dict() for d in dets],
        "fire_detected": "fire" in labels,
        "smoke_detected": "smoke" in labels,
        "fire_pixel_ratio": fire_px,
        "smoke_pixel_ratio": smoke_px,
        "timings": timings,
    }


@app.get("/api/health")
def health() -> dict:
    key = _model_key(None)
    d = detector(key)
    return {"status": "ok", "default_model": key, "classes": d.names, "input_size": d.size, "runtime": "onnxruntime-cpu"}


@app.get("/api/models")
def models() -> dict:
    return {"default": _model_key(None), "models": available_models()}


@app.post("/api/detect", summary="Detect fire/smoke in one image")
async def detect(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.05, le=0.95),
    model: str | None = Query(None, description="dfire (default) or obb-v1"),
) -> dict:
    key = _model_key(model)
    img = _decode_image(await file.read(MAX_BYTES + 1))
    res = analyse(img, detector(key), conf)
    hit = res["fire_detected"] or res["smoke_detected"]
    res["model"] = key
    res["alert"] = {"would_trigger": hit, "sent": False, "reason": "alerts are disabled in the public demo"}
    log.info("detect", extra={"fields": {"model": key, "n": len(res["detections"]), "inference_ms": res["timings"]["inference_ms"]}})
    return res


@app.post("/api/detect/video", summary="Analyse a short clip frame by frame with alert debouncing")
async def detect_video(
    file: UploadFile = File(...),
    sample_fps: float = Query(4.0, gt=0, le=15),
    max_frames: int = Query(40, ge=1, le=80),
    conf: float = Query(0.25, ge=0.05, le=0.95),
    alert_after: int = Query(3, ge=1, le=20, description="Consecutive positive frames before an alert would fire"),
    model: str | None = Query(None),
) -> dict:
    import cv2

    key = _model_key(model)
    det = detector(key)
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Clip too large (max 4 MB). Trim it or lower the resolution.")
    suffix = os.path.splitext(file.filename or "")[1].lower() or ".mp4"
    if suffix not in VIDEO_TYPES:
        raise HTTPException(415, f"unsupported video type {suffix}")
    start = time.perf_counter()
    frames: list[tuple[int, Image.Image]] = []
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise HTTPException(415, "could not decode video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step, idx = max(1, round(fps / sample_fps)), 0
        while len(frames) < max_frames and cap.grab():
            if idx % step == 0:
                ok, bgr = cap.retrieve()
                if not ok:
                    break
                img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if max(img.size) > MAX_SIDE:
                    img.thumbnail((MAX_SIDE, MAX_SIDE))
                frames.append((idx, img))
            idx += 1
        cap.release()
    if not frames:
        raise HTTPException(415, "no decodable frames")

    out, streak, alert_at, inf = [], 0, None, []
    for fi, img in frames:
        r = analyse(img, det, conf)
        hit = r["fire_detected"] or r["smoke_detected"]
        streak = streak + 1 if hit else 0
        if streak >= alert_after and alert_at is None:
            alert_at = round(fi / fps, 2)
        inf.append(r["timings"]["inference_ms"])
        out.append({"frame": fi, "t": round(fi / fps, 3), **{k: r[k] for k in ("detections", "fire_detected", "smoke_detected", "fire_pixel_ratio")}})
    elapsed = time.perf_counter() - start
    positives = sum(1 for f in out if f["fire_detected"] or f["smoke_detected"])
    return {
        "model": key,
        "video": {"width": frames[0][1].width, "height": frames[0][1].height, "fps": round(fps, 2), "frames": total, "duration_s": round(total / fps, 2)},
        "frames_analysed": len(out),
        "frame_step": step,
        "positive_frames": positives,
        "alert": {"would_trigger": alert_at is not None, "first_alert_at_s": alert_at, "rule": f"{alert_after} consecutive positive frames", "sent": False},
        "frames": out,
        "timings": {
            "total_ms": round(elapsed * 1000, 1),
            "inference_ms_p50": round(float(np.percentile(inf, 50)), 2),
            "processing_fps": round(len(out) / elapsed, 2),
        },
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(PUBLIC / "index.html")
