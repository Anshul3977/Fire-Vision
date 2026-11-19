"""CPU latency benchmark for the Fire-Vision ONNX models (and an optional PyTorch baseline).

    python benchmarks/latency.py --threads 1 4 --iters 200
    python benchmarks/latency.py --torch ../models/yoloNano.pt   # needs ultralytics + torch

Measures, per model and thread count, after warm-up:
  * forward: ONNX Runtime session.run on a preprocessed 640x640 tensor
  * pipeline: letterbox + forward + decode + NMS on a real 960x720 photo (ONNX only)
  * PyTorch baseline: forward only, same tensor, same thread count, Conv+BN fused.
    (Ultralytics' predict() manages its own threads and uses a rectangular letterbox,
    so it is not a like-for-like pipeline comparison.)
p50 / p95 in milliseconds, and fps = 1000 / mean pipeline latency.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from firevision.detector import MODELS_DIR, FireDetector, letterbox  # noqa: E402

SAMPLE = HERE.parent / "public" / "samples" / "forest-burn.jpg"


def pct(xs: list[float], p: float) -> float:
    return round(float(np.percentile(xs, p)), 2)


def summarise(xs: list[float]) -> dict:
    return {"p50_ms": pct(xs, 50), "p95_ms": pct(xs, 95), "mean_ms": round(statistics.fmean(xs), 2), "fps": round(1000 / statistics.fmean(xs), 1)}


def bench_onnx(model: Path, threads: int, iters: int, warmup: int) -> dict:
    det = FireDetector(model, threads=threads)
    img = Image.open(SAMPLE).convert("RGB")
    tensor, _, _ = letterbox(img, det.size)
    for _ in range(warmup):
        det.session.run(None, {det.input_name: tensor})
    fwd = []
    for _ in range(iters):
        t = time.perf_counter()
        det.session.run(None, {det.input_name: tensor})
        fwd.append((time.perf_counter() - t) * 1000)
    pipe = []
    for _ in range(iters):
        t = time.perf_counter()
        det.detect(img)
        pipe.append((time.perf_counter() - t) * 1000)
    return {"model": model.name, "backend": "onnxruntime", "imgsz": det.size, "threads": threads, "forward": summarise(fwd), "pipeline": summarise(pipe)}


def bench_torch(weights: Path, threads: int, iters: int, warmup: int, size: int = 640) -> dict:
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(threads)
    m = YOLO(str(weights))
    net = m.model.fuse(verbose=False).eval()  # Conv+BN fused, as Ultralytics does before predict
    img = Image.open(SAMPLE).convert("RGB")
    x = torch.from_numpy(letterbox(img, size)[0])
    with torch.inference_mode():
        for _ in range(warmup):
            net(x)
        fwd = []
        for _ in range(iters):
            t = time.perf_counter()
            net(x)
            fwd.append((time.perf_counter() - t) * 1000)
    return {"model": weights.name, "backend": f"pytorch {torch.__version__}", "threads": threads, "imgsz": size, "forward": summarise(fwd)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--torch", nargs="*", default=[], help="PyTorch baselines as path.pt:imgsz, e.g. ../models/yoloNano.pt:640")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "latency.json")
    a = ap.parse_args()
    env = {"cpu": platform.processor() or platform.machine(), "cores": os.cpu_count(), "python": platform.python_version(),
           "onnxruntime": __import__("onnxruntime").__version__, "iters": a.iters, "warmup": a.warmup, "image": SAMPLE.name}
    rows = []
    for model in sorted(MODELS_DIR.glob("*.onnx")):
        for t in a.threads:
            rows.append(bench_onnx(model, t, a.iters, a.warmup))
            print(json.dumps(rows[-1]), flush=True)
    for spec in a.torch:
        w, _, size = spec.partition(":")
        for t in a.threads:
            rows.append(bench_torch(Path(w), t, a.iters, a.warmup, int(size or 640)))
            print(json.dumps(rows[-1]), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"env": env, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
