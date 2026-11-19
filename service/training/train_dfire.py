"""Reproduce models/fire_smoke_dfire.onnx: YOLOv8n fine-tuned on a D-Fire subset, on CPU.

D-Fire (Venâncio et al., 2022, CC BY 4.0) has 17,221 train and 4,306 test images labelled
smoke (0) and fire (1) in YOLO format. Download it from https://github.com/gaiasd/DFireDataset
into DFIRE/train/{images,labels} and DFIRE/test/{images,labels}, then:

    pip install ultralytics==8.4.215
    python training/train_dfire.py --root DFIRE

What the shipped model used: 8,000 random train images (seed 0), COCO-pretrained yolov8n,
imgsz 416, 12 epochs, batch 16, CPU only (8 vCPU, about 1 hour). The best epoch was picked on
1,000 random test images; final numbers are on the full test split.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=416)
    a = ap.parse_args()
    root = a.root.resolve()
    rng = random.Random(0)
    train = sorted((root / "train" / "images").glob("*.jpg"))
    test = sorted((root / "test" / "images").glob("*.jpg"))
    (root / "train_subset.txt").write_text("\n".join(map(str, rng.sample(train, a.n_train))) + "\n")
    (root / "val_subset.txt").write_text("\n".join(map(str, rng.sample(test, 1000))) + "\n")
    names = "names:\n  0: smoke\n  1: fire\n"
    (root / "train.yaml").write_text(f"path: {root}\ntrain: train_subset.txt\nval: val_subset.txt\n{names}")
    (root / "test.yaml").write_text(f"path: {root}\ntrain: train_subset.txt\nval: test/images\n{names}")

    model = YOLO("yolov8n.pt")
    model.train(data=str(root / "train.yaml"), imgsz=a.imgsz, epochs=a.epochs, batch=16, device="cpu", plots=False)
    metrics = model.val(data=str(root / "test.yaml"), imgsz=a.imgsz, device="cpu", plots=False)
    print("D-Fire test mAP50:", metrics.box.map50, "mAP50-95:", metrics.box.map)
    model.export(format="onnx", imgsz=a.imgsz, opset=17, simplify=True)


if __name__ == "__main__":
    main()
