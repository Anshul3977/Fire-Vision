"""Export models/yoloNano.pt to ONNX for the CPU-only web service.

PyTorch and Ultralytics are not installed in the Render image. Run this once
when the checkpoint changes:

    pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu
    pip install ultralytics==8.2.80 --no-deps
    pip install pyyaml tqdm psutil scipy ultralytics-thop pandas seaborn requests
    python scripts/export_onnx.py

The training stack is AGPL-3.0. See NOTICE.
"""

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "yoloNano.pt"


def main():
    model = YOLO(str(WEIGHTS))
    exported = model.export(format="onnx", imgsz=320, opset=12, dynamic=False)
    print(exported)


if __name__ == "__main__":
    main()
