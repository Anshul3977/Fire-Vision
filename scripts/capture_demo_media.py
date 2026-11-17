"""Save real analysis frames and a short GIF from the bundled sample."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code_web"))

from alerts import AlertGate, AlertSender  # noqa: E402
from config import SAMPLE_PATH  # noqa: E402
from jobs import Runner  # noqa: E402
from processing import process_video  # noqa: E402

OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    motion = []
    yolo = []
    graph = {"png": None}

    def emit(event, data):
        if event == "stable_update":
            motion.append(base64.b64decode(data))
        elif event == "yolo_update":
            yolo.append(base64.b64decode(data))
        elif event in {"analysis_update", "analysis_final"} and data.get("graph"):
            graph["png"] = base64.b64decode(data["graph"])

    process_video(
        SAMPLE_PATH,
        run_motion=True,
        run_yolo_detection=True,
        size_factor=1,
        m_frames=12,
        n_frames=8,
        emit=emit,
        runner=Runner(),
        gate=AlertGate(AlertSender()),
        delete_after=False,
    )
    if not motion or not yolo or not graph["png"]:
        raise SystemExit(f"missing frames motion={len(motion)} yolo={len(yolo)} graph={bool(graph['png'])}")

    mid = len(motion) // 2
    (OUT / "motion.jpg").write_bytes(motion[mid])
    (OUT / "yolo.jpg").write_bytes(yolo[mid])
    (OUT / "analysis.png").write_bytes(graph["png"])

    frames = []
    step = max(1, len(yolo) // 12)
    for blob in yolo[::step][:12]:
        image = Image.open(__import__("io").BytesIO(blob)).convert("RGB")
        image.thumbnail((480, 480))
        frames.append(image)
    frames[0].save(
        OUT / "demo.gif",
        save_all=True,
        append_images=frames[1:],
        duration=180,
        loop=0,
        optimize=True,
    )
    print(f"wrote {len(motion)} motion frames, {len(yolo)} yolo frames, gif {len(frames)}")


if __name__ == "__main__":
    main()
