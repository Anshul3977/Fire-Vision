"""Process the bundled sample and print peak RSS.

Run inside the deploy image with the free-tier memory cap:

    docker run --rm --memory=512m --memory-swap=512m --cpus=0.1 \\
        -e PORT=10000 fire-vision:ci python scripts/measure_rss.py

The script is not copied into the image by default; mount it, or run it from
a local virtualenv that has the runtime requirements installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code_web"))

from alerts import AlertGate, AlertSender  # noqa: E402
from config import SAMPLE_PATH  # noqa: E402
from jobs import Runner  # noqa: E402
from processing import peak_rss_mb, process_video  # noqa: E402


def main():
    gate = AlertGate(AlertSender())

    def emit(event, data):
        if event == "status":
            print(data, flush=True)

    stats = process_video(
        SAMPLE_PATH,
        run_motion=True,
        run_yolo_detection=True,
        size_factor=1,
        m_frames=12,
        n_frames=8,
        emit=emit,
        runner=Runner(),
        gate=gate,
        delete_after=False,
    )
    print(
        f"MEASURE frames={stats['frames']} elapsed_sec={stats['elapsed_sec']:.2f} "
        f"peak_rss_mb={max(stats['peak_rss_mb'], peak_rss_mb()):.1f} "
        f"alert_mode={gate.sender.events[0]['mode'] if gate.sender.events else 'none'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
