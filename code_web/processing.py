"""Video loop: color/flow analysis and the YOLO overlay, one pass.

The original server decoded the file twice and joined both threads inside
the upload request. On a 512 MB instance that holds the only worker until
the browser times out. This pass emits frames while the request has already
returned.
"""

from __future__ import annotations

import base64
import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np

import config
from alerts import AlertGate
from analysis import graph
from detection import is_fire_class
from fire_flow import fire_flow, fire_pixel_segmentation, spread_direction_degrees
from jobs import Runner
from smoke_flow import smoke_flow, smoke_pixel_segmentation
from yolo_detection import run_yolo

logger = logging.getLogger("firevision.processing")


class VideoRejected(ValueError):
    pass


def scaled_size(width, height, size_factor, edge_limit):
    try:
        factor = float(size_factor)
    except (TypeError, ValueError):
        factor = 1.0
    if not math.isfinite(factor) or factor <= 0:
        factor = 1.0
    factor = min(4.0, max(0.25, factor))
    new_w = max(1, int(round(width / factor)))
    new_h = max(1, int(round(height / factor)))
    edge = max(new_w, new_h)
    if edge > edge_limit:
        scale = edge_limit / edge
        new_w = max(1, int(round(new_w * scale)))
        new_h = max(1, int(round(new_h * scale)))
    return new_w, new_h


def validate_upload(path: Path) -> None:
    size = path.stat().st_size
    if size <= 0:
        raise VideoRejected("The video file is empty.")
    limit = config.max_upload_bytes()
    if size > limit:
        raise VideoRejected(
            f"Video is {size / (1024 * 1024):.1f} MB. The limit is {limit // (1024 * 1024)} MB."
        )
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoRejected("Could not open that video. Use a short MP4 or WebM.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if fps > 1 and frames > 0:
        duration = frames / fps
        if duration > config.max_duration_sec() + 0.05:
            limit = config.max_duration_sec()
            raise VideoRejected(
                f"Video is {duration:.1f}s. The free instance accepts up to {limit:.0f}s."
            )


def _jpeg_b64(image) -> str | None:
    ok, buffer = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), config.jpeg_quality()],
    )
    if not ok:
        return None
    return base64.b64encode(buffer).decode("ascii")


def _sane_fps(value) -> float:
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return 30.0
    if not math.isfinite(fps) or fps < 1 or fps > 240:
        return 30.0
    return fps


def peak_rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return 0.0
    return 0.0


def _alert_text(result: dict) -> str:
    if result.get("mode") == "demo":
        return "Demo alert (Twilio is off). This fire would have been sent once:\n" + result.get(
            "sms", ""
        )
    if result.get("mode") == "live":
        return "Twilio SMS and call sent once for this fire."
    return "Alert attempt failed once and will not be retried. " + result.get("error", "")


def _emit_graph(emit, event, times, areas, smoke_dir, path_pts):
    graph_str = graph(times, areas, smoke_dir, path_pts, dpi=config.graph_dpi())
    if graph_str is None:
        return
    total_area = float(areas[-1])
    area_increase = float(areas[-1] - areas[-2]) if len(areas) > 2 else 0.0
    payload = {"graph": graph_str, "total_area": total_area}
    if event == "analysis_update":
        payload["area_increase"] = area_increase
    emit(event, payload)


def process_video(
    video_path,
    *,
    run_motion,
    run_yolo_detection,
    size_factor,
    m_frames,
    n_frames,
    emit,
    runner: Runner,
    gate: AlertGate,
    delete_after: bool = False,
    yolo_runner=None,
):
    cv2.setNumThreads(1)
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    path = Path(video_path)
    started = time.perf_counter()
    stats = {"frames": 0, "fired_alert": False, "peak_rss_mb": 0.0, "elapsed_sec": 0.0}
    cap = cv2.VideoCapture(str(path))
    yolo_fn = yolo_runner or run_yolo
    try:
        if not cap.isOpened():
            emit("status", "Could not open the video.")
            return stats
        fps = _sane_fps(cap.get(cv2.CAP_PROP_FPS))
        ok, first = cap.read()
        if not ok or first is None:
            emit("status", "Could not read the first frame.")
            return stats

        height, width = first.shape[:2]
        new_w, new_h = scaled_size(width, height, size_factor, config.max_edge())
        first = cv2.resize(first, (new_w, new_h), interpolation=cv2.INTER_AREA)

        areas = [0.0]
        times = [0.0]
        area_frame = np.zeros_like(first)
        fire_x: list[float] = []
        fire_y: list[float] = []
        path_pts = []
        prev = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        wind_dir: list[float] = []
        smoke_dir: list[float] = []
        smoke_seen = False
        index = 0
        last_graph = 0
        graph_every = max(1, int(round(fps * 5)))
        stride = config.frame_stride()
        deadline = started + config.max_process_sec()
        frame_cap = config.max_video_frames()
        emit(
            "status",
            (
                f"Processing at {new_w}x{new_h}, every {stride} frame(s), "
                f"YOLO input {config.yolo_imgsz()}. Free-tier pace, not 30 fps."
            ),
        )

        while cap.isOpened() and not runner.stopped():
            if time.perf_counter() > deadline or index >= frame_cap:
                emit("status", "Stopped at the free-tier time or frame cap. Shorter clips finish.")
                break
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            if index % stride != 0:
                continue

            im = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            processed = im.copy()
            fire_now = False

            if run_motion:
                fire_mask = fire_pixel_segmentation(im)
                area, area_frame, processed, endpoint = fire_flow(
                    fire_mask, area_frame, fire_x, fire_y, processed, m_frames
                )
                if endpoint is not None:
                    path_pts.append(endpoint)
                areas.append(float(area))
                times.append(index / fps)
                if area > 0:
                    fire_now = True

                smoke_mask = smoke_pixel_segmentation(im)
                prev, processed, smoke = smoke_flow(
                    im, prev, smoke_mask, wind_dir, processed, n_frames
                )
                if smoke is not None:
                    smoke_seen = True
                    smoke_dir.append(float(smoke))

                if len(path_pts) > 2:
                    start_point = path_pts[0]
                    end_point = path_pts[-1]
                    cv2.arrowedLine(
                        processed, start_point, end_point, (0, 255, 0), 2, tipLength=0.5
                    )
                    angle_deg = spread_direction_degrees(start_point, end_point)
                    text_position = (start_point[0] + 10, max(start_point[1] - 10, 12))
                    cv2.putText(
                        processed,
                        f"Fire Spread Direction: {angle_deg:.2f}",
                        text_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

                fire_bgr = cv2.cvtColor(fire_mask, cv2.COLOR_GRAY2BGR)
                smoke_bgr = cv2.cvtColor(smoke_mask, cv2.COLOR_GRAY2BGR)
                row1 = np.hstack((im, fire_bgr))
                row2 = np.hstack((smoke_bgr, processed))
                mosaic = np.vstack((row1, row2))
                encoded = _jpeg_b64(mosaic)
                if encoded:
                    emit("stable_update", encoded)

                if index - last_graph >= graph_every and len(areas) > 1:
                    _emit_graph(emit, "analysis_update", times, areas, smoke_dir, path_pts)
                    last_graph = index

            if run_yolo_detection:
                try:
                    yolo_frame, _boxes, classes = yolo_fn(im.copy())
                except Exception as exc:
                    logger.exception("yolo inference failed")
                    emit("status", f"Detector error: {exc}")
                    yolo_frame, classes = im, []
                if any(is_fire_class(class_id) for class_id in classes):
                    fire_now = True
                encoded = _jpeg_b64(yolo_frame)
                if encoded:
                    emit("yolo_update", encoded)

            result = gate.consider(fire_now, smoke_seen)
            if result:
                stats["fired_alert"] = True
                emit("status", _alert_text(result))
            stats["frames"] += 1

        if run_motion and len(areas) > 1:
            _emit_graph(emit, "analysis_final", times, areas, smoke_dir, path_pts)

        elapsed = time.perf_counter() - started
        stats["elapsed_sec"] = elapsed
        stats["peak_rss_mb"] = peak_rss_mb()
        rate = stats["frames"] / elapsed if elapsed else 0.0
        emit(
            "status",
            (
                f"Done. {stats['frames']} processed frames in {elapsed:.1f}s "
                f"({rate:.2f} fps on this CPU). Peak RSS {stats['peak_rss_mb']:.0f} MB."
            ),
        )
        return stats
    finally:
        cap.release()
        if delete_after:
            try:
                path.unlink()
            except OSError:
                logger.warning("could not delete %s", path)
