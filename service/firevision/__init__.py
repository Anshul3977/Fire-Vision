"""Fire-Vision serving package: ONNX Runtime fire detector + pixel heuristics."""

from .detector import Detection, FireDetector
from .pixels import fire_pixel_ratio, smoke_pixel_ratio

__all__ = ["Detection", "FireDetector", "fire_pixel_ratio", "smoke_pixel_ratio"]
