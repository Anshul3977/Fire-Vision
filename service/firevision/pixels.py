"""Colour-rule heuristics ported from code_web/fire_flow.py and smoke_flow.py.

Pure NumPy + Pillow so the hosted demo does not need OpenCV. These are cheap
secondary signals shown next to the YOLO detections, not a replacement for them.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def fire_mask(img: Image.Image) -> np.ndarray:
    """Seven-rule fire pixel segmentation (Celik & Demirel) on RGB + YCbCr."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.int16)
    ycc = np.asarray(img.convert("YCbCr"), dtype=np.int16)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y, Cb, Cr = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    m = (R > G) & (G > B)
    m &= (R > 190) & (G > 100) & (B < 140)
    m &= (Y >= Cb) & (Cr >= Cb)
    m &= (Y >= Y.mean()) & (Cb <= Cb.mean()) & (Cr >= Cr.mean())
    m &= np.abs(Cb - Cr) >= 70
    m &= (Cb <= 120) & (Cr >= 150)
    return m


def smoke_mask(img: Image.Image) -> np.ndarray:
    """Low/very-high saturation pixels (inverse of S in [50, 200]), as in smoke_flow.py."""
    s = np.asarray(img.convert("HSV"), dtype=np.uint8)[..., 1]
    return ~((s >= 50) & (s <= 200))


def fire_pixel_ratio(img: Image.Image) -> float:
    return round(float(fire_mask(img).mean()), 4)


def smoke_pixel_ratio(img: Image.Image) -> float:
    return round(float(smoke_mask(img).mean()), 4)
