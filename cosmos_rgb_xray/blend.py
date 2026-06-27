"""
Blend modes used for layer compositing.

screen  — used for X-ray overlay (preserves galaxy colours)
overlay — used for luminance (F150W) layer over RGB
"""
from __future__ import annotations
import numpy as np


def screen(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Screen blend: 1 - (1-a)(1-b).  Both inputs clipped to [0,1]."""
    return 1.0 - (1.0 - np.clip(a, 0, 1)) * (1.0 - np.clip(b, 0, 1))


def overlay_channel(
    base: np.ndarray,
    layer: np.ndarray,
    opacity: float = 1.0,
) -> np.ndarray:
    """
    Photoshop OVERLAY blend for a single channel.

    dark pixels multiply, bright pixels screen — boosts local contrast.
    opacity mixes the blended result back with the base.
    """
    base  = np.clip(base,  0, 1)
    layer = np.clip(layer, 0, 1)
    blended = np.where(
        layer < 0.5,
        2.0 * base * layer,
        1.0 - 2.0 * (1.0 - base) * (1.0 - layer),
    )
    return np.clip(base * (1.0 - opacity) + blended * opacity, 0.0, 1.0)
