"""
Per-channel image stretching.

Strategy (webbster / PixInsight STF approach):
  1. asinh stretch with percentile clip
  2. CLAHE (adaptive histogram equalisation) for local contrast
"""
from __future__ import annotations

import numpy as np


def stretch_asinh(
    data: np.ndarray,
    plo: float = 0.1,
    phi: float = 99.8,
    beta: float = 15.0,
) -> np.ndarray:
    """
    PixInsight STF-style asinh stretch.

    Parameters
    ----------
    data : 2-D float array
    plo, phi : percentile clip limits applied to positive pixels only
    beta : asinh softening — higher = more aggressive (brighter faint features)
    """
    pos = data[data > 0]
    if pos.size == 0:
        return np.zeros_like(data, dtype=np.float64)
    vmin, vmax = np.percentile(pos, (plo, phi))
    x = np.clip((data - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    stretched = np.arcsinh(beta * x) / np.arcsinh(beta)
    return np.clip(stretched, 0.0, 1.0).astype(np.float64)


def apply_clahe(data: np.ndarray, clip_limit: float = 0.015) -> np.ndarray:
    """
    CLAHE — adaptive histogram equalisation (scikit-image).

    Lifts faint background structure without blowing out bright galaxy cores.
    clip_limit controls how much local contrast enhancement is allowed.
    """
    from skimage import exposure
    return exposure.equalize_adapthist(
        np.clip(data, 0.0, 1.0), clip_limit=clip_limit
    ).astype(np.float64)


def stretch_channel(
    data: np.ndarray,
    plo: float = 0.1,
    phi: float = 99.8,
    beta: float = 15.0,
    clahe_clip: float = 0.015,
) -> np.ndarray:
    """Full pipeline: asinh stretch → CLAHE."""
    return apply_clahe(stretch_asinh(data, plo, phi, beta), clahe_clip)
