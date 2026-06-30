"""
Image stretching algorithms for astronomical visualization.

Methods implemented:
  GHS   — Generalized Hyperbolic Stretch (Tomlinson & Rusbel 2021)
           State-of-the-art in PixInsight; better faint-structure visibility
           than asinh while preventing saturation of bright galaxy cores.
  MTF   — Midtone Transfer Function (PixInsight STF-style)
           Simple and fast; good for quick previews.
  asinh — Lupton et al. (2004) softening stretch
           Classic; used in SDSS, HST; slightly inferior to GHS for JWST depth.

References
----------
Tomlinson & Rusbel (2021) — "Generalized Hyperbolic Stretching"
  https://ghsastro.co.uk/
Lupton et al. (2004) — "Preparing Red-Green-Blue Images from CCD Data"
  PASP 116, 133
"""
from __future__ import annotations
import logging
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generalized Hyperbolic Stretch
# ---------------------------------------------------------------------------

def ghs(
    x: np.ndarray,
    b: float = 6.0,
    D: float = 1.0,
    SP_pct: float = 1.0,
    HP_pct: float = 99.9,
    LP: float = 0.15,
) -> np.ndarray:
    """
    Generalized Hyperbolic Stretch.

    Parameters
    ----------
    x       : 2-D float array, arbitrary units (will be normalised internally)
    b       : stretch intensity, 1–20 typical (larger = more aggressive)
    D       : local stretch at inflection point (0 = no local contrast, 1 = moderate)
    SP_pct  : shadow protection percentile — pixels below this level are
              linearly mapped (sky + noise preserved, not crushed to black)
    HP_pct  : highlight protection percentile — prevents saturation of bright cores
    LP      : linear/inflection point in [0,1] after SP/HP normalisation;
              this is where maximum stretch is applied (typical: 0.1–0.2)

    Returns
    -------
    Float32 array in [0, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    pos = x[x > 0]
    if pos.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    SP = float(np.percentile(pos, SP_pct))
    HP = float(np.percentile(pos, HP_pct))
    span = HP - SP
    if span < 1e-15:
        return np.zeros_like(x, dtype=np.float32)

    # Normalise to [0, 1] relative to SP … HP
    xn = np.clip((x - SP) / span, 0.0, 1.0)

    # GHS core transform
    # q = arcsinh(D) = log(D + sqrt(D^2 + 1))
    q = float(np.log(D + np.sqrt(D * D + 1.0))) if D > 0 else 0.0

    if q < 1e-10 or b < 1e-10:
        # Linear fallback (no stretch)
        return xn.astype(np.float32)

    bq = b * q
    # Denominator: value at xn = 1 (HP point)
    denom = np.sinh(bq * (1.0 - LP)) + np.sinh(bq * LP)
    if abs(denom) < 1e-15:
        return xn.astype(np.float32)

    numerator = np.sinh(bq * (xn - LP)) + np.sinh(bq * LP)
    stretched = np.clip(numerator / denom, 0.0, 1.0)

    log.debug("GHS: SP=%.3e HP=%.3e b=%.1f D=%.1f LP=%.2f", SP, HP, b, D, LP)
    return stretched.astype(np.float32)


# ---------------------------------------------------------------------------
# Midtone Transfer Function (PixInsight STF approximation)
# ---------------------------------------------------------------------------

def mtf(x: np.ndarray, m: float = 0.25) -> np.ndarray:
    """
    PixInsight Midtone Transfer Function.

    Maps x such that the midtone value m → 0.5 in the output.
    Simple, fast, good for quick previews.

    m : midtone (0–1); lower = brighter image
    """
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    if abs(m) < 1e-10:
        return x.astype(np.float32)
    out = ((m - 1.0) * x) / ((2.0 * m - 1.0) * x - m)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Classic asinh (Lupton 2004)
# ---------------------------------------------------------------------------

def asinh_stretch(
    x: np.ndarray,
    plo: float = 0.5,
    phi: float = 99.5,
    beta: float = 10.0,
) -> np.ndarray:
    """
    Lupton et al. (2004) softened asinh stretch.
    Kept for reference / comparison with v1.
    """
    x = np.asarray(x, dtype=np.float64)
    pos = x[x > 0]
    if pos.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    vmin = float(np.percentile(pos, plo))
    vmax = float(np.percentile(pos, phi))
    xn = np.clip((x - vmin) / max(vmax - vmin, 1e-15), 0.0, 1.0)
    out = np.arcsinh(beta * xn) / np.arcsinh(beta)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Normalise a single band to [0,1] before stretching
# ---------------------------------------------------------------------------

def percentile_norm(
    x: np.ndarray,
    plo: float = 0.5,
    phi: float = 99.5,
) -> np.ndarray:
    """
    Robust linear normalisation using positive-pixel percentiles.
    Returns float32 in [0,1], safe to pass to any stretch function.
    """
    x = np.asarray(x, dtype=np.float64)
    pos = x[x > 0]
    if pos.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo = float(np.percentile(pos, plo))
    hi = float(np.percentile(pos, phi))
    return np.clip((x - lo) / max(hi - lo, 1e-15), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Tone curve applied after stretch
# ---------------------------------------------------------------------------

def tone_curve(
    x: np.ndarray,
    gamma: float = 0.85,
    sky_floor: float = 0.02,
) -> np.ndarray:
    """
    Apply a power-law tone curve above the sky floor.

    x^gamma for x > sky_floor (gamma < 1 brightens midtones).
    Below sky_floor, linear to preserve noise structure.
    """
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    above = x > sky_floor
    out = x.copy()
    out[above] = sky_floor + (x[above] - sky_floor) ** gamma
    return np.clip(out, 0.0, 1.0)
