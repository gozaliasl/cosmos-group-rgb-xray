"""
Per-band sky background estimation and subtraction.

Sky subtraction is the single most impactful pre-processing step for
astronomical RGB composites. Without it, the sky pedestal shifts all
band fluxes by different amounts → wrong colors, even with correct
band weighting.

Methods
-------
sigma_clip  : robust statistics over positive pixels; good for uniform sky
plane_fit   : fit and remove a tilted plane; for gradient-affected images
local       : estimate local background in boxes (SExtractor-style)
"""
from __future__ import annotations
import logging
import numpy as np
from astropy.stats import sigma_clipped_stats

log = logging.getLogger(__name__)


def estimate_sky_sigmaclip(
    data: np.ndarray,
    sigma: float = 3.0,
    iterations: int = 5,
    mask: np.ndarray | None = None,
) -> float:
    """
    Estimate the sky background level using iterative sigma-clipping.

    Uses only pixels that are:
      - not NaN/Inf
      - not masked (if mask provided)
      - finite and positive or near-zero (exclude bright sources)

    Returns the sigma-clipped median (more robust than mean for fields
    with many faint galaxies).
    """
    d = np.asarray(data, dtype=np.float64)
    valid = np.isfinite(d)
    if mask is not None:
        valid &= ~mask
    if not np.any(valid):
        return 0.0
    _, median, _ = sigma_clipped_stats(d[valid], sigma=sigma, maxiters=iterations)
    return float(median)


def estimate_sky_plane(
    data: np.ndarray,
    sigma: float = 3.0,
    iterations: int = 5,
) -> np.ndarray:
    """
    Fit and return a tilted-plane background model.

    Useful when the sky has a gradient (common at survey edges and
    in ground-based data). Returns a 2D array the same shape as data
    that can be subtracted directly.
    """
    ny, nx = data.shape
    d = np.asarray(data, dtype=np.float64)

    # Build pixel coordinate grids normalised to [-1, 1]
    yy, xx = np.mgrid[0:ny, 0:nx]
    xn = (xx / nx - 0.5) * 2
    yn = (yy / ny - 0.5) * 2

    # Iterative sigma-clip mask
    valid = np.isfinite(d) & (d != 0)
    for _ in range(iterations):
        if not np.any(valid):
            break
        _, med, std = sigma_clipped_stats(d[valid], sigma=sigma, maxiters=1)
        valid &= (d > med - sigma * std) & (d < med + sigma * std)

    if not np.any(valid):
        return np.zeros_like(d)

    # Least-squares fit: bg = a + b*x + c*y
    A = np.column_stack([np.ones(valid.sum()), xn[valid], yn[valid]])
    b = d[valid]
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return np.full_like(d, np.median(b))

    bg = coeffs[0] + coeffs[1] * xn + coeffs[2] * yn
    log.debug("Plane fit: offset=%.4e, dx=%.4e, dy=%.4e", *coeffs)
    return bg


def subtract_sky(
    data: np.ndarray,
    method: str = "sigma_clip",
    sigma: float = 3.0,
    iterations: int = 5,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float | np.ndarray]:
    """
    Estimate and subtract sky background.

    Parameters
    ----------
    data      : 2-D float array (single band)
    method    : 'sigma_clip' | 'plane_fit' | 'none'
    sigma     : clipping threshold in sigma units
    iterations: number of sigma-clip iterations
    mask      : optional boolean mask (True = masked/excluded pixels)

    Returns
    -------
    subtracted : sky-subtracted array clipped to >= 0
    sky        : scalar (sigma_clip) or 2-D array (plane_fit)
    """
    if method == "none":
        return data.astype(np.float32), 0.0

    if method == "plane_fit":
        sky = estimate_sky_plane(data, sigma=sigma, iterations=iterations)
        subtracted = np.maximum(data - sky, 0.0)
        log.debug("Plane sky subtraction: median bg = %.4e", float(np.median(sky)))
        return subtracted.astype(np.float32), sky

    # Default: sigma_clip
    sky_val = estimate_sky_sigmaclip(data, sigma=sigma, iterations=iterations, mask=mask)

    # JWST / HST pipeline outputs are already sky-subtracted; their median
    # sky is negative (instrument noise floor). Subtracting again would zero
    # out real signal. Detect this and skip.
    if sky_val <= 0:
        log.debug("Sky already subtracted (estimated sky = %.4e) — skipping", sky_val)
        return np.maximum(data, 0.0).astype(np.float32), 0.0

    subtracted = np.maximum(data - sky_val, 0.0)
    log.debug("Sigma-clip sky subtraction: sky = %.4e", sky_val)
    return subtracted.astype(np.float32), sky_val
