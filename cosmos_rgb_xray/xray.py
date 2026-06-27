"""
X-ray overlay: cutout → reproject → smooth → screen-blend onto RGB.

Two pre-processed global maps are supported:
  XRAY_LARGE  cosmos_chaxmm14_noem_520.fits  — diffuse ICM (point srcs removed)
  XRAY_SMALL  cosmos_chaxmm14_520_wv.3.fits  — compact sources (wavelet scale 3)

Per-group pre-cleaned FITS (e.g. gg15/15_large_scale.fits) take priority
over the global maps when they exist and cover the group sky position.

IMPORTANT: never fall back to the raw combined map cosmos_chaxmm14_520.fits
— it contains photon-count noise that looks like real structure when smoothed.

Colour convention:
  z < 0.3  → magenta  (0.85, 0.0, 1.0)
  z ≥ 0.3  → cyan     (0.0,  1.0, 1.0)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

from .blend import screen
from .io_fits import cutout_from_map, load_fits, reproject_to, wcs_covers

# Paths to global COSMOS X-ray maps — update for your cluster/server
XRAY_DIR = Path(
    "/Users/gozalig1/Projects/compact-groups-xray-analysis/data/xray-map"
)
XRAY_LARGE = XRAY_DIR / "cosmos_chaxmm14_noem_520.fits"   # diffuse / large
XRAY_SMALL = XRAY_DIR / "cosmos_chaxmm14_520_wv.3.fits"   # compact / wavelet


def xray_color(redshift: float) -> Tuple[float, float, float]:
    """Magenta for z<0.3 (low-z groups), cyan for z≥0.3 (higher-z)."""
    return (0.85, 0.0, 1.0) if redshift < 0.3 else (0.0, 1.0, 1.0)


def find_per_group_xray(
    group_dir: Path,
    group_id: int,
    ra: float,
    dec: float,
    verbose: bool = False,
) -> Optional[Path]:
    """
    Look for a pre-cleaned per-group X-ray FITS in the group data directory.
    Validates WCS coverage before returning — files with broken CRPIX that
    don't actually cover the group sky position are rejected.
    """
    candidates = [
        f"{group_id}_large_scale.fits",
        f"{group_id}_small_scale.fits",
        f"{group_id}_xray.fits",
        f"{group_id}_xray_large.fits",
    ]
    for name in candidates:
        p = group_dir / name
        if not p.exists():
            continue
        data, hdr = load_fits(p)
        if not np.any(data > 0):
            continue
        if not wcs_covers(hdr, ra, dec):
            if verbose:
                print(f"  {name}: WCS does not cover group — skipping", file=sys.stderr)
            continue
        if verbose:
            print(f"  Per-group X-ray found: {name}", file=sys.stderr)
        return p
    return None


def overlay_xray(
    rgb: np.ndarray,
    ref_hdr: fits.Header,
    ra: float,
    dec: float,
    redshift: float,
    radius_arcmin: float = 4.0,
    smooth_sigma: float = 60.0,
    alpha: float = 0.65,
    gamma: float = 0.55,
    pmin: float = 30.0,
    pmax: float = 99.5,
    use_small_scale: bool = False,
    per_group_xray: Optional[Path] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Reproject and screen-blend X-ray onto the RGB image.

    Parameters
    ----------
    rgb            : float HxWx3 RGB array in [0,1]
    ref_hdr        : WCS header matching the RGB pixel grid
    ra, dec        : X-ray centre (prefer RA_xray_peak / Dec_xray_peak from catalog)
    redshift       : group redshift (determines overlay colour)
    radius_arcmin  : cutout half-width when using global maps
    smooth_sigma   : Gaussian smoothing in output pixels (diffuse ICM glow)
    alpha          : maximum screen-blend opacity
    gamma          : power-law applied to normalised X-ray map (lifts faint emission)
    pmin, pmax     : percentiles for background clipping and peak normalisation
    use_small_scale: use compact/wavelet map instead of diffuse map
    per_group_xray : path to a pre-cleaned per-group FITS (overrides global maps)
    verbose        : print progress
    """
    # --- Load X-ray data ---
    if per_group_xray is not None and per_group_xray.exists():
        if verbose:
            print(f"  X-ray: using per-group file {per_group_xray.name}", file=sys.stderr)
        xdata, xhdr = load_fits(per_group_xray)
    else:
        xray_map = XRAY_SMALL if use_small_scale else XRAY_LARGE
        if not xray_map.exists():
            if verbose:
                print("  X-ray: no global map found — skipping", file=sys.stderr)
            return rgb

        xdata, xhdr = cutout_from_map(xray_map, ra, dec, radius_arcmin * 1.5)
        if not np.any(xdata > 0):
            if verbose:
                print(f"  X-ray: no signal in {xray_map.name} at this position", file=sys.stderr)
            return rgb
        if verbose:
            print(f"  X-ray: {xray_map.name}", file=sys.stderr)

    # --- Reproject onto RGB grid ---
    reproj = reproject_to(xdata, xhdr, ref_hdr)

    # --- Smooth (diffuse ICM glow) ---
    # Two-pass smoothing: fine pass preserves core structure,
    # coarse pass (3× sigma) feathers the boundary into a haze
    # that fills the whole field rather than stopping at a hard edge.
    raw = np.maximum(reproj, 0.0)
    smoothed_core = gaussian_filter(raw, sigma=smooth_sigma)
    smoothed_haze = gaussian_filter(raw, sigma=smooth_sigma * 3.0)
    # Combine: core structure + wide haze background
    smoothed = smoothed_core + 0.35 * smoothed_haze

    mask = smoothed > 0
    if not np.any(mask):
        if verbose:
            print("  X-ray: empty after reproject — skipping", file=sys.stderr)
        return rgb

    # --- Background clip + normalise ---
    noise_floor = np.percentile(smoothed[mask], pmin)
    smoothed    = np.maximum(smoothed - noise_floor, 0.0)

    peak = np.percentile(smoothed[smoothed > 0], pmax)
    # Log stretch: compresses the bright core so galaxy colours survive,
    # while lifting the faint extended haze into the purple background
    k    = 10.0  # log softness — higher = more compression in core
    norm = np.log1p(k * np.clip(smoothed / (peak + 1e-12), 0.0, 1.0)) / np.log1p(k)

    # --- Colour layer ---
    r, g, b = xray_color(redshift)
    colour_layer = np.zeros_like(rgb)
    colour_layer[..., 0] = r * norm
    colour_layer[..., 1] = g * norm
    colour_layer[..., 2] = b * norm

    # --- Screen blend ---
    blended = np.clip(screen(rgb, colour_layer * alpha), 0.0, 1.0)

    # --- Faint purple background haze in unlit corners ---
    # Where the X-ray norm is essentially zero (corners of the image),
    # add a very subtle purple tint so the background is deep indigo
    # rather than black, matching the full-field purple look of the
    # reference image.
    haze_strength = 0.08
    haze_norm = gaussian_filter(norm, sigma=smooth_sigma * 5.0)
    haze_norm = np.clip(haze_norm / (haze_norm.max() + 1e-12), 0.0, 1.0)
    corner_mask = 1.0 - haze_norm          # strongest where X-ray is faint
    purple = np.array([r * 0.4, 0.0, b * 0.6], dtype=np.float32)
    for c in range(3):
        blended[..., c] = np.clip(
            blended[..., c] + corner_mask * haze_strength * purple[c], 0.0, 1.0
        )

    return blended
