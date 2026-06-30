"""
V2 X-ray overlay — improved alpha falloff and sharper outer boundary.

Key improvements over v1:
  - Quadratic alpha ramp: alpha ∝ norm² so faint outer emission is
    nearly transparent instead of a dark brownish halo
  - Hard background cutoff: emission below bg_floor is set to alpha=0
  - Reuses v1 smoothing, hole-filling, and contour logic
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from cosmos_rgb_xray.xray import (
    xray_color,
    _largest_component_mask,
    find_per_group_xray,
    optical_coverage_mask,
    hst_background_fill,
)
from cosmos_rgb_xray.io_fits import reproject_to, cutout_from_map

log = logging.getLogger(__name__)

# Full-survey X-ray maps (fallback when no per-group file exists)
_XRAY_DIR = Path("/n23data2/gozaliasl/xray_maps")
_LARGE_MAP = _XRAY_DIR / "cosmos_chaxmm14_520.fits"
_SMALL_MAP = _XRAY_DIR / "cosmos_chaxmm14_520_wv.3.fits"


def _xray_cmap_v2(r: float, g: float, b: float) -> LinearSegmentedColormap:
    """
    V2 colormap: outer glow is pale/white so it reads as luminous on black.

    Faint outer emission → pale pink (near-white) at very low alpha so it
    glows rather than creating a dark brownish tint on the black background.
    Inner emission transitions to the saturated halo color.
    """
    # Pale outer glow: blend the halo color toward white
    rp = r * 0.5 + 0.5   # pale R
    gp = g * 0.5 + 0.5   # pale G
    bp = b * 0.5 + 0.5   # pale B
    colors_rgba = [
        (0.0, 0.0, 0.0, 0.00),    # norm=0.00 → fully transparent
        (rp,  gp,  bp,  0.00),    # norm=0.20 → transparent (cutoff handles 0–0.20)
        (rp,  gp,  bp,  0.08),    # norm=0.30 → pale pink glow (reads bright not dark)
        (r,   g,   b,   0.28),    # norm=0.50 → main halo color
        (r,   g,   b,   0.65),    # norm=0.75 → deep halo
        (r*0.85+0.1, g, b, 1.00), # norm=1.00 → bright center
    ]
    nodes = [0.0, 0.20, 0.30, 0.50, 0.75, 1.00]
    return LinearSegmentedColormap.from_list(
        "xray_v2",
        list(zip(nodes, colors_rgba)),
    )


def overlay_xray_v2(
    rgb: np.ndarray,
    ref_hdr: fits.Header,
    ra: float,
    dec: float,
    redshift: float = 0.0,
    per_group_xray: Optional[Path] = None,
    use_small_scale: bool = False,
    smooth_sigma: float = 160.0,
    smooth_haze_sigma: float = 200.0,
    smooth_haze_weight: float = 0.10,
    radius_arcmin: float = 4.0,
    alpha_peak: float = 0.32,
    norm_power: float = 1.0,
    noise_floor_pct: float = 42.0,
    show_contours: bool = True,
    contour_levels: Sequence[float] = (0.20, 0.38, 0.62, 0.85),
    contour_linewidths: Sequence[float] = (0.7, 0.9, 1.1, 1.3),
    contour_alpha: float = 0.85,
    coverage: Optional[np.ndarray] = None,
    ax=None,
    verbose: bool = False,
) -> np.ndarray:
    """
    V2 X-ray overlay with clean outer boundary.

    Identical to v1 overlay_xray but uses _xray_cmap_v2 which has a
    quadratic alpha ramp — faint outer emission stays transparent.
    """
    rgb = np.clip(rgb, 0, 1).astype(np.float32)

    # Coverage mask
    if coverage is None:
        from cosmos_rgb_xray.xray import optical_coverage_mask
        coverage = optical_coverage_mask(rgb)
    cov = coverage.astype(np.float32)

    # Load X-ray data
    if per_group_xray is not None and Path(per_group_xray).exists():
        xray_map = Path(per_group_xray)
        if verbose:
            log.info("  X-ray: per-group %s", xray_map.name)
        from astropy.io import fits as _fits
        with _fits.open(xray_map) as h:
            xdata = np.nan_to_num(h[0].data.astype(np.float64), nan=0.0)
            xhdr  = h[0].header.copy()
    else:
        xray_map_path = _SMALL_MAP if use_small_scale else _LARGE_MAP
        if not xray_map_path.exists():
            if verbose:
                log.warning("  X-ray map not found: %s", xray_map_path)
            return rgb
        xdata, xhdr = cutout_from_map(xray_map_path, ra, dec, radius_arcmin * 1.5)
        if not np.any(xdata > 0):
            return rgb

    # Reproject + mask
    raw = np.maximum(reproject_to(xdata, xhdr, ref_hdr), 0.0) * cov

    # Fill point-source holes
    from scipy.ndimage import median_filter as _med
    if np.any(raw > 0):
        raw = np.where(raw > 0, raw, _med(raw, size=15))
        raw = np.maximum(raw, 0.0) * cov

    # Smooth — use truncate=3 to limit kernel to 3σ and avoid edge pull-down
    sc = gaussian_filter(raw, sigma=smooth_sigma, truncate=3.0)
    sh = gaussian_filter(raw, sigma=smooth_haze_sigma, truncate=3.0)
    smoothed = (sc + smooth_haze_weight * sh) * cov

    pos_mask = smoothed > 0
    if not np.any(pos_mask):
        return rgb

    # Normalize: background → 0, peak → 1
    pos_vals = smoothed[pos_mask]
    bg   = np.percentile(pos_vals, noise_floor_pct)
    peak = np.percentile(pos_vals, 99.0)
    span = peak - bg + 1e-12
    shifted = np.clip((smoothed - bg) / span, 0.0, 1.0) * cov

    k    = 6.0
    norm = np.clip(np.log1p(k * shifted) / np.log1p(k), 0.0, 1.0) * cov

    # V2 RGBA colormap — cmap alpha ∈ [0,1] scaled by alpha_peak so the
    # final peak alpha == alpha_peak (controls overall X-ray opacity)
    r_c, g_c, b_c = xray_color(redshift)
    cmap = _xray_cmap_v2(r_c, g_c, b_c)
    xray_rgba = cmap(norm ** norm_power)
    xray_rgba[..., 3] = np.clip(xray_rgba[..., 3] * alpha_peak * cov, 0.0, 1.0)

    # Restrict fill to the largest connected X-ray component to suppress
    # isolated point-source blobs from other cluster/AGN in the field
    from skimage.measure import label as _label
    from scipy.ndimage import distance_transform_edt as _dist
    # Fill only the main cluster body (norm ≥ 0.35 ≈ 2nd contour level).
    # Outer faint emission (norm 0.22–0.35) is shown as contour lines only,
    # not filled — this avoids the dark outer halo on the black background.
    fill_mask = norm >= 0.35
    if np.any(fill_mask):
        lbl = _label(fill_mask)
        largest = np.argmax(np.bincount(lbl.ravel())[1:]) + 1
        fill_mask = (lbl == largest)

    # Smooth feather at fill boundary: fade alpha over 120 px
    dist_in = _dist(fill_mask).astype(np.float32)
    feather  = np.clip(dist_in / 120.0, 0.0, 1.0)
    xray_rgba[..., 3] *= feather

    # Pure additive blending: emission only adds light, never darkens.
    add = xray_rgba[..., :3] * xray_rgba[..., 3:4]
    rgb_out = np.clip(rgb + add, 0.0, 1.0)

    # Contours
    if show_contours and ax is not None and len(contour_levels) > 0:
        norm_c = gaussian_filter(norm, sigma=12.0) * cov
        if norm_c.max() > 0:
            norm_c = norm_c / norm_c.max()
        pink_shades = ["#FFB6C1", "#FF69B4", "#FF1493", "#C71585"]
        lws = list(contour_linewidths) + [1.2] * max(0, len(contour_levels) - len(contour_linewidths))
        for i, lvl in enumerate(contour_levels):
            if lvl >= norm_c.max():
                continue
            mask = _largest_component_mask(norm_c * cov, lvl)
            col  = pink_shades[min(i, len(pink_shades) - 1)]
            ax.contour(mask, levels=[0.5], colors=[col],
                       linewidths=[lws[i]], alpha=contour_alpha, zorder=3)

    return rgb_out.astype(np.float32)
