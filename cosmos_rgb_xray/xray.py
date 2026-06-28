"""
X-ray overlay: cutout → reproject → smooth → RGBA colormap composite + contours.

Two pre-processed global maps are supported:
  XRAY_LARGE  cosmos_chaxmm14_noem_520.fits  — diffuse ICM (point srcs removed)
  XRAY_SMALL  cosmos_chaxmm14_520_wv.3.fits  — compact sources (wavelet scale 3)

Per-group pre-cleaned FITS (e.g. 15_xray_cutout.fits) take priority
over the global maps when they exist and cover the group sky position.

IMPORTANT: never fall back to the raw combined map cosmos_chaxmm14_520.fits
— it contains photon-count noise that looks like real structure when smoothed.

Colour convention:
  z < 0.3  → magenta  (0.85, 0.0, 1.0)
  z ≥ 0.3  → cyan     (0.0,  1.0, 1.0)

Rendering:
  The X-ray is composited as a transparent RGBA colormap (background is fully
  transparent, only emission regions get colour).  Contour lines at 4 levels
  mark the X-ray surface-brightness structure.  This preserves galaxy colours
  and BCG visibility at all X-ray intensities.

WCS orientation note:
  When the reference RGB was loaded from a PIL image (TIFF/PNG, origin='upper'),
  build the scratch WCS with CD2_2 = -scale_deg (negative).  PIL row-0 = north
  (top), but FITS convention with CD2_2>0 puts row-0 = south, which would flip
  the X-ray north-south relative to the optical image.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, label, binary_erosion
import matplotlib.colors as mcolors

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


def _xray_cmap(r: float, g: float, b: float) -> mcolors.LinearSegmentedColormap:
    """
    Build an RGBA colormap where the background is fully transparent and only
    emission regions get colour.  Alpha ramps from 0 → 0.08 → 0.19 → 0.30
    so faint halo is visible while the peak stays semi-transparent (galaxies
    show through).
    """
    colors_rgba = [
        (0,   0,   0,    0.00),   # fully transparent — empty sky
        (1.0, 0.5, 0.7,  0.08),   # faint outer halo — subtle tint
        (r,   g,   b,    0.19),   # mid emission
        (r*0.9+0.05, g, b, 0.30), # peak — semi-transparent
    ]
    return mcolors.LinearSegmentedColormap.from_list("xray_rgba", colors_rgba)


def _largest_component_mask(arr: np.ndarray, level: float) -> np.ndarray:
    """Binary mask keeping only the single largest connected component ≥ level."""
    binary = arr >= level
    labeled, n = label(binary)
    if n == 0:
        return np.zeros_like(arr, dtype=float)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == sizes.argmax()).astype(float)


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
        f"{group_id}_xray_cutout.fits",
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


def optical_coverage_mask(rgb: np.ndarray, erode_px: int = 8) -> np.ndarray:
    """
    Binary mask of pixels where the optical image has real data.

    The sky floor added by build_rgb_fits is 0.002 per channel, so pixels
    with no JWST coverage remain at or very near that floor.  Threshold at
    0.004 (max across channels) to detect genuine data, then erode slightly
    to avoid artefacts right at the mosaic edge.

    Returns a float32 array of 0/1 with the same H×W as rgb.
    """
    mask = (rgb.max(axis=-1) > 0.004).astype(np.float32)
    if erode_px > 0:
        mask = binary_erosion(mask, iterations=erode_px).astype(np.float32)
    return mask


def hst_background_fill(
    gdir: Path,
    gid: str,
    ref_hdr: fits.Header,
    nw: int,
    nh: int,
    coverage: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Load HST F814W (preferring ``{gid}_F814W_large.fits`` over the standard
    cutout), reproject to the output WCS, and return a warm-gray float32
    H×W×3 array normalised to [0,1] for use as background fill in regions
    where ``coverage == 0``.

    Returns None if no HST file exists or there is no signal outside the
    JWST footprint.
    """
    for name in (f"{gid}_F814W_large.fits", f"{gid}_F814W.fits"):
        hst_path = gdir / name
        if hst_path.exists():
            break
    else:
        return None

    hst_data, hst_hdr = load_fits(hst_path)
    reproj = np.maximum(reproject_to(hst_data, hst_hdr, ref_hdr), 0.0)

    outside = coverage < 0.5
    if not reproj[outside].any():
        return None

    pos = reproj[reproj > 0]
    if len(pos) < 10:
        return None

    peak = np.percentile(pos, 99.5)
    norm = np.clip(
        np.arcsinh(3.0 * reproj / (peak + 1e-12)) / np.arcsinh(3.0), 0.0, 1.0
    )
    # Warm-gray tint — slightly warmer than neutral to distinguish from JWST sky
    return (norm[..., None] * np.array([0.72, 0.68, 0.60], dtype=np.float32))


def overlay_xray(
    rgb: np.ndarray,
    ref_hdr: fits.Header,
    ra: float,
    dec: float,
    redshift: float,
    radius_arcmin: float = 4.0,
    smooth_sigma: float = 20.0,
    smooth_haze_sigma: float = 50.0,
    smooth_haze_weight: float = 0.15,
    alpha_peak: float = 0.30,
    norm_power: float = 2.2,
    noise_floor_pct: float = 65.0,
    contour_levels: Tuple[float, ...] = (0.20, 0.35, 0.58, 0.82),
    contour_linewidths: Tuple[float, ...] = (0.4, 0.55, 0.75, 1.0),
    contour_alpha: float = 0.85,
    show_contours: bool = True,
    use_small_scale: bool = False,
    per_group_xray: Optional[Path] = None,
    coverage: Optional[np.ndarray] = None,
    ax=None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Reproject and RGBA-composite X-ray emission onto the RGB image.

    The X-ray is rendered as a transparent colormap (background = fully
    transparent) composited over the optical RGB.  Contour lines are drawn
    on the provided matplotlib Axes if ``ax`` is supplied and
    ``show_contours`` is True.

    Parameters
    ----------
    rgb              : float HxWx3 RGB array in [0,1]
    ref_hdr          : WCS header matching the RGB pixel grid.
                       IMPORTANT: if the RGB was loaded from a PIL image with
                       origin='upper', build this header with CD2_2 = -scale
                       (negative) so north is up in display coordinates.
    ra, dec          : X-ray centre (prefer RA_xray_peak / Dec_xray_peak)
    redshift         : group redshift (determines overlay colour)
    radius_arcmin    : cutout half-width when using global maps
    smooth_sigma     : core Gaussian smoothing in output pixels (default 20)
    smooth_haze_sigma: wide haze Gaussian sigma (default 50)
    smooth_haze_weight: weight of haze pass (default 0.15)
    alpha_peak       : maximum alpha in the RGBA colormap
    norm_power       : power applied to normalised map (default 2.2 — suppresses
                       shallow X-ray from spreading into empty-sky regions)
    noise_floor_pct  : percentile of positive pixels used as noise floor (default 65)
    contour_levels   : normalised levels at which to draw contours (0–1)
    contour_linewidths: linewidths for each contour level
    contour_alpha    : alpha for contour lines
    show_contours    : draw contour lines on ``ax`` (requires ax != None)
    use_small_scale  : use compact/wavelet map instead of diffuse map
    per_group_xray   : path to a pre-cleaned per-group FITS (overrides global)
    coverage         : optional float32 H×W mask (1=optical data, 0=no data).
                       When provided, the X-ray overlay and contours are
                       clipped to the optical footprint so the X-ray never
                       bleeds into black sky.  Compute with
                       ``optical_coverage_mask(rgb)``.
    ax               : matplotlib Axes on which to draw contours
    verbose          : print progress
    """
    # ── Optical coverage mask ─────────────────────────────────────────────────
    cov = coverage if coverage is not None else np.ones(rgb.shape[:2], dtype=np.float32)

    # ── Load X-ray data ───────────────────────────────────────────────────────
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
                print(f"  X-ray: no signal in {xray_map.name} at this position",
                      file=sys.stderr)
            return rgb
        if verbose:
            print(f"  X-ray: {xray_map.name}", file=sys.stderr)

    # ── Reproject onto RGB grid and mask to optical footprint ─────────────────
    raw = np.maximum(reproject_to(xdata, xhdr, ref_hdr), 0.0) * cov

    # ── Smooth (dual-pass: core structure + wide haze) ────────────────────────
    sc = gaussian_filter(raw, sigma=smooth_sigma)
    sh = gaussian_filter(raw, sigma=smooth_haze_sigma)
    smoothed = (sc + smooth_haze_weight * sh) * cov   # re-apply mask after blur

    pos_mask = smoothed > 0
    if not np.any(pos_mask):
        if verbose:
            print("  X-ray: empty after reproject — skipping", file=sys.stderr)
        return rgb

    # ── Noise floor + normalise ───────────────────────────────────────────────
    noise_floor = np.percentile(smoothed[pos_mask], noise_floor_pct)
    smoothed    = np.maximum(smoothed - noise_floor, 0.0) * cov

    pos2 = smoothed[smoothed > 0]
    if len(pos2) == 0:
        return rgb

    peak = np.percentile(pos2, 98.0)
    k    = 8.0
    norm = np.clip(
        np.log1p(k * np.clip(smoothed / (peak + 1e-12), 0.0, 1.0)) / np.log1p(k),
        0.0, 1.0,
    ) * cov

    # ── RGBA colormap composite ───────────────────────────────────────────────
    r, g, b = xray_color(redshift)
    cmap     = _xray_cmap(r, g, b)
    alpha_scale = alpha_peak / 0.30
    xray_rgba   = cmap(norm ** norm_power)
    xray_rgba[..., 3] = np.clip(xray_rgba[..., 3] * alpha_scale * cov, 0.0, 1.0)

    alpha_ch = xray_rgba[..., 3:4]
    rgb_out  = np.clip(xray_rgba[..., :3] * alpha_ch + rgb * (1.0 - alpha_ch), 0.0, 1.0)

    # ── Contour lines ─────────────────────────────────────────────────────────
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
            ax.contour(
                mask, levels=[0.5],
                colors=[col], linewidths=[lws[i]],
                alpha=contour_alpha, zorder=3,
            )

    return rgb_out
