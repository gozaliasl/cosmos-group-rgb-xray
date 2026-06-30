"""
V2 RGB builder — GHS stretch + sky subtraction + smart band mixing.

Pipeline per group
------------------
1. Discover available bands
2. Select R/G/B/L channels from priority lists
3. Per-band: load → sky subtract → GHS stretch
4. Combine to RGB with optional luminance blending
5. Ground-based fill for survey-edge gaps
6. North-up flip + scratch WCS
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits
from reproject import reproject_interp
from astropy.wcs import WCS

from .config import V2Config, load_config
from .background import subtract_sky
from .stretch import ghs, asinh_stretch, tone_curve, percentile_norm
from .bands import discover_bands, select_channels, load_band, build_scratch_wcs

log = logging.getLogger(__name__)


def _reproject_to_ref(
    data: np.ndarray,
    src_hdr: fits.Header,
    ref_hdr: fits.Header,
) -> np.ndarray:
    """Reproject data from src_hdr onto the ref_hdr pixel grid."""
    shape = (ref_hdr["NAXIS2"], ref_hdr["NAXIS1"])
    out, _ = reproject_interp((data, WCS(src_hdr)), WCS(ref_hdr), shape_out=shape)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def _stretch_band(
    data: np.ndarray,
    cfg: V2Config,
) -> np.ndarray:
    """Apply sky subtraction + GHS stretch to a single band."""
    # Sky subtraction
    if cfg.background.subtract:
        data, sky = subtract_sky(
            data,
            method=cfg.background.method,
            sigma=cfg.background.sigma,
            iterations=cfg.background.iterations,
        )

    # Stretch
    method = cfg.stretch.method
    if method == "ghs":
        out = ghs(
            data,
            b=cfg.stretch.ghs_b,
            D=cfg.stretch.ghs_D,
            SP_pct=cfg.stretch.ghs_SP_pct,
            HP_pct=cfg.stretch.ghs_HP_pct,
            LP=cfg.stretch.ghs_LP,
        )
    elif method == "asinh":
        out = asinh_stretch(data)
    else:
        out = percentile_norm(data)

    return out


def _luminance_blend(
    rgb: np.ndarray,
    lum: np.ndarray,
    weight: float = 0.25,
) -> np.ndarray:
    """
    Blend a luminance (sharpening) layer into the RGB image.

    Converts to HSV, replaces V channel with a weighted blend of the
    original V and the luminance band. This sharpens without altering hue.
    """
    from skimage.color import rgb2hsv, hsv2rgb
    hsv = rgb2hsv(np.clip(rgb, 0, 1))
    hsv[..., 2] = np.clip(
        (1 - weight) * hsv[..., 2] + weight * np.clip(lum, 0, 1),
        0, 1
    )
    return hsv2rgb(hsv).astype(np.float32)


def _ground_fill(
    rgb: np.ndarray,
    ref_hdr: fits.Header,
    group_dir: Path,
    group_id: str,
    ground_bands: list,
    threshold: float = 0.05,
    blend_width: int = 50,
) -> np.ndarray:
    """
    Fill survey-edge gaps (zero/near-zero pixels) with ground-based data.

    For each pixel where all JWST channels are below threshold, replace
    with the ground-based equivalent, feathered at the edge.
    """
    # Coverage mask: True where JWST has data
    lum = np.max(rgb, axis=2)
    max_val = lum.max()
    if max_val <= 0:
        return rgb
    covered = lum > threshold * max_val

    if covered.all():
        return rgb  # no gaps

    # Find first available ground band
    for band in ground_bands:
        p = group_dir / f"{group_id}_{band}.fits"
        if not p.exists():
            continue
        try:
            gdata, ghdr = load_band(p)
            if gdata.shape[0] < 100:
                continue
            # Reproject to JWST grid
            gdata_repr = _reproject_to_ref(gdata, ghdr, ref_hdr)

            # Subtract sky + stretch
            gdata_repr, _ = subtract_sky(gdata_repr, method="sigma_clip")
            gdata_s = percentile_norm(gdata_repr)

            # Build 3-channel fill (gray from single band)
            fill_rgb = np.stack([gdata_s, gdata_s * 0.9, gdata_s * 0.8], axis=2)

            # Distance transform for smooth feathering at coverage edge
            from scipy.ndimage import distance_transform_edt
            dist = distance_transform_edt(~covered).astype(np.float32)
            alpha = np.clip(dist / max(blend_width, 1), 0, 1)[..., np.newaxis]

            rgb = rgb * (1 - alpha) + fill_rgb * alpha
            log.info("  Ground fill: %s (gap fraction %.1f%%)",
                     band, (~covered).mean() * 100)
            break
        except Exception as e:
            log.debug("  Ground fill %s failed: %s", band, e)

    return np.clip(rgb, 0, 1).astype(np.float32)


def build_rgb_v2(
    group_dir: Path,
    group_id: str,
    cfg: Optional[V2Config] = None,
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[fits.Header]]:
    """
    Build a publication-quality RGB composite using the V2 pipeline.

    Parameters
    ----------
    group_dir : directory containing per-band FITS files
    group_id  : group identifier string (matches filename prefix)
    cfg       : V2Config (defaults used if None)
    verbose   : print progress

    Returns
    -------
    (rgb, ref_hdr) where rgb is float32 HxWx3 in [0,1] and ref_hdr is
    a north-up TAN WCS FITS header, or (None, None) on failure.
    """
    if cfg is None:
        cfg = load_config()

    if verbose:
        log.setLevel(logging.DEBUG)

    # ── 1. Discover and select bands ────────────────────────────────────────
    available = discover_bands(group_dir, group_id)
    if not available:
        log.error("No valid FITS bands found in %s", group_dir)
        return None, None

    channels = select_channels(
        available,
        priority_r=cfg.band_r,
        priority_g=cfg.band_g,
        priority_b=cfg.band_b,
        priority_l=cfg.band_l,
    )

    if channels["R"] is None or channels["G"] is None or channels["B"] is None:
        log.error("Cannot assign R/G/B channels from: %s", list(available.keys()))
        return None, None

    # ── 2. Load reference band (highest priority available JWST band) ───────
    ref_path = (channels["L"] or channels["R"])
    ref_data, ref_hdr_orig = load_band(ref_path)
    ny, nx = ref_data.shape[-2:]
    ref_hdr = build_scratch_wcs(ny, nx, ref_hdr_orig)
    log.info("  Reference band: %s  (%dx%d px)", ref_path.stem, nx, ny)

    # ── 3. Load + reproject all channels to reference grid ──────────────────
    band_arrays: dict[str, np.ndarray] = {}
    for ch, path in channels.items():
        if path is None:
            continue
        data, hdr = load_band(path)
        if data.shape != (ny, nx):
            log.debug("  Reprojecting %s → reference grid", path.stem)
            data = _reproject_to_ref(data, hdr, ref_hdr)
        band_arrays[ch] = data

    # ── 4. Sky subtract + stretch each channel ───────────────────────────────
    stretched: dict[str, np.ndarray] = {}
    for ch, data in band_arrays.items():
        stretched[ch] = _stretch_band(data, cfg)
        log.debug("  %s: min=%.4f max=%.4f", ch,
                  stretched[ch].min(), stretched[ch].max())

    # ── 5. Build RGB array ───────────────────────────────────────────────────
    R = stretched["R"]
    G = stretched["G"]
    B = stretched["B"]

    # If G and B are the same band (NIR-only), boost color diversity
    if channels["G"] == channels["B"]:
        B = np.clip(B * 0.85 + 0.015, 0, 1)  # slight blue boost
        log.debug("  G==B: applying blue boost for color diversity")

    rgb = np.stack([R, G, B], axis=2)

    # ── 6. Tone curve (brighten midtones) ────────────────────────────────────
    rgb = tone_curve(rgb, gamma=cfg.stretch.tone_gamma, sky_floor=0.015)

    # ── 7. Luminance blending (sharpening) ───────────────────────────────────
    if cfg.stretch.use_luminance and "L" in stretched and channels["L"] != channels["R"]:
        rgb = _luminance_blend(rgb, stretched["L"],
                               weight=cfg.stretch.luminance_weight)
        log.info("  Luminance blend applied")

    # ── 8. Ground-based gap fill at survey edges ─────────────────────────────
    if cfg.ground_fill.enable:
        rgb = _ground_fill(
            rgb, ref_hdr, group_dir, group_id,
            ground_bands=cfg.ground_fill.bands,
            threshold=cfg.ground_fill.coverage_threshold,
            blend_width=cfg.ground_fill.blend_width_px,
        )

    # ── 9. Per-channel white balance: NIR-only combos skew yellow-green
    #       because F444W>>F277W>>F115W in surface brightness.
    #       Scale channels so the median of bright-star pixels is neutral.
    #       Use a fixed empirical matrix tuned for F444W/F277W/F115W.
    wb = np.array([0.80, 0.90, 1.00], dtype=np.float32)   # dim R, dim G, keep B
    rgb = np.clip(rgb * wb, 0, 1)

    # ── 10. North-up flip ────────────────────────────────────────────────────
    rgb = rgb[::-1, :, :].copy().astype(np.float32)

    log.info("  RGB built: shape=%s  min=%.3f max=%.3f",
             rgb.shape, rgb.min(), rgb.max())
    return rgb, ref_hdr
