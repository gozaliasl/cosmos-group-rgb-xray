"""
Build a colour RGB composite from JWST NIRCam bands (+ optional HST F814W).

Two approaches are provided:

build_rgb_fits()
    Asinh-stretch pipeline used for FITS-based batch processing.  Produces the
    publication-quality composites for COSMOS-Web groups.  Band mixing:
      R = F444W×2.2 (85%) + F150W (15%)
      G = F277W (25%) + F150W (65%) + F444W (10%)
      B = F115W (62%) + F814W (32%) + F150W (6%)  [or F115W/F150W if no HST]
    Includes green-bell correction, sky floor, and north-up flip.

build_rgb()
    Legacy CLAHE / percentile-stretch pipeline (PixInsight LRGB-style).
    Kept for backward compatibility with TIFF-based workflows.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from astropy.io import fits

from .blend import overlay_channel
from .io_fits import load_fits, reproject_to
from .stretch import stretch_channel

# NIRCam channel weights
BAND_WEIGHTS: Dict[str, Tuple[str, float]] = {
    "red":   ("F444W", 1.6),
    "green": ("F277W", 1.5),
    "blue":  ("F115W", 1.0),
    "lum":   ("F150W", 1.25),
}
HST_BLEND = 0.45   # fraction of HST F814W added to final RGB


def find_band(directory: Path, group_id: int, filter_name: str) -> Optional[Path]:
    """Locate a FITS file for a given filter in a group directory."""
    for name in [
        f"{group_id}_{filter_name}.fits",
        f"{group_id}_{filter_name.lower()}.fits",
        f"{group_id}_{filter_name.upper()}.fits",
    ]:
        p = directory / name
        if p.exists():
            return p
    # Glob fallback
    matches = list(directory.glob(f"*{filter_name}*.fits")) + \
              list(directory.glob(f"*{filter_name.lower()}*.fits"))
    return matches[0] if matches else None


def build_rgb(
    group_dir: Path,
    group_id: int,
    plo: float = 0.1,
    phi: float = 99.8,
    clahe_clip: float = 0.015,
    lum_opacity: float = 0.80,
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[fits.Header]]:
    """
    Build a float32 HxWx3 RGB array from JWST FITS files.

    Returns (rgb, ref_header) or (None, None) if required bands are missing.

    The reference WCS is taken from the largest input image so that all
    other bands are reprojected onto it.

    Parameters
    ----------
    group_dir   : directory containing <group_id>_F*.fits files
    group_id    : integer group identifier
    plo, phi    : asinh percentile clip
    clahe_clip  : CLAHE clip limit
    lum_opacity : overlay opacity for F150W luminance layer
    verbose     : print progress
    """
    # Locate band files
    band_files: Dict[str, Path] = {}
    for role, (fname, _) in BAND_WEIGHTS.items():
        f = find_band(group_dir, group_id, fname)
        if f is not None:
            band_files[role] = f

    required = {"red", "green", "blue"}
    missing  = required - set(band_files)
    if missing:
        if verbose:
            print(f"  Missing bands: {missing}", flush=True)
        return None, None

    # Reference header = largest image (best resolution cutout)
    ref_hdr, ref_npix = None, 0
    for f in band_files.values():
        _, hdr = load_fits(f)
        npix = hdr.get("NAXIS1", 0) * hdr.get("NAXIS2", 0)
        if npix > ref_npix:
            ref_npix, ref_hdr = npix, hdr

    # Load and reproject each band
    channels: Dict[str, np.ndarray] = {}
    for role, (fname, _) in BAND_WEIGHTS.items():
        if role not in band_files:
            channels[role] = np.zeros((ref_hdr["NAXIS2"], ref_hdr["NAXIS1"]))
            continue
        data, hdr = load_fits(band_files[role])
        if verbose:
            print(f"  {fname}: {hdr['NAXIS1']}×{hdr['NAXIS2']}", flush=True)
        channels[role] = reproject_to(data, hdr, ref_hdr)

    # Stretch each channel
    r = stretch_channel(channels["red"],   plo, phi, clahe_clip=clahe_clip)
    g = stretch_channel(channels["green"], plo, phi, clahe_clip=clahe_clip)
    b = stretch_channel(channels["blue"],  plo, phi, clahe_clip=clahe_clip)

    # Apply channel weights and normalise
    rw = BAND_WEIGHTS["red"][1]
    gw = BAND_WEIGHTS["green"][1]
    bw = BAND_WEIGHTS["blue"][1]
    peak = max(rw, gw, bw)
    rgb = np.stack([r * rw / peak, g * gw / peak, b * bw / peak], axis=-1)

    # F150W luminance overlay (PixInsight LRGB-style local contrast boost)
    if "lum" in band_files:
        lum = stretch_channel(channels["lum"], plo, phi, clahe_clip=clahe_clip)
        for c in range(3):
            rgb[..., c] = overlay_channel(rgb[..., c], lum, lum_opacity)

    # Optional HST F814W — adds optical blue/green colour contrast
    hst_file = find_band(group_dir, group_id, "F814W")
    if hst_file is not None:
        hst_data, hst_hdr = load_fits(hst_file)
        hst = stretch_channel(reproject_to(hst_data, hst_hdr, ref_hdr), plo, phi)
        hst_rgb = np.stack([hst, hst, hst], axis=-1)
        rgb = rgb * (1.0 - HST_BLEND) + hst_rgb * HST_BLEND
        if verbose:
            print(f"  HST F814W blended at {HST_BLEND:.0%}", flush=True)

    return np.clip(rgb, 0.0, 1.0).astype(np.float32), ref_hdr


# ── Asinh pipeline (batch / publication quality) ──────────────────────────────

def _asinh(x: np.ndarray, s: float) -> np.ndarray:
    return np.arcsinh(s * np.clip(x, 0, None)) / np.arcsinh(s)


def build_rgb_fits(
    group_dir: Path,
    group_id: str,
) -> Tuple[Optional[np.ndarray], Optional[fits.Header]]:
    """
    Build a float32 H×W×3 RGB array directly from FITS cutouts using an
    asinh stretch with publication-tuned band mixing.

    Band mixing:
      R = F444W×2.2 (85 %) + F150W×1.8 (15 %)
      G = F277W (25 %) + F150W×1.8 (65 %) + F444W×2.2 (10 %)
      B = F115W (62 %) + F814W×1.2 (32 %) + F150W×1.8 (6 %)
          → falls back to F115W (74 %) + F150W (26 %) when HST is absent
            or has a different array shape than the JWST bands.

    Additional processing:
      - Green-bell correction suppresses the mid-green hump.
      - Sky floor of [0.002, 0.002, 0.022] mimics a faint sky background.
      - Output is flipped north-up (row 0 = north) for display with
        ``origin='upper'`` and a scratch WCS with ``CD2_2 = -scale``.

    Returns
    -------
    (float32 H×W×3 array in [0,1], FITS header) or (None, None) if required bands missing.
    """
    ref_hdr: Optional[fits.Header] = None

    def _load(name: str) -> Optional[np.ndarray]:
        nonlocal ref_hdr
        p = group_dir / f"{group_id}_{name}.fits"
        if not p.exists():
            return None
        with fits.open(p, memmap=False) as h:
            d = np.asarray(h[0].data, dtype=np.float32)
            if ref_hdr is None:
                ref_hdr = h[0].header.copy()
        return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    b = _load("F115W")
    l = _load("F150W")
    g = _load("F277W")
    r = _load("F444W")

    if any(x is None for x in (b, l, g, r)):
        return None, None

    h_d = _load("F814W")
    use_hst = h_d is not None and h_d.shape == b.shape and h_d.max() > 0

    R = np.clip(_asinh(r*2.2, 14)*0.85 + _asinh(l*1.8, 12)*0.15, 0, 1)
    G = np.clip(_asinh(g,     12)*0.25 + _asinh(l*1.8, 12)*0.65
                + _asinh(r*2.2, 14)*0.10, 0, 1)
    B = (np.clip(_asinh(b, 7)*0.62 + _asinh(h_d*1.2, 7)*0.32
                 + _asinh(l*1.8, 12)*0.06, 0, 1)
         if use_hst else
         np.clip(_asinh(b, 7)*0.74 + _asinh(l*1.8, 12)*0.26, 0, 1))

    # Green-bell correction — suppress mid-green hump
    G = np.clip(G * (1.0 - 0.10 * np.exp(-0.5 * ((G - 0.30) / 0.20)**2)), 0, 1)

    rgb = np.clip(np.stack([R, G, B], axis=-1) + [0.002, 0.002, 0.022], 0, 1)
    rgb = rgb[::-1, :, :].astype(np.float32)  # flip north-up

    # Fill JWST zero-coverage gaps with UltraVista ground-based NIR if available
    uvista_fill = _build_uvista_fill(group_dir, group_id, rgb.shape[:2])
    if uvista_fill is not None:
        rgb = _blend_fill(rgb, uvista_fill)

    # Adjust header WCS to match the north-up flip so overlay_xray reprojects correctly.
    # Flipping rows means: CD2_2 negated, CRPIX2 mirrored.
    if ref_hdr is not None:
        ny = rgb.shape[0]
        ref_hdr = ref_hdr.copy()
        for key in ("CD2_2", "CDELT2"):
            if key in ref_hdr:
                ref_hdr[key] = -ref_hdr[key]
        if "CRPIX2" in ref_hdr:
            ref_hdr["CRPIX2"] = ny + 1 - ref_hdr["CRPIX2"]

    return rgb, ref_hdr


# ── UltraVista gap-fill helpers ───────────────────────────────────────────────

def _load_ground_band(group_dir: Path, group_id: str, filename: str,
                      shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Load a ground-based cutout and resize to JWST stamp shape."""
    from scipy.ndimage import zoom
    p = group_dir / filename
    if not p.exists():
        return None
    with fits.open(p, memmap=False) as h:
        d = np.asarray(h[0].data, dtype=np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    if d.shape != shape:
        zy = shape[0] / d.shape[0]
        zx = shape[1] / d.shape[1]
        d = zoom(d, (zy, zx), order=3)
    return d


def _build_uvista_fill(
    group_dir: Path,
    group_id: str,
    shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Build a float32 H×W×3 fill RGB for JWST zero-coverage regions.

    Priority:
      1. HSC i/r/g  (optical, from COSMOS2020) → vivid blue/green optical colors
      2. UltraVista Ks/H/J (NIR fallback)

    Both are at 0.15"/pix; JWST is 0.03"/pix so we upsample ~5×.
    The optical HSC fill intentionally looks bluer than the warm JWST NIR,
    making the space/ground boundary visible but natural.
    """
    # ── Try HSC g/r/i first ───────────────────────────────────────────────────
    hsc_i = _load_ground_band(group_dir, group_id, f"{group_id}_HSC_i.fits", shape)
    hsc_r = _load_ground_band(group_dir, group_id, f"{group_id}_HSC_r.fits", shape)
    hsc_g = _load_ground_band(group_dir, group_id, f"{group_id}_HSC_g.fits", shape)

    if hsc_i is not None or hsc_r is not None or hsc_g is not None:
        r_band = hsc_i if hsc_i is not None else (hsc_r if hsc_r is not None else hsc_g)
        g_band = hsc_r if hsc_r is not None else r_band
        b_band = hsc_g if hsc_g is not None else g_band
        R = _asinh(r_band, 10) * 0.90
        G = _asinh(g_band, 10) * 0.85
        B = _asinh(b_band,  8) * 0.95  # boost blue for clear optical appearance
        fill = np.clip(np.stack([R, G, B], axis=-1) + [0.001, 0.001, 0.010], 0, 1)
        return fill[::-1, :, :].astype(np.float32)  # north-up flip

    # ── Fall back to UltraVista Ks/H/J ───────────────────────────────────────
    ks = _load_ground_band(group_dir, group_id, f"{group_id}_UVISTA_Ks.fits", shape)
    h  = _load_ground_band(group_dir, group_id, f"{group_id}_UVISTA_H.fits",  shape)
    j  = _load_ground_band(group_dir, group_id, f"{group_id}_UVISTA_J.fits",  shape)

    if ks is None and h is None and j is None:
        return None

    r_band = ks if ks is not None else (h if h is not None else j)
    g_band = h  if h  is not None else r_band
    b_band = j  if j  is not None else g_band
    R = _asinh(r_band, 8) * 0.85
    G = _asinh(g_band, 8) * 0.80
    B = _asinh(b_band, 8) * 0.75
    fill = np.clip(np.stack([R, G, B], axis=-1) + [0.002, 0.002, 0.015], 0, 1)
    return fill[::-1, :, :].astype(np.float32)


def _blend_fill(
    jwst_rgb: np.ndarray,
    fill_rgb: np.ndarray,
    feather_sigma: float = 20.0,
) -> np.ndarray:
    """
    Replace zero-coverage JWST pixels with ground-based fill, feathered at
    the boundary so there is no hard edge.

    jwst_weight = 1  → pure JWST
    jwst_weight = 0  → pure ground fill
    """
    from scipy.ndimage import gaussian_filter

    # Coverage mask: JWST has data where any band > sky floor
    coverage = (jwst_rgb.max(axis=-1) > 0.005).astype(np.float32)

    # Feather: smooth the mask edge
    weight = gaussian_filter(coverage, sigma=feather_sigma)
    weight = np.clip(weight, 0, 1)[..., np.newaxis]  # H×W×1

    blended = jwst_rgb * weight + fill_rgb * (1.0 - weight)
    return np.clip(blended, 0, 1).astype(np.float32)
