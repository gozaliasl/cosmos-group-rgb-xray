"""
Build a colour RGB composite from JWST NIRCam bands (+ optional HST F814W).

Band assignment:
  R = F444W  (NIR red,    1.6×)
  G = F277W  (NIR green,  1.5×)
  B = F115W  (NIR blue,   1.0×)
  L = F150W  (luminance overlay, PixInsight LRGB-style)
  + optional HST F814W blended at 45 % for optical colour
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
