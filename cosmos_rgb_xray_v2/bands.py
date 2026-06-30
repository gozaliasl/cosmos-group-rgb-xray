"""
Band discovery, selection, and loading for V2 pipeline.

Auto-selects the best available bands for R/G/B/L channels from
whatever instruments are present in the group data directory.
Includes ground-based gap filling at survey edges.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

log = logging.getLogger(__name__)

# All bands the pipeline knows about, grouped by instrument
BAND_WAVELENGTH_UM = {
    # JWST NIRCam
    "F115W": 1.15, "F150W": 1.50, "F200W": 2.00,
    "F277W": 2.77, "F356W": 3.56, "F410M": 4.10, "F444W": 4.44,
    # HST ACS
    "F435W": 0.435, "F606W": 0.606, "F814W": 0.814,
    # HST WFC3
    "F098M": 0.98, "F125W": 1.25, "F160W": 1.60, "F275W": 0.275,
    # UltraVISTA
    "Y": 1.02, "J": 1.25, "H": 1.64, "Ks": 2.15,
    # HSC
    "g": 0.48, "r": 0.62, "i": 0.77,
}


def discover_bands(group_dir: Path, group_id: str) -> Dict[str, Path]:
    """
    Find all available band FITS files in a group directory.

    Returns dict mapping band_name → Path, only for files that exist
    AND have data shape > 100×100 (skips 10×10 placeholder files).
    """
    found: Dict[str, Path] = {}
    for band in BAND_WAVELENGTH_UM:
        p = group_dir / f"{group_id}_{band}.fits"
        if p.exists():
            try:
                with fits.open(p, memmap=True) as h:
                    sh = h[0].data.shape[-2:]
                    if min(sh) >= 100:
                        found[band] = p
                    else:
                        log.debug("  %s: skipped (%s too small)", band, sh)
            except Exception:
                pass
    log.info("  Available bands: %s", list(found.keys()))
    return found


def select_channels(
    available: Dict[str, Path],
    priority_r: List[str],
    priority_g: List[str],
    priority_b: List[str],
    priority_l: List[str],
) -> Dict[str, Optional[Path]]:
    """
    Select best band for each channel from priority lists.

    A band can appear in multiple priority lists — it will be used in
    whichever channel it is most appropriate for (first match wins per channel).
    """
    def pick(priority):
        for b in priority:
            if b in available:
                return available[b]
        return None

    sel = {
        "R": pick(priority_r),
        "G": pick(priority_g),
        "B": pick(priority_b),
        "L": pick(priority_l),
    }
    log.info("  Channel assignment: R=%s G=%s B=%s L=%s",
             *[p.stem if p else "None" for p in sel.values()])
    return sel


def load_band(path: Path) -> Tuple[np.ndarray, fits.Header]:
    """
    Load a single FITS band to float32.

    Handles multi-extension FITS, gzipped files, and NaN/Inf replacement.
    Returns (data, header) with data as float32.
    """
    with fits.open(path, memmap=True) as h:
        # Try primary HDU first; if empty try extensions
        if h[0].data is not None and h[0].data.ndim >= 2:
            data = np.asarray(h[0].data, dtype=np.float32)
            hdr  = h[0].header.copy()
        else:
            for ext in h[1:]:
                if ext.data is not None and ext.data.ndim >= 2:
                    data = np.asarray(ext.data, dtype=np.float32)
                    hdr  = ext.header.copy()
                    break
            else:
                raise ValueError(f"No image data found in {path}")

    # Squeeze trailing size-1 dimensions (some FITS have NAXIS=4 stubs)
    while data.ndim > 2:
        data = data[0]

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data, hdr


def pixel_scale_arcsec(hdr: fits.Header) -> float:
    """Return pixel scale in arcsec/pixel, works for CDELT and CD-matrix WCS."""
    w = WCS(hdr, naxis=2)
    return float(proj_plane_pixel_scales(w)[0]) * 3600.0


def build_scratch_wcs(ny: int, nx: int, ref_hdr: fits.Header) -> fits.Header:
    """
    Build a clean north-up TAN WCS centred on the reference image centre.

    CD2_2 is negative so row-0 = north (standard display convention).
    This is the same approach as v1's _scratch_wcs_from_fits.
    """
    w_in = WCS(ref_hdr, naxis=2)
    ny_orig = ref_hdr.get("NAXIS2", ny)
    nx_orig = ref_hdr.get("NAXIS1", nx)
    sky = w_in.pixel_to_world(nx_orig / 2.0, ny_orig / 2.0)
    ra  = float(sky.ra.deg)
    dec = float(sky.dec.deg)
    scale = pixel_scale_arcsec(ref_hdr) / 3600.0  # degrees/pixel

    hdr = fits.Header()
    hdr["NAXIS"]  = 2
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL1"] = ra
    hdr["CRVAL2"] = dec
    hdr["CRPIX1"] = nx / 2.0
    hdr["CRPIX2"] = ny / 2.0
    hdr["CD1_1"]  = -scale
    hdr["CD1_2"]  = 0.0
    hdr["CD2_1"]  = 0.0
    hdr["CD2_2"]  = -scale
    return hdr
