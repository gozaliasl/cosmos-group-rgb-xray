"""
FITS I/O helpers — load, reproject, cutout.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from reproject import reproject_interp


def load_fits(path: Path) -> Tuple[np.ndarray, fits.Header]:
    """
    Load a FITS image to float64, handling multi-extension files.
    NaN / Inf replaced with 0.
    """
    with fits.open(path, memmap=True) as h:
        # Prefer SCI extension; fall back to primary
        if len(h) > 1 and h[0].data is None:
            for ext in h[1:]:
                if ext.data is not None:
                    data = np.asarray(ext.data, dtype=np.float64)
                    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0), ext.header.copy()
        data = np.asarray(h[0].data, dtype=np.float64)
        hdr  = h[0].header.copy()
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0), hdr


def reproject_to(
    data: np.ndarray,
    src_hdr: fits.Header,
    ref_hdr: fits.Header,
) -> np.ndarray:
    """Reproject data from src_hdr WCS onto ref_hdr WCS grid."""
    shape = (ref_hdr["NAXIS2"], ref_hdr["NAXIS1"])
    out, _ = reproject_interp((data, WCS(src_hdr)), WCS(ref_hdr), shape_out=shape)
    return np.nan_to_num(out, nan=0.0)


def cutout_from_map(
    map_path: Path,
    ra: float,
    dec: float,
    radius_arcmin: float,
) -> Tuple[np.ndarray, fits.Header]:
    """
    Extract a sky cutout from a large FITS map.

    The cutout WCS is preserved exactly — CRPIX outside the parent image
    is valid for cutouts of a parent map (do NOT override CRPIX/CRVAL).

    Parameters
    ----------
    map_path      : full FITS map (e.g. COSMOS Chandra+XMM 2048x2048)
    ra, dec       : centre of cutout in decimal degrees
    radius_arcmin : half-width of cutout in arcminutes
    """
    data, hdr = load_fits(map_path)
    wcs = WCS(hdr)
    pix_scale_arcsec = abs(hdr["CDELT1"]) * 3600.0
    size_pix = int(radius_arcmin * 60.0 / pix_scale_arcsec * 2)
    coord  = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    cutout = Cutout2D(data, coord, size_pix, wcs=wcs, mode="partial", fill_value=0.0)
    new_hdr = hdr.copy()
    new_hdr.update(cutout.wcs.to_header())
    new_hdr["NAXIS1"] = cutout.data.shape[1]
    new_hdr["NAXIS2"] = cutout.data.shape[0]
    return np.asarray(cutout.data, dtype=np.float64), new_hdr


def wcs_covers(hdr: fits.Header, ra: float, dec: float) -> bool:
    """Return True if the FITS WCS footprint contains (ra, dec)."""
    try:
        wcs = WCS(hdr)
        nx, ny = hdr["NAXIS1"], hdr["NAXIS2"]
        corners = wcs.all_pix2world([[0, 0], [nx, 0], [0, ny], [nx, ny]], 0)
        return (corners[:, 0].min() < ra  < corners[:, 0].max() and
                corners[:, 1].min() < dec < corners[:, 1].max())
    except Exception:
        return False
