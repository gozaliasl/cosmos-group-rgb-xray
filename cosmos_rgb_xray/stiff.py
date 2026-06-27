"""
STIFF-based RGB builder (Bertin 2012).

STIFF is a command-line tool that converts FITS to TIFF/PNG with proper
astronomical colour scaling. It is usually available on clusters where
Bertin's tools (SExtractor, SWarp, PSFEx) are installed.

  stiff R.fits G.fits B.fits -OUTFILE_NAME rgb.tiff

Band assignment (same as rgb.py):
  R = F444W
  G = F277W  (or mean of F150W+F277W if both available)
  B = F115W

Usage
-----
  from cosmos_rgb_xray.stiff import build_rgb_stiff
  rgb, ref_hdr = build_rgb_stiff(group_dir, group_id, output_dir)
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits
from PIL import Image

from .io_fits import load_fits, reproject_to
from .rgb import find_band

# STIFF band assignment
STIFF_BANDS = {
    "R": "F444W",
    "G": "F277W",
    "B": "F115W",
}


def stiff_available() -> bool:
    return shutil.which("stiff") is not None


def build_rgb_stiff(
    group_dir: Path,
    group_id: int,
    output_dir: Path,
    gamma: float = 2.2,
    min_type: str = "GREYLEVEL",   # GREYLEVEL | QUANTILE | MANUAL
    max_type: str = "QUANTILE",
    max_val: float = 0.9999,
    colour_sat: float = 1.5,
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[fits.Header]]:
    """
    Build RGB using STIFF. Falls back to asinh pipeline if STIFF not found.

    Parameters
    ----------
    group_dir    : directory with per-group FITS cutouts
    group_id     : integer group ID
    output_dir   : where to write the intermediate TIFF
    gamma        : display gamma (default 2.2)
    min_type     : how STIFF sets the black level
    max_type     : how STIFF sets the white level  (QUANTILE recommended)
    max_val      : quantile for white level (0.9999 = clip top 0.01%)
    colour_sat   : colour saturation boost (1.0 = neutral)
    verbose      : print STIFF stdout

    Returns
    -------
    (rgb_array HxWx3 float32, ref_header) or (None, None) on failure.
    """
    if not stiff_available():
        if verbose:
            print("  STIFF not found — falling back to asinh pipeline", flush=True)
        from .rgb import build_rgb
        return build_rgb(group_dir, group_id, verbose=verbose)

    # Locate band files
    band_files = {}
    for ch, filt in STIFF_BANDS.items():
        f = find_band(group_dir, group_id, filt)
        if f is not None:
            band_files[ch] = f

    if not {"R", "G", "B"}.issubset(band_files):
        missing = {"R", "G", "B"} - set(band_files)
        if verbose:
            print(f"  STIFF: missing bands {missing}", flush=True)
        return None, None

    # Reference header from R band
    _, ref_hdr = load_fits(band_files["R"])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_tiff = output_dir / f"{group_id}_stiff_rgb.tiff"

    # Reproject G and B onto R grid if shapes differ
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fits_paths = {}
        for ch, filt in STIFF_BANDS.items():
            p = band_files[ch]
            data, hdr = load_fits(p)
            if (hdr.get("NAXIS1"), hdr.get("NAXIS2")) != \
               (ref_hdr.get("NAXIS1"), ref_hdr.get("NAXIS2")):
                data = reproject_to(data, hdr, ref_hdr)
                hdr  = ref_hdr
            tmp_fits = tmp / f"{group_id}_{ch}.fits"
            fits.PrimaryHDU(data=data.astype(np.float32), header=hdr).writeto(
                tmp_fits, overwrite=True)
            fits_paths[ch] = tmp_fits

        cmd = [
            "stiff",
            str(fits_paths["R"]),
            str(fits_paths["G"]),
            str(fits_paths["B"]),
            "-OUTFILE_NAME",    str(out_tiff),
            "-GAMMA",           str(gamma),
            "-MIN_TYPE",        min_type,
            "-MAX_TYPE",        max_type,
            "-MAX_LEVEL",       str(max_val),
            "-COLOUR_SAT",      str(colour_sat),
            "-VERBOSE_TYPE",    "QUIET",
        ]
        if verbose:
            print(f"  STIFF: {' '.join(cmd)}", flush=True)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  STIFF failed: {result.stderr.strip()}", flush=True)
            return None, None

    if not out_tiff.exists():
        print("  STIFF: output TIFF not created", flush=True)
        return None, None

    img = Image.open(out_tiff).convert("RGB")
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    if verbose:
        print(f"  STIFF RGB: {img.size[0]}×{img.size[1]}", flush=True)
    return rgb, ref_hdr
