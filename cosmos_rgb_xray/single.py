"""
Single-group pipeline — works from a pre-built RGB TIFF + HST WCS reference.

This is the workflow used for group 15 (the ESA potm2504b image), where a
10000×10000 px TIFF already exists and only the X-ray overlay is needed.

Usage
-----
  python -m cosmos_rgb_xray.single \
      --rgb   group_15_rgb/jwst_hst_rgb_15.tiff \
      --wcs   gg15/15_F814W_reprojected.fits \
      --xray  gg15/15_large_scale.fits \
      --ra    150.395 --dec 2.404 --redshift 0.11 \
      --output outputs/group15_rgb_xray.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

from .io_fits import load_fits
from .xray import overlay_xray


def run_single(
    rgb_path: Path,
    wcs_path: Path,
    ra: float,
    dec: float,
    redshift: float,
    xray_path: Optional[Path] = None,
    output_path: Path = Path("outputs/rgb_xray.png"),
    smooth_sigma: float = 80.0,
    alpha: float = 0.88,
    gamma: float = 0.55,
    pmin: float = 30.0,
    pmax: float = 99.5,
    annotate: bool = False,
    verbose: bool = False,
) -> None:
    from typing import Optional

    MAX_PX = 4000   # downsample before X-ray overlay to keep processing fast

    # Load RGB (TIFF or PNG)
    img = Image.open(rgb_path).convert("RGB")
    orig_w, orig_h = img.size
    if verbose:
        print(f"RGB loaded: {orig_w}×{orig_h}", flush=True)

    # Downsample to MAX_PX before expensive X-ray processing
    if max(orig_w, orig_h) > MAX_PX:
        scale = MAX_PX / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        if verbose:
            print(f"  Downsampled to {new_w}×{new_h} for processing", flush=True)

    rgb = np.asarray(img, dtype=np.float32) / 255.0
    w, h = img.size

    # Build WCS reference header from HST / any FITS covering the group
    _, ref_hdr = load_fits(wcs_path)
    # Patch NAXIS and pixel scale to match the (possibly downsampled) RGB
    orig_ref_w = ref_hdr.get("NAXIS1", orig_w)
    scale_factor = w / orig_ref_w
    ref_hdr["NAXIS1"] = w
    ref_hdr["NAXIS2"] = h
    ref_hdr["CRPIX1"] = ref_hdr.get("CRPIX1", orig_w / 2) * scale_factor
    ref_hdr["CRPIX2"] = ref_hdr.get("CRPIX2", orig_h / 2) * scale_factor
    for key in ["CDELT1", "CDELT2"]:
        if key in ref_hdr:
            ref_hdr[key] = ref_hdr[key] / scale_factor
    if "CD1_1" in ref_hdr:
        for key in ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]:
            if key in ref_hdr:
                ref_hdr[key] = ref_hdr[key] / scale_factor

    # Overlay X-ray
    rgb_xray = overlay_xray(
        rgb, ref_hdr,
        ra=ra, dec=dec,
        redshift=redshift,
        per_group_xray=xray_path,
        smooth_sigma=smooth_sigma,
        alpha=alpha,
        gamma=gamma,
        pmin=pmin,
        pmax=pmax,
        verbose=verbose,
    )

    arr = np.clip(rgb_xray, 0, 1)
    if annotate:
        from .annotate import annotate_and_save
        annotate_and_save(arr, ref_hdr, output_path,
                          redshift=redshift, scale_kpc=500.0,
                          save_tiff=True, verbose=verbose)
    else:
        from PIL import Image as _PIL
        _PIL.fromarray((arr * 255).astype(np.uint8)).save(
            output_path, dpi=(300, 300))
        print(f"PNG  → {output_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="RGB + X-ray composite for a single group")
    p.add_argument("--rgb",       required=True, type=Path, help="TIFF/PNG of pre-built RGB")
    p.add_argument("--wcs",       required=True, type=Path, help="FITS providing sky WCS")
    p.add_argument("--xray",      type=Path, default=None,  help="Per-group X-ray FITS")
    p.add_argument("--ra",        required=True, type=float)
    p.add_argument("--dec",       required=True, type=float)
    p.add_argument("--redshift",  required=True, type=float)
    p.add_argument("--output",    type=Path,  default=Path("outputs/rgb_xray.png"))
    p.add_argument("--smooth",    type=float, default=80.0,  dest="smooth_sigma")
    p.add_argument("--alpha",     type=float, default=0.88)
    p.add_argument("--gamma",     type=float, default=0.55)
    p.add_argument("--annotate",  action="store_true",
                   help="Add RA/Dec axes and scale bar (for paper figures)")
    p.add_argument("--verbose",   action="store_true")
    args = p.parse_args()

    run_single(
        rgb_path=args.rgb,
        wcs_path=args.wcs,
        ra=args.ra,
        dec=args.dec,
        redshift=args.redshift,
        xray_path=args.xray,
        output_path=args.output,
        smooth_sigma=args.smooth_sigma,
        alpha=args.alpha,
        gamma=args.gamma,
        annotate=args.annotate,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
