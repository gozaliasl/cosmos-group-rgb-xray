"""
Single-group pipeline — works from a pre-built RGB TIFF + optional WCS reference.

Two WCS modes are supported:

  1. scratch (default for TIFFs without a matching FITS)
     A WCS header is built from scratch using the known JWST pixel scale
     (30 mas/px) and the group sky position.  CD2_2 is set *negative* so that
     row-0 = north (top), matching the PIL/TIFF convention displayed with
     origin='upper'.  This is the correct setting discovered while working on
     group 1555 — using +scale flips the X-ray north-south relative to the
     optical image.

  2. wcs (legacy, from a reference FITS)
     The WCS header is patched from an existing FITS file.  The pixel scale is
     scaled to match the (possibly downsampled) RGB size.  CD2_2 sign is
     preserved from the source FITS — use only if the source FITS was produced
     with north-down convention (rare).

Usage (scratch WCS, recommended for JWST TIFFs)
-----
  python -m cosmos_rgb_xray.single \
      --rgb   group_15_rgb/jwst_hst_rgb_15.tiff \
      --ra    150.4139 --dec 2.4370 --redshift 0.16 \
      --output outputs/group1555_xray.png

Usage (legacy: provide a reference FITS for WCS)
-----
  python -m cosmos_rgb_xray.single \
      --rgb   group_15_rgb/jwst_hst_rgb_15.tiff \
      --wcs   gg15/15_F814W_reprojected.fits \
      --xray  gg15/15_xray_cutout.fits \
      --ra    150.395 --dec 2.404 --redshift 0.11 \
      --output outputs/group15_rgb_xray.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

from .io_fits import load_fits
from .xray import overlay_xray


# Native JWST NIRCam pixel scale at 30 mas/px mosaics
JWST_SCALE_DEG = 30e-3 / 3600.0   # 0.03 arcsec → degrees


def _scratch_wcs(nw: int, nh: int, ra: float, dec: float,
                 scale_deg: float) -> fits.Header:
    """
    Build a minimal TAN WCS header for a PIL-loaded image displayed with
    origin='upper' (row-0 = north).

    CD2_2 is *negative* so that pixel-y increases southward, matching the
    PIL array convention where row-0 is the top (north) of the image.
    Using +scale_deg would flip the X-ray north-south relative to the
    optical image.
    """
    hdr = fits.Header()
    hdr["NAXIS"]  = 2
    hdr["NAXIS1"] = nw
    hdr["NAXIS2"] = nh
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL1"] = ra
    hdr["CRVAL2"] = dec
    hdr["CRPIX1"] = nw / 2.0
    hdr["CRPIX2"] = nh / 2.0
    hdr["CD1_1"]  = -scale_deg   # RA decreases with increasing pixel-x (east left)
    hdr["CD1_2"]  = 0.0
    hdr["CD2_1"]  = 0.0
    hdr["CD2_2"]  = -scale_deg   # Dec decreases with increasing pixel-y (south down)
    return hdr


def _patch_wcs_from_fits(ref_hdr: fits.Header, orig_w: int,
                          new_w: int) -> fits.Header:
    """
    Patch a FITS WCS header to match a (possibly downsampled) image size.
    Scales CRPIX and CD/CDELT accordingly.
    """
    hdr = ref_hdr.copy()
    orig_ref_w = hdr.get("NAXIS1", orig_w)
    sf = new_w / orig_ref_w
    hdr["NAXIS1"] = new_w
    hdr["NAXIS2"] = int(hdr.get("NAXIS2", orig_w) * sf)
    hdr["CRPIX1"] = hdr.get("CRPIX1", orig_w / 2) * sf
    hdr["CRPIX2"] = hdr.get("CRPIX2", orig_w / 2) * sf
    for key in ["CDELT1", "CDELT2"]:
        if key in hdr:
            hdr[key] = hdr[key] / sf
    if "CD1_1" in hdr:
        for key in ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]:
            if key in hdr:
                hdr[key] = hdr[key] / sf
    return hdr


def run_single(
    rgb_path: Path,
    ra: float,
    dec: float,
    redshift: float,
    wcs_path: Optional[Path] = None,
    xray_path: Optional[Path] = None,
    output_path: Path = Path("outputs/rgb_xray.png"),
    out_size: int = 3000,
    smooth_sigma: float = 35.0,
    smooth_haze_sigma: float = 90.0,
    smooth_haze_weight: float = 0.25,
    alpha_peak: float = 0.30,
    norm_power: float = 1.5,
    noise_floor_pct: float = 50.0,
    contour_levels: tuple = (0.20, 0.35, 0.58, 0.82),
    show_contours: bool = True,
    annotate: bool = False,
    verbose: bool = False,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Load RGB ──────────────────────────────────────────────────────────────
    img = Image.open(rgb_path).convert("RGB")
    orig_w, orig_h = img.size
    if verbose:
        print(f"RGB loaded: {orig_w}×{orig_h}", flush=True)

    sf   = out_size / max(orig_w, orig_h)
    nw   = int(orig_w * sf)
    nh   = int(orig_h * sf)
    img  = img.resize((nw, nh), Image.LANCZOS)
    rgb  = np.asarray(img, dtype=np.float32) / 255.0
    if verbose:
        print(f"  Resized to {nw}×{nh}", flush=True)

    # ── Build WCS header ──────────────────────────────────────────────────────
    if wcs_path is not None and wcs_path.exists():
        _, ref_hdr_raw = load_fits(wcs_path)
        ref_hdr = _patch_wcs_from_fits(ref_hdr_raw, orig_w, nw)
        if verbose:
            print(f"  WCS: patched from {wcs_path.name}", flush=True)
    else:
        scale_deg = JWST_SCALE_DEG / sf
        ref_hdr   = _scratch_wcs(nw, nh, ra, dec, scale_deg)
        if verbose:
            print(f"  WCS: scratch ({scale_deg*3600*1000:.1f} mas/px)", flush=True)

    # ── X-ray overlay ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(nw / 300, nh / 300), dpi=300)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    rgb_out = overlay_xray(
        rgb, ref_hdr,
        ra=ra, dec=dec,
        redshift=redshift,
        per_group_xray=xray_path,
        smooth_sigma=smooth_sigma,
        smooth_haze_sigma=smooth_haze_sigma,
        smooth_haze_weight=smooth_haze_weight,
        alpha_peak=alpha_peak,
        norm_power=norm_power,
        noise_floor_pct=noise_floor_pct,
        contour_levels=contour_levels,
        show_contours=show_contours,
        ax=ax,
        verbose=verbose,
    )

    ax.imshow(rgb_out, origin="upper", interpolation="nearest", zorder=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if annotate:
        plt.close(fig)
        from .annotate import annotate_and_save
        annotate_and_save(rgb_out, ref_hdr, output_path,
                          redshift=redshift, scale_kpc=500.0,
                          save_tiff=True, verbose=verbose)
    else:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        sz = output_path.stat().st_size / 1e6
        print(f"PNG  → {output_path}  ({sz:.1f} MB)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="RGB + X-ray composite for a single group")
    p.add_argument("--rgb",        required=True, type=Path, help="TIFF/PNG of pre-built RGB")
    p.add_argument("--wcs",        type=Path,  default=None,
                   help="FITS providing sky WCS (omit to use scratch WCS at --ra/--dec)")
    p.add_argument("--xray",       type=Path,  default=None, help="Per-group X-ray FITS")
    p.add_argument("--ra",         required=True, type=float)
    p.add_argument("--dec",        required=True, type=float)
    p.add_argument("--redshift",   required=True, type=float)
    p.add_argument("--output",     type=Path,  default=Path("outputs/rgb_xray.png"))
    p.add_argument("--size",       type=int,   default=3000, dest="out_size",
                   help="Output image size in pixels (longest axis)")
    p.add_argument("--smooth",     type=float, default=35.0,  dest="smooth_sigma")
    p.add_argument("--alpha",      type=float, default=0.30,  dest="alpha_peak",
                   help="Peak X-ray alpha (0–1)")
    p.add_argument("--power",      type=float, default=1.5,   dest="norm_power",
                   help="Power applied to normalised map (>1 suppresses faint emission)")
    p.add_argument("--no-contours", action="store_false", dest="show_contours",
                   help="Disable X-ray contour lines")
    p.add_argument("--annotate",   action="store_true",
                   help="Add RA/Dec axes and scale bar")
    p.add_argument("--verbose",    action="store_true")
    args = p.parse_args()

    run_single(
        rgb_path=args.rgb,
        wcs_path=args.wcs,
        ra=args.ra,
        dec=args.dec,
        redshift=args.redshift,
        xray_path=args.xray,
        output_path=args.output,
        out_size=args.out_size,
        smooth_sigma=args.smooth_sigma,
        alpha_peak=args.alpha_peak,
        norm_power=args.norm_power,
        show_contours=args.show_contours,
        annotate=args.annotate,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
