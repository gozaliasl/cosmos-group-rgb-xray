"""
Batch pipeline — process all groups in a top-20% X-ray catalog.

Usage (CLI)
-----------
  python -m cosmos_rgb_xray.batch \
      --catalog catalogs/top20_cutout_combined.csv \
      --data-root /path/to/group_inputs \
      --output-dir outputs/ \
      --jobs 4

Each group needs FITS cutouts at its correct sky position:
  <data_root>/<group_id>/<group_id>_F115W.fits
  <data_root>/<group_id>/<group_id>_F150W.fits
  <data_root>/<group_id>/<group_id>_F277W.fits
  <data_root>/<group_id>/<group_id>_F444W.fits
  <data_root>/<group_id>/<group_id>_F814W.fits    (HST, optional but recommended)
  <data_root>/<group_id>/<group_id>_large_scale.fits  (per-group X-ray, optional)

When running on a remote cluster (e.g. Candide), point --data-root to the
directory where JWST cutouts from the COSMOS-Web mosaic were placed.
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from .rgb import build_rgb
from .xray import find_per_group_xray, overlay_xray

# Cutout radius used when no per-group X-ray file exists
DEFAULT_RADIUS_ARCMIN = 4.0


# --------------------------------------------------------------------------- #
#  Catalog loading
# --------------------------------------------------------------------------- #

def load_catalog(csv_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Load a top-20% X-ray group catalog CSV.

    Expected columns (subset used):
      group_id, RA, Dec, z, SNR_xray
      RA_xray_peak, Dec_xray_peak   (optional, falls back to RA/Dec)
      cutout_arcsec                 (optional, default 240)

    Returns {group_id: info_dict}.
    """
    groups: Dict[int, Dict[str, Any]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gid = int(float(row.get("group_id", row.get("ID", 0))))
            ra  = float(row.get("RA",  row.get("RA_MODEL",  0)))
            dec = float(row.get("Dec", row.get("DEC_MODEL", 0)))
            info: Dict[str, Any] = {
                "ra":            ra,
                "dec":           dec,
                "z":             float(row.get("z",        row.get("LP_zfinal", 0))),
                "snr":           float(row.get("SNR_xray", 0)),
                "ra_xray":       float(row.get("RA_xray_peak",  ra)),
                "dec_xray":      float(row.get("Dec_xray_peak", dec)),
                "cutout_arcsec": float(row.get("cutout_arcsec", 240.0)),
                "catalog":       row.get("catalog", ""),
            }
            groups[gid] = info
    return groups


# --------------------------------------------------------------------------- #
#  Per-group parameter selection
# --------------------------------------------------------------------------- #

def xray_params(snr: float, cutout_arcsec: float = 240.0) -> Dict[str, Any]:
    """
    Choose X-ray overlay parameters from group SNR and angular size.

    Extended / bright groups (SNR>10 or large cutout):
      → diffuse (noem) map, wide smooth, large radius
    Compact groups:
      → compact (wv.3) map, tight smooth, narrow radius
    """
    extended = snr > 10.0 or cutout_arcsec >= 240.0
    if extended:
        return dict(
            use_small_scale=False,
            smooth_sigma=60.0,
            radius_arcmin=cutout_arcsec / 60.0,
            alpha=0.75,
            gamma=0.55,
            pmin=30.0,
            pmax=99.5,
        )
    else:
        return dict(
            use_small_scale=True,
            smooth_sigma=15.0,
            radius_arcmin=max(cutout_arcsec / 60.0, DEFAULT_RADIUS_ARCMIN),
            alpha=0.65,
            gamma=0.50,
            pmin=20.0,
            pmax=99.0,
        )


# --------------------------------------------------------------------------- #
#  Single group
# --------------------------------------------------------------------------- #

def process_group(
    group_id: int,
    info: Dict[str, Any],
    data_root: Path,
    output_dir: Path,
    overwrite: bool = False,
    verbose: bool = False,
    rgb_method: str = "asinh",   # "asinh" | "stiff"
    save_tiff: bool = False,     # also save 16-bit TIFF
) -> bool:
    """
    Build RGB + X-ray composite for one group. Returns True on success.

    Data lookup order for group directory:
      1.  <data_root>/<group_id>/
      2.  <data_root>/         (flat layout — all groups in same dir)
    """
    out_png = output_dir / f"group_{group_id:05d}_rgb_xray.png"
    if out_png.exists() and not overwrite:
        if verbose:
            print(f"  [{group_id}] already done — skipping", flush=True)
        return True

    # Locate group data directory
    group_dir = data_root / str(group_id)
    if not group_dir.is_dir():
        group_dir = data_root   # flat layout fallback

    if verbose:
        print(f"[{group_id}] data dir: {group_dir}", flush=True)

    ra       = info["ra_xray"]   # prefer X-ray peak
    dec      = info["dec_xray"]
    redshift = info["z"]
    snr      = info["snr"]
    cutout_arcsec = info.get("cutout_arcsec", 240.0)

    # Build RGB
    if rgb_method == "stiff":
        from .stiff import build_rgb_stiff
        rgb, ref_hdr = build_rgb_stiff(group_dir, group_id,
                                       output_dir=output_dir, verbose=verbose)
    else:
        rgb, ref_hdr = build_rgb(group_dir, group_id, verbose=verbose)

    if rgb is None:
        print(f"  [{group_id}] SKIP — RGB build failed", file=sys.stderr)
        return False

    # X-ray parameters
    xp = xray_params(snr, cutout_arcsec)

    # Per-group X-ray (WCS-validated)
    per_group = find_per_group_xray(group_dir, group_id, info["ra"], info["dec"], verbose=verbose)

    # Overlay
    rgb_xray = overlay_xray(
        rgb, ref_hdr,
        ra=ra, dec=dec,
        redshift=redshift,
        per_group_xray=per_group,
        verbose=verbose,
        **xp,
    )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    arr = np.clip(rgb_xray, 0, 1)

    # Resize to publication size if larger than MAX_PX on either axis
    # A&A / ApJ accept 300 dpi at ~3.5" column width → ~1050 px sufficient
    # We cap at 4000 px to keep files reasonable (<10 MB PNG, <25 MB TIFF)
    MAX_PX = info.get("max_px", 4000)
    h, w = arr.shape[:2]
    if max(h, w) > MAX_PX:
        scale  = MAX_PX / max(h, w)
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        img_rs = Image.fromarray((arr * 255).astype(np.uint8)).resize(
            (new_w, new_h), Image.LANCZOS)
        arr = np.asarray(img_rs, dtype=np.float32) / 255.0
        if verbose:
            print(f"  [{group_id}] resized {w}×{h} → {new_w}×{new_h}", flush=True)

    # PNG — 8-bit, web/screen/journal submission
    img8 = Image.fromarray((arr * 255).astype(np.uint8))
    img8.save(out_png, dpi=(300, 300), optimize=True)
    if verbose:
        sz = out_png.stat().st_size / 1e6
        print(f"  [{group_id}] PNG  → {out_png}  ({sz:.1f} MB)", flush=True)

    # TIFF — 16-bit LZW, publication / print quality
    if save_tiff:
        out_tiff = output_dir / f"group_{group_id:05d}_rgb_xray.tiff"
        img16 = Image.fromarray((arr * 65535).astype(np.uint16))
        img16.save(out_tiff, compression="tiff_lzw", dpi=(300, 300))
        if verbose:
            sz = out_tiff.stat().st_size / 1e6
            print(f"  [{group_id}] TIFF → {out_tiff}  ({sz:.1f} MB)", flush=True)

    return True


# --------------------------------------------------------------------------- #
#  Batch runner
# --------------------------------------------------------------------------- #

def run_batch(
    catalog: Path,
    data_root: Path,
    output_dir: Path,
    group_ids: Optional[List[int]] = None,
    jobs: int = 1,
    overwrite: bool = False,
    verbose: bool = False,
    rgb_method: str = "asinh",
    save_tiff: bool = False,
    max_px: int = 4000,
) -> None:
    groups = load_catalog(catalog)
    if group_ids:
        groups = {k: v for k, v in groups.items() if k in group_ids}

    print(f"Processing {len(groups)} groups from {catalog.name}", flush=True)
    print(f"RGB method : {rgb_method}", flush=True)

    # Inject max_px into each group info dict for process_group
    for info in groups.values():
        info["max_px"] = max_px

    if jobs == 1:
        for gid, info in groups.items():
            process_group(gid, info, data_root, output_dir,
                          overwrite, verbose, rgb_method, save_tiff)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futures = {
                ex.submit(process_group, gid, info, data_root, output_dir,
                          overwrite, verbose, rgb_method, save_tiff): gid
                for gid, info in groups.items()
            }
            for fut in as_completed(futures):
                gid = futures[fut]
                try:
                    ok = fut.result()
                    if not ok:
                        print(f"  [{gid}] FAILED", file=sys.stderr)
                except Exception as e:
                    print(f"  [{gid}] ERROR: {e}", file=sys.stderr)


# --------------------------------------------------------------------------- #
#  CLI entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description="Batch RGB+X-ray composites for COSMOS-Web groups")
    p.add_argument("--catalog",    required=True, type=Path)
    p.add_argument("--data-root",  required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--ids",        nargs="*", type=int, default=None,
                   help="Process only these group IDs (default: all)")
    p.add_argument("--jobs",       type=int, default=1)
    p.add_argument("--overwrite",  action="store_true")
    p.add_argument("--verbose",    action="store_true")
    p.add_argument("--rgb-method", choices=["asinh", "stiff"], default="asinh",
                   help="RGB stretching method: asinh+CLAHE (default) or STIFF (Bertin)")
    p.add_argument("--tiff", action="store_true", dest="save_tiff",
                   help="Also save 16-bit LZW-compressed TIFF alongside PNG")
    p.add_argument("--max-px", type=int, default=4000,
                   help="Maximum pixel dimension of output images (default: 4000). "
                        "Larger images are downsampled with Lanczos.")
    args = p.parse_args()

    run_batch(
        catalog=args.catalog,
        data_root=args.data_root,
        output_dir=args.output_dir,
        group_ids=args.ids,
        jobs=args.jobs,
        rgb_method=args.rgb_method,
        save_tiff=args.save_tiff,
        max_px=args.max_px,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
