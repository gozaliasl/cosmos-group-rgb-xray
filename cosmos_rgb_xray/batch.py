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

from astropy.io import fits
from astropy.wcs import WCS

import numpy as np
from PIL import Image

from .rgb import build_rgb
from .xray import find_per_group_xray, overlay_xray

# Cutout radius used when no per-group X-ray file exists
DEFAULT_RADIUS_ARCMIN = 4.0


# --------------------------------------------------------------------------- #
#  Catalog loading
# --------------------------------------------------------------------------- #

def _get(row: Dict[str, str], *keys: str, default: str = "0") -> str:
    """Return the first non-empty matching key from a CSV row (case-insensitive fallback)."""
    for k in keys:
        if k in row and row[k].strip():
            return row[k].strip()
    lower = {kk.lower(): v for kk, v in row.items()}
    for k in keys:
        v = lower.get(k.lower(), "")
        if v.strip():
            return v.strip()
    return default


def load_catalog(csv_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Load a top-20% X-ray group catalog CSV.

    Accepted column name variants (case-insensitive):
      group_id / Group_ID / ID
      RA / RA_MODEL
      Dec / DEC / DEC_MODEL
      z / Redshift / LP_zfinal
      SNR_xray / SNR
      RA_xray_peak, Dec_xray_peak  (optional)
      cutout_arcsec                (optional, default 240)

    Returns {group_id: info_dict}.
    """
    groups: Dict[int, Dict[str, Any]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gid = int(float(_get(row, "group_id", "Group_ID", "ID")))
            ra  = float(_get(row, "RA", "RA_MODEL"))
            dec = float(_get(row, "Dec", "DEC", "DEC_MODEL"))
            info: Dict[str, Any] = {
                "ra":            ra,
                "dec":           dec,
                "z":             float(_get(row, "z", "Redshift", "LP_zfinal")),
                "snr":           float(_get(row, "SNR_xray", "SNR")),
                "ra_xray":       float(_get(row, "RA_xray_peak",  default=str(ra))),
                "dec_xray":      float(_get(row, "Dec_xray_peak", default=str(dec))),
                "cutout_arcsec": float(_get(row, "cutout_arcsec", default="240.0")),
                "catalog":       _get(row, "catalog", "Catalog", default=""),
            }
            groups[gid] = info
    return groups


def scan_data_root(data_root: Path) -> Dict[int, Dict[str, Any]]:
    """
    Auto-discover groups by scanning data_root for subdirectories that contain
    FITS cutouts. RA/Dec are read from the F115W (or first available) FITS WCS;
    z and SNR default to 0 / 5.0 (compact overlay params).
    """
    groups: Dict[int, Dict[str, Any]] = {}
    band_priority = ["F115W", "F150W", "F277W", "F444W", "F814W"]
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        try:
            gid = int(d.name)
        except ValueError:
            continue
        ref_fits: Optional[Path] = None
        for band in band_priority:
            candidate = d / f"{gid}_{band}.fits"
            if candidate.exists():
                ref_fits = candidate
                break
        if ref_fits is None:
            fits_files = list(d.glob("*.fits"))
            if fits_files:
                ref_fits = fits_files[0]
        ra, dec = 0.0, 0.0
        if ref_fits is not None:
            try:
                with fits.open(ref_fits) as hdul:
                    hdr = hdul[0].header
                    wcs = WCS(hdr, naxis=2)
                    ny, nx = hdul[0].data.shape[-2], hdul[0].data.shape[-1]
                    sky = wcs.pixel_to_world(nx / 2, ny / 2)
                    ra, dec = float(sky.ra.deg), float(sky.dec.deg)
            except Exception:
                pass
        groups[gid] = {
            "ra": ra, "dec": dec,
            "z": 0.0, "snr": 5.0,
            "ra_xray": ra, "dec_xray": dec,
            "cutout_arcsec": 240.0,
            "catalog": "",
        }
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
            alpha_peak=0.75,
            norm_power=2.0,
            noise_floor_pct=60.0,
        )
    else:
        return dict(
            use_small_scale=True,
            smooth_sigma=15.0,
            radius_arcmin=max(cutout_arcsec / 60.0, DEFAULT_RADIUS_ARCMIN),
            alpha_peak=0.65,
            norm_power=2.2,
            noise_floor_pct=65.0,
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
    rgb_method: str = "asinh",
    save_tiff: bool = False,
    annotate: bool = False,      # add RA/Dec axes + scale bar
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
    arr = np.clip(rgb_xray, 0, 1)
    output_dir.mkdir(parents=True, exist_ok=True)

    if annotate:
        from .annotate import annotate_and_save
        annotate_and_save(
            arr, ref_hdr, out_png,
            redshift=redshift,
            scale_kpc=500.0,
            save_tiff=save_tiff,
            max_px=info.get("max_px", 4000),
            verbose=verbose,
        )
    else:
        from PIL import Image
        max_px = info.get("max_px", 4000)
        h, w   = arr.shape[:2]
        if max(h, w) > max_px:
            scale = max_px / max(h, w)
            img_rs = Image.fromarray((arr * 255).astype(np.uint8)).resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS)
            arr = np.asarray(img_rs, dtype=np.float32) / 255.0
        img8 = Image.fromarray((arr * 255).astype(np.uint8))
        img8.save(out_png, dpi=(300, 300), optimize=True)
        if verbose:
            sz = out_png.stat().st_size / 1e6
            print(f"  [{group_id}] PNG  → {out_png}  ({sz:.1f} MB)", flush=True)
        if save_tiff:
            out_tiff = out_png.with_suffix(".tiff")
            img8.save(out_tiff, compression="tiff_lzw", dpi=(300, 300))
            if verbose:
                print(f"  [{group_id}] TIFF → {out_tiff}", flush=True)

    if verbose:
        print(f"  [{group_id}] done", flush=True)
    return True


# --------------------------------------------------------------------------- #
#  Batch runner
# --------------------------------------------------------------------------- #

def run_batch(
    data_root: Path,
    output_dir: Path,
    catalog: Optional[Path] = None,
    group_ids: Optional[List[int]] = None,
    jobs: int = 1,
    overwrite: bool = False,
    verbose: bool = False,
    rgb_method: str = "asinh",
    save_tiff: bool = False,
    max_px: int = 4000,
    annotate: bool = False,
) -> None:
    if catalog is not None:
        groups = load_catalog(catalog)
        # supplement missing groups via scan so hz_detected groups not in catalog still run
        scanned = scan_data_root(data_root)
        for gid, info in scanned.items():
            if gid not in groups:
                groups[gid] = info
    else:
        groups = scan_data_root(data_root)

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
                          overwrite, verbose, rgb_method, save_tiff, annotate)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futures = {
                ex.submit(process_group, gid, info, data_root, output_dir,
                          overwrite, verbose, rgb_method, save_tiff, annotate): gid
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
    p.add_argument("--catalog",    required=False, default=None, type=Path,
                   help="Group catalog CSV (optional; auto-scans data-root if omitted)")
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
    p.add_argument("--annotate", action="store_true",
                   help="Add RA/Dec axes and physical scale bar (for paper figures)")
    p.add_argument("--scale-kpc", type=float, default=500.0,
                   help="Scale bar physical size in kpc (default: 500)")
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
        annotate=args.annotate,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )



if __name__ == "__main__":
    main()
