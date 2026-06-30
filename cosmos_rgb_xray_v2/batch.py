"""
V2 batch pipeline — process groups using the V2 RGB builder + V1 X-ray overlay.

The X-ray overlay from v1 is reused (it is stable and tested). V2 improves
the RGB rendering (GHS stretch, sky subtraction, smart band selection).

Usage
-----
python -m cosmos_rgb_xray_v2.batch \\
    --catalog  catalogs/top20_cutout_combined.csv \\
    --data-root /Volumes/extHD/groups_cutouts/group_inputs/CW-All \\
    --output-dir outputs/v2 \\
    --ids 1 --verbose

On CANDID:
    python -m cosmos_rgb_xray_v2.batch \\
        --catalog  catalogs/hz_detected_cutout.csv \\
        --data-root /automnt/n23data2/gozaliasl/groups_cutouts/hz_detected/CW-All \\
        --output-dir /automnt/n23data2/gozaliasl/groups_cutouts/hz_detected_rgb_v2/CW-All \\
        --jobs 8
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import V2Config, load_config
from .rgb import build_rgb_v2

# V2 X-ray overlay (improved alpha ramp) + v1 helpers
from cosmos_rgb_xray_v2.xray import overlay_xray_v2
from cosmos_rgb_xray.xray import find_per_group_xray
from cosmos_rgb_xray.rgb import build_rgb_trilogy

log = logging.getLogger(__name__)

DEFAULT_RADIUS_ARCMIN = 4.0


# ---------------------------------------------------------------------------
# Catalog loading (same as v1)
# ---------------------------------------------------------------------------

def _get(row, *keys, default="0"):
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
    groups: Dict[int, Dict[str, Any]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            gid = int(float(_get(row, "group_id", "Group_ID", "ID")))
            ra  = float(_get(row, "RA", "RA_MODEL"))
            dec = float(_get(row, "Dec", "DEC", "DEC_MODEL"))
            groups[gid] = {
                "ra":            ra,
                "dec":           dec,
                "z":             float(_get(row, "z", "Redshift", "LP_zfinal")),
                "snr":           float(_get(row, "SNR_xray", "SNR")),
                "ra_xray":       float(_get(row, "RA_xray_peak",  default=str(ra))),
                "dec_xray":      float(_get(row, "Dec_xray_peak", default=str(dec))),
                "cutout_arcsec": float(_get(row, "cutout_arcsec", default="240.0")),
                "catalog":       _get(row, "catalog", "Catalog", default=""),
            }
    return groups


# ---------------------------------------------------------------------------
# X-ray parameter selection (improved over v1)
# ---------------------------------------------------------------------------

def xray_params_v2(snr: float, cutout_arcsec: float, cfg: V2Config) -> Dict[str, Any]:
    extended = snr > cfg.xray_extended_snr or cutout_arcsec >= cfg.xray_extended_cutout_arcsec
    if extended:
        return dict(
            use_small_scale=False,
            smooth_sigma=cfg.xray.smooth_sigma_extended,
            smooth_haze_sigma=cfg.xray.smooth_haze_sigma,
            smooth_haze_weight=cfg.xray.smooth_haze_weight,
            radius_arcmin=cutout_arcsec / 60.0,
            alpha_peak=cfg.xray.alpha_peak_extended,
            norm_power=cfg.xray.norm_power,
            noise_floor_pct=cfg.xray.bg_percentile,
            contour_levels=cfg.xray.contour_levels,
            contour_linewidths=cfg.xray.contour_linewidths,
            contour_alpha=cfg.xray.contour_alpha,
        )
    else:
        return dict(
            use_small_scale=True,
            smooth_sigma=cfg.xray.smooth_sigma_compact,
            smooth_haze_sigma=cfg.xray.smooth_haze_sigma,
            smooth_haze_weight=cfg.xray.smooth_haze_weight,
            radius_arcmin=max(cutout_arcsec / 60.0, DEFAULT_RADIUS_ARCMIN),
            alpha_peak=cfg.xray.alpha_peak_compact,
            norm_power=cfg.xray.norm_power,
            noise_floor_pct=cfg.xray.bg_percentile,
            contour_levels=cfg.xray.contour_levels,
            contour_linewidths=cfg.xray.contour_linewidths,
            contour_alpha=cfg.xray.contour_alpha,
        )


# ---------------------------------------------------------------------------
# Single group
# ---------------------------------------------------------------------------

def process_group_v2(
    group_id: int,
    info: Dict[str, Any],
    data_root: Path,
    output_dir: Path,
    cfg: V2Config,
    overwrite: bool = False,
    verbose: bool = False,
    rgb_method: str = "trilogy",
) -> bool:

    out_png    = output_dir / f"group_{group_id:05d}_rgb_xray.png"
    out_png_nc = output_dir / f"group_{group_id:05d}_rgb_xray_nocontours.png"

    if out_png.exists() and out_png_nc.exists() and not overwrite:
        log.info("[%d] already done — skipping", group_id)
        return True

    group_dir = data_root / str(group_id)
    if not group_dir.is_dir():
        group_dir = data_root

    log.info("[%d] data dir: %s", group_id, group_dir)

    ra       = info["ra_xray"]
    dec      = info["dec_xray"]
    redshift = info["z"]
    snr      = info["snr"]
    cutout_arcsec = info.get("cutout_arcsec", 240.0)

    # Build RGB — trilogy uses v1's proven pipeline; ghs uses v2's GHS stretch
    if rgb_method == "trilogy":
        rgb, ref_hdr = build_rgb_trilogy(group_dir, str(group_id))
        if rgb is not None:
            import numpy as np
            from scipy.ndimage import gaussian_filter
            # Color balance: reduce NIR green cast, boost blue for navy background
            rgb = np.clip(rgb * np.array([0.98, 0.65, 1.45], dtype=np.float32), 0, 1)
            # Unsharp mask: gentle sharpening to enhance galaxy structure
            # sigma=1.5px, amount=0.20 — subtle, avoids noise amplification
            blurred = gaussian_filter(rgb, sigma=[1.5, 1.5, 0])
            rgb = np.clip(rgb + 0.20 * (rgb - blurred), 0, 1)
    else:
        rgb, ref_hdr = build_rgb_v2(group_dir, str(group_id), cfg=cfg, verbose=verbose)

    if rgb is None:
        log.error("[%d] RGB build failed", group_id)
        return False

    # X-ray params
    xp = xray_params_v2(snr, cutout_arcsec, cfg)
    per_group = find_per_group_xray(group_dir, group_id, info["ra"], info["dec"],
                                    verbose=verbose)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h_px, w_px = rgb.shape[:2]
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Version 1: with contours ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(w_px / 300, h_px / 300), dpi=300)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    rgb_xray = overlay_xray_v2(
        rgb, ref_hdr, ra=ra, dec=dec, redshift=redshift,
        per_group_xray=per_group, verbose=verbose, ax=ax,
        show_contours=True, **xp,
    )
    ax.imshow(rgb_xray, origin="upper", interpolation="nearest")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # ── Version 2: no contours (heavier smooth, tighter noise floor) ─────────
    xp_nc = dict(xp,
                 smooth_sigma=xp["smooth_sigma"] * cfg.xray.nocontour_smooth_factor,
                 noise_floor_pct=cfg.xray.nocontour_bg_percentile)
    fig2, ax2 = plt.subplots(figsize=(w_px / 300, h_px / 300), dpi=300)
    ax2.axis("off")
    fig2.subplots_adjust(0, 0, 1, 1)
    rgb_xray_nc = overlay_xray_v2(
        rgb, ref_hdr, ra=ra, dec=dec, redshift=redshift,
        per_group_xray=per_group, verbose=False, ax=ax2,
        show_contours=False, **xp_nc,
    )
    ax2.imshow(rgb_xray_nc, origin="upper", interpolation="nearest")
    fig2.savefig(out_png_nc, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig2)

    if verbose:
        sz  = out_png.stat().st_size / 1e6
        sz2 = out_png_nc.stat().st_size / 1e6
        log.info("[%d] PNG  → %s (%.1f MB)", group_id, out_png.name, sz)
        log.info("[%d] PNG  → %s (%.1f MB)", group_id, out_png_nc.name, sz2)
        log.info("[%d] done", group_id)

    return True


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch_v2(
    data_root: Path,
    output_dir: Path,
    catalog: Optional[Path] = None,
    group_ids: Optional[List[int]] = None,
    jobs: int = 1,
    overwrite: bool = False,
    verbose: bool = False,
    config_path: Optional[Path] = None,
    rgb_method: str = "trilogy",
) -> None:
    cfg = load_config(config_path)

    if catalog is not None:
        groups = load_catalog(catalog)
    else:
        log.error("No catalog provided"); return

    if group_ids:
        groups = {k: v for k, v in groups.items() if k in group_ids}

    log.info("V2 pipeline: %d groups, rgb_method=%s", len(groups), rgb_method)

    if jobs == 1:
        for gid, info in groups.items():
            process_group_v2(gid, info, data_root, output_dir,
                             cfg, overwrite, verbose, rgb_method)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futures = {
                ex.submit(process_group_v2, gid, info, data_root, output_dir,
                          cfg, overwrite, verbose, rgb_method): gid
                for gid, info in groups.items()
            }
            for fut in as_completed(futures):
                gid = futures[fut]
                try:
                    if not fut.result():
                        log.error("[%d] FAILED", gid)
                except Exception as e:
                    log.error("[%d] ERROR: %s", gid, e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(message)s",
                        stream=sys.stdout)

    p = argparse.ArgumentParser(description="V2 RGB+X-ray batch pipeline")
    p.add_argument("--catalog",    required=False, default=None, type=Path)
    p.add_argument("--data-root",  required=True,  type=Path)
    p.add_argument("--output-dir", required=True,  type=Path)
    p.add_argument("--config",     default=None,   type=Path,
                   help="YAML config file (defaults used if omitted)")
    p.add_argument("--ids",        nargs="*", type=int, default=None)
    p.add_argument("--jobs",       type=int, default=1)
    p.add_argument("--overwrite",  action="store_true")
    p.add_argument("--verbose",    action="store_true")
    p.add_argument("--rgb-method", choices=["trilogy", "ghs"], default="trilogy",
                   help="RGB stretch method: trilogy (default, proven) or ghs (V2)")
    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_batch_v2(
        catalog=args.catalog,
        data_root=args.data_root,
        output_dir=args.output_dir,
        group_ids=args.ids,
        jobs=args.jobs,
        overwrite=args.overwrite,
        verbose=args.verbose,
        config_path=args.config,
        rgb_method=args.rgb_method,
    )


if __name__ == "__main__":
    main()
