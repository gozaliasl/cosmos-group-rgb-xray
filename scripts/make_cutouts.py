#!/usr/bin/env python3
"""
Cut JWST NIRCam + HST F814W postage stamps for COSMOS-Web galaxy groups.

Reads a top-20% X-ray group catalog and produces per-group cutouts:

  <output_root>/<group_id>/
      <group_id>_F115W.fits
      <group_id>_F150W.fits
      <group_id>_F277W.fits
      <group_id>_F444W.fits
      <group_id>_F814W.fits     (HST ACS)

Run on Candide where the mosaics live:

  python scripts/make_cutouts.py \
      --catalog catalogs/top20_cutout_combined.csv \
      --output  /n23data2/gozaliasl/groups_cutout/group_inputs \
      --size    240 \
      --res     30

Mosaic paths (Candide):
  JWST NIRCam: /n23data2/cosmosweb/COSMOS-Web_Jan24/NIRCam/v0.8/
  HST F814W:   /n17data/shuntov/COSMOS-Web/Images_HST-ACS/Jan24Tiles/
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from shapely.geometry import Polygon

warnings.filterwarnings("ignore", category=UserWarning, module="astropy")

# ─── Paths (Candide) ──────────────────────────────────────────────────────────
JWST_BASE = Path("/n23data2/cosmosweb/COSMOS-Web_Jan24/NIRCam/v0.8")
HST_BASE  = Path("/n17data/shuntov/COSMOS-Web/Images_HST-ACS/Jan24Tiles")
JWST_FILTERS = ["F115W", "F150W", "F277W", "F444W"]
RES = 30   # mas pixel scale

# ─── COSMOS-Web tile footprints ───────────────────────────────────────────────
TILE_POLYGONS: Dict[str, List[Tuple[float, float]]] = {
    "A1":  [(149.8703317,2.0856512),(149.7198796,2.1403395),(149.7908786,2.3354095),(149.9413496,2.2807163)],
    "A2":  [(150.0058959,2.0363591),(149.8554506,2.0910612),(149.9264667,2.2861269),(150.0769300,2.2314186)],
    "A3":  [(150.1414523,1.9870553),(149.9910155,2.0417704),(150.0620479,2.2368306),(150.2125019,2.1821081)],
    "A4":  [(150.2769995,1.9377408),(150.1265729,1.9924679),(150.1976208,2.1875215),(150.3480637,2.1327859)],
    "A5":  [(150.4125359,1.8884166),(150.2621212,1.9431545),(150.3331838,2.1382005),(150.4836139,2.0834528)],
    "A6":  [(149.8045087,1.9048087),(149.6540746,1.9594923),(149.7250552,2.1545612),(149.8755087,2.0998725)],
    "A7":  [(149.9400575,1.8555218),(149.7896293,1.9102182),(149.8606274,2.1052826),(150.0110740,2.0505800)],
    "A8":  [(150.0755992,1.8062243),(149.9251788,1.8609325),(149.9961935,2.0559913),(150.1466316,2.0012757)],
    "A9":  [(150.2111325,1.7569171),(150.0607214,1.8116361),(150.1317520,2.0066883),(150.2821799,1.9519607)],
    "A10": [(150.3466557,1.7076011),(150.1962556,1.7623299),(150.2673014,1.9573744),(150.4177173,1.9026358)],
    "B1":  [(150.0020274,2.4473359),(149.8515406,2.5020333),(149.9225757,2.6970916),(150.0730806,2.6423895)],
    "B2":  [(150.1376214,2.3980335),(149.9871430,2.4527469),(150.0581944,2.6478011),(150.2086900,2.5930817)],
    "B3":  [(150.2732061,2.3487174),(150.1227378,2.4034461),(150.1938048,2.5984949),(150.3442894,2.5437590)],
    "B4":  [(150.4087801,2.2993886),(150.2583236,2.3541315),(150.3294054,2.5491739),(150.4798772,2.4944226)],
    "B5":  [(150.5443418,2.2500480),(150.3938989,2.3048040),(150.4649946,2.4998389),(150.6154520,2.4450733)],
    "B6":  [(149.9361713,2.2664951),(149.7857017,2.3211879),(149.8567188,2.5162544),(150.0072070,2.4615567)],
    "B7":  [(150.0717506,2.2171978),(149.9212885,2.2719056),(149.9923224,2.4669678),(150.1428020,2.4122539)],
    "B8":  [(150.2073213,2.1678878),(150.0568686,2.2226097),(150.1279183,2.4176665),(150.2783878,2.3629373)],
    "B9":  [(150.3428821,2.1185662),(150.1924404,2.1733011),(150.2635052,2.3683514),(150.4139629,2.3136080)],
    "B10": [(150.4784314,2.0692337),(150.3280023,2.1239807),(150.3990815,2.3190234),(150.5495255,2.2642668)],
}

# ─── Tile determination ───────────────────────────────────────────────────────

def determine_tile(coord: SkyCoord) -> str:
    pt = Polygon([
        (coord.ra.deg, coord.dec.deg),
        (coord.ra.deg + 1e-7, coord.dec.deg - 1e-7),
        (coord.ra.deg + 1e-7, coord.dec.deg + 1e-7),
        (coord.ra.deg - 1e-7, coord.dec.deg + 1e-7),
    ])
    for name, corners in TILE_POLYGONS.items():
        if Polygon(corners).intersects(pt):
            return name
    return "A1"   # fallback (prints a warning below)


def hst_tile_for_coord(coord: SkyCoord) -> str:
    """Map sky position to HST tile name (mirrors JWST tile layout)."""
    return determine_tile(coord)


# ─── Mosaic path helpers ──────────────────────────────────────────────────────

def jwst_path(filter_name: str, tile: str, res: int = RES) -> Path:
    fname = (f"mosaic_nircam_{filter_name.lower()}_COSMOS-Web_"
             f"{res}mas_{tile}_v1.0_i2d.fits.gz")
    return JWST_BASE / fname


def hst_path(tile: str, res: int = RES) -> Path:
    fname = (f"mosaic_cosmos_web_2024jan_{res}mas_tile_{tile}_"
             f"hst_acs_wfc_f814w_drz_zp-28.09.fits")
    return HST_BASE / fname


# ─── Mosaic cache ─────────────────────────────────────────────────────────────

class MosaicCache:
    def __init__(self, max_size: int = 4):
        self._cache: Dict[str, dict] = {}
        self._order: List[str] = []
        self.max_size = max_size

    def get(self, path: Path) -> Optional[dict]:
        key = str(path)
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        if not path.exists():
            return None
        print(f"    Loading {path.name} ...", flush=True)
        t0 = time.time()
        with fits.open(path, memmap=True) as h:
            # JWST i2d: SCI in ext 1; HST drz: primary
            ext = 1 if (len(h) > 1 and h[1].data is not None) else 0
            data   = h[ext].data.copy()
            header = h[ext].header.copy()
        wcs = WCS(header)
        print(f"    {data.shape}  {time.time()-t0:.1f}s", flush=True)
        entry = {"data": data, "header": header, "wcs": wcs}
        self._cache[key] = entry
        self._order.append(key)
        while len(self._cache) > self.max_size:
            old = self._order.pop(0)
            del self._cache[old]
        return entry


_cache = MosaicCache(max_size=4)


# ─── Cutout writer ────────────────────────────────────────────────────────────

def write_cutout(
    mosaic_path: Path,
    ra: float,
    dec: float,
    size_arcsec: float,
    out_path: Path,
) -> bool:
    """
    Cut size_arcsec × size_arcsec stamp centred on (ra, dec) from mosaic_path.
    Saves as a minimal FITS (PrimaryHDU + correct WCS). Returns True on success.
    """
    entry = _cache.get(mosaic_path)
    if entry is None:
        print(f"    NOT FOUND: {mosaic_path.name}", flush=True)
        return False

    data, wcs_obj = entry["data"], entry["wcs"]
    pix_scale_arcsec = abs(wcs_obj.wcs.cdelt[0]) * 3600.0
    size_pix = int(size_arcsec / pix_scale_arcsec)

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    px, py = wcs_obj.world_to_pixel(coord)

    try:
        cutout = Cutout2D(
            data, (float(px), float(py)), size_pix,
            wcs=wcs_obj, mode="partial", fill_value=0.0,
        )
    except Exception as e:
        print(f"    Cutout failed: {e}", flush=True)
        return False

    if cutout.data.shape[0] == 0 or cutout.data.shape[1] == 0:
        print("    Empty cutout — position outside tile?", flush=True)
        return False

    hdu = fits.PrimaryHDU(
        data=cutout.data.astype(np.float32),
        header=cutout.wcs.to_header(),
    )
    hdu.header["RA_CUT"]  = (ra,  "cutout centre RA  [deg]")
    hdu.header["DEC_CUT"] = (dec, "cutout centre Dec [deg]")
    hdu.header["SZ_ARCS"] = (size_arcsec, "cutout size [arcsec]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(out_path, overwrite=True)
    print(f"    {out_path.name}  {cutout.data.shape}", flush=True)
    return True


# ─── Per-group processing ─────────────────────────────────────────────────────

def process_group(
    group_id: int,
    ra: float,
    dec: float,
    size_arcsec: float,
    output_root: Path,
    res: int = RES,
    overwrite: bool = False,
) -> bool:
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    tile  = determine_tile(coord)
    out_dir = output_root / str(group_id)

    print(f"\n[{group_id}] RA={ra:.5f} Dec={dec:.5f} tile={tile} "
          f"size={size_arcsec:.0f}\"", flush=True)

    any_ok = False

    # JWST NIRCam
    for filt in JWST_FILTERS:
        out = out_dir / f"{group_id}_{filt}.fits"
        if out.exists() and not overwrite:
            print(f"    {out.name} exists — skip", flush=True)
            any_ok = True
            continue
        ok = write_cutout(jwst_path(filt, tile, res), ra, dec, size_arcsec, out)
        any_ok = any_ok or ok

    # HST F814W
    hst_tile = hst_tile_for_coord(coord)
    hst_p    = hst_path(hst_tile, res)
    out_hst  = out_dir / f"{group_id}_F814W.fits"
    if out_hst.exists() and not overwrite:
        print(f"    {out_hst.name} exists — skip", flush=True)
    else:
        write_cutout(hst_p, ra, dec, size_arcsec, out_hst)

    return any_ok


# ─── Catalog loading ──────────────────────────────────────────────────────────

def load_groups(catalog_path: Path) -> List[dict]:
    groups = []
    with open(catalog_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gid = int(float(row.get("group_id", row.get("ID", 0))))
                ra  = float(row.get("RA",  row.get("RA_MODEL",  0)))
                dec = float(row.get("Dec", row.get("DEC_MODEL", 0)))
                # Prefer X-ray peak position for cutout centre
                ra  = float(row.get("RA_xray_peak",  ra))
                dec = float(row.get("Dec_xray_peak", dec))
                size = float(row.get("cutout_arcsec", 240.0))
                groups.append({"id": gid, "ra": ra, "dec": dec, "size": size})
            except (ValueError, KeyError):
                continue
    return groups


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Cut JWST+HST postage stamps for galaxy groups")
    p.add_argument("--catalog", required=True, type=Path,
                   help="top-20% CSV catalog (group_id, RA, Dec, cutout_arcsec)")
    p.add_argument("--output",  required=True, type=Path,
                   help="root output directory  e.g. /n23data2/gozaliasl/groups_cutout/group_inputs")
    p.add_argument("--size",    type=float, default=None,
                   help="Override cutout size [arcsec] for all groups")
    p.add_argument("--res",     type=int,   default=RES,
                   help=f"Pixel scale in mas (default: {RES})")
    p.add_argument("--ids",     nargs="*",  type=int, default=None,
                   help="Process only these group IDs")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    groups = load_groups(args.catalog)
    if args.ids:
        groups = [g for g in groups if g["id"] in args.ids]

    print(f"Cutting {len(groups)} groups → {args.output}", flush=True)
    t0 = time.time()
    ok = err = 0
    for g in groups:
        size = args.size if args.size else g["size"]
        try:
            if process_group(g["id"], g["ra"], g["dec"], size,
                             args.output, args.res, args.overwrite):
                ok += 1
            else:
                err += 1
        except Exception as e:
            print(f"  [{g['id']}] ERROR: {e}", flush=True)
            err += 1

    print(f"\nDone — {ok} ok, {err} failed — {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
