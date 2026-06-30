#!/usr/bin/env python3
"""
COSMOS-Web Group Cutout Pipeline
=================================
Cuts postage stamps for galaxy groups from survey mosaics.

Supported instruments / data types:
  --jwst    JWST NIRCam  F115W, F150W, F277W, F444W
  --hst     HST ACS/WFC  F814W
  --xray    Chandra+XMM  large-scale (noem) and/or wavelet (wv.3) maps

Input catalog CSV must contain:
  group_id, RA, Dec
  Optional: RA_xray_peak, Dec_xray_peak, cutout_arcsec

Output layout — per catalog sample, then per group:
  <output>/<Catalog>/<group_id>/
      <group_id>_F115W.fits       JWST
      <group_id>_F150W.fits
      <group_id>_F277W.fits
      <group_id>_F444W.fits
      <group_id>_F814W.fits       HST
      <group_id>_large_scale.fits X-ray diffuse
      <group_id>_small_scale.fits X-ray compact

  <Catalog> comes from the "Catalog" column in the CSV
  (e.g. "CW-All" for all COSMOS-Web groups, "CW-HCG" for Hickson compact groups).
  Groups from different samples never collide even if IDs overlap.

Examples
--------
# JWST only
python scripts/make_cutouts.py --jwst \\
    --catalog catalogs/top20_cutout_combined.csv \\
    --output  /n23data2/gozaliasl/groups_cutout/group_inputs

# JWST + HST
python scripts/make_cutouts.py --jwst --hst \\
    --catalog catalogs/top20_cutout_combined.csv \\
    --output  /n23data2/gozaliasl/groups_cutout/group_inputs

# X-ray only (for groups already processed)
python scripts/make_cutouts.py --xray \\
    --catalog catalogs/top20_cutout_combined.csv \\
    --output  /n23data2/gozaliasl/groups_cutout/group_inputs

# Everything, specific groups only
python scripts/make_cutouts.py --jwst --hst --xray \\
    --catalog catalogs/top20_cutout_combined.csv \\
    --output  /n23data2/gozaliasl/groups_cutout/group_inputs \\
    --ids 15 41 376 --size 300 --overwrite
"""
from __future__ import annotations

import argparse
import csv
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from shapely.geometry import Polygon

# Support running as a script without pip install
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cosmos_rgb_xray.cutout_size import cutout_arcsec as _cutout_arcsec

warnings.filterwarnings("ignore", category=UserWarning, module="astropy")

# ── Default mosaic paths (Candide) — all overridable via CLI ─────────────────
DEFAULT_JWST_DIR    = Path("/n23data2/cosmosweb/COSMOS-Web_Jan24/NIRCam/v0.8")
DEFAULT_HST_DIR     = Path("/n23data2/cosmosweb/COSMOS-Web_Jan24/ACS")
DEFAULT_HST_DIR_APR = Path("/n23data2/cosmosweb/COSMOS-Web_Apr23/ACS")
DEFAULT_CLUTCH_DIR  = Path("/n23data1/sharish/hst/mosaics")   # CLUTCH HST bands
DEFAULT_XRAY_DIR    = Path("/n23data2/gozaliasl/xray_maps")
DEFAULT_UVISTA_DIR  = Path("/automnt/n23data1/UltraVista/DR4-RC2")
DEFAULT_COSMOS2020_DIR = Path("/n08data/COSMOS2020/images")

JWST_FILTERS    = ["F115W", "F150W", "F277W", "F444W"]

# CLUTCH HST bands at /n23data1/sharish/hst/mosaics/<FILTER>/
# Pattern: mosaic_clutch_hst_<FILTER>_<res>mas_<tile>_<type>_sci.fits.gz
# F098M uses drz; F435W and F606W use drc
CLUTCH_HST_BANDS = {
    "F606W": "drc",
    "F435W": "drc",
    "F098M": "drz",
}
UVISTA_FILTERS  = ["Y", "J", "H", "Ks"]

# HSC bands used for optical gap-fill: filename stems in COSMOS2020/images/
HSC_BANDS = {
    "g": "HSC_G_SSP_PDR2.fits",
    "r": "HSC_R_SSP_PDR2.fits",
    "i": "HSC_i_SSP_PDR2_19_07_19_v3.fits",
}
XRAY_LARGE_MAP = "cosmos_chaxmm14_520.fits"        # full Chandra+XMM map (large scale)
XRAY_SMALL_MAP = "cosmos_chaxmm14_520_wv.3.fits"  # wavelet scale-3 (compact/small scale)


# ── COSMOS-Web tile footprints ────────────────────────────────────────────────
TILES: Dict[str, List[Tuple[float, float]]] = {
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


def find_tile(coord: SkyCoord) -> str:
    pt = Polygon([
        (coord.ra.deg, coord.dec.deg),
        (coord.ra.deg + 1e-7, coord.dec.deg - 1e-7),
        (coord.ra.deg + 1e-7, coord.dec.deg + 1e-7),
        (coord.ra.deg - 1e-7, coord.dec.deg + 1e-7),
    ])
    for name, corners in TILES.items():
        if Polygon(corners).intersects(pt):
            return name
    print(f"  WARNING: ({coord.ra.deg:.4f}, {coord.dec.deg:.4f}) outside all tiles — using A1")
    return "A1"


# ── Mosaic paths ──────────────────────────────────────────────────────────────

def jwst_mosaic_path(filter_name: str, tile: str, res: int = 30,
                     jwst_dir: Path = DEFAULT_JWST_DIR) -> Path:
    return jwst_dir / (
        f"mosaic_nircam_{filter_name.lower()}_COSMOS-Web_{res}mas_{tile}_v1.0_i2d.fits.gz"
    )


def hst_mosaic_path(tile: str, res: int = 30,
                    hst_dir: Path = DEFAULT_HST_DIR) -> Path:
    """Return path to F814W mosaic, trying Jan24 then Apr23 tile locations."""
    # Jan24 tiles (A and B series)
    jan24 = hst_dir / f"mosaic_cosmos_web_2024jan_{res}mas_tile_{tile}_hst_acs_wfc_f814w_drz_zp-28.09.fits"
    if jan24.exists():
        return jan24
    # Apr23 tiles — same filename convention, different base dir
    apr23 = DEFAULT_HST_DIR_APR / f"mosaic_cosmos_web_2023apr_{res}mas_tile_{tile}_hst_acs_wfc_f814w_drz_zp-28.09.fits"
    if apr23.exists():
        return apr23
    # Return Jan24 path as default (will be flagged as not found by caller)
    return jan24


def clutch_mosaic_path(filter_name: str, tile: str, res: int = 30,
                       clutch_dir: Path = DEFAULT_CLUTCH_DIR) -> Path:
    drz_or_drc = CLUTCH_HST_BANDS.get(filter_name, "drz")
    return clutch_dir / filter_name / (
        f"mosaic_clutch_hst_{filter_name}_{res}mas_{tile}_{drz_or_drc}_sci.fits.gz"
    )


def uvista_mosaic_path(band: str, uvista_dir: Path = DEFAULT_UVISTA_DIR) -> Optional[Path]:
    """Find the UltraVista mosaic for a given band (Y/J/H/Ks) by globbing the directory."""
    matches = sorted(uvista_dir.glob(f"UVISTA_{band}_*.fits"))
    # Exclude weight maps
    science = [p for p in matches if "weight" not in p.name]
    return science[0] if science else None


# ── Mosaic LRU cache ──────────────────────────────────────────────────────────

class _MosaicCache:
    def __init__(self, max_size: int = 5):
        self._cache: Dict[str, dict] = {}
        self._order: List[str] = []
        self.max_size = max_size
        self.hits = self.misses = 0

    def get(self, path: Path) -> Optional[dict]:
        key = str(path)
        if key in self._cache:
            self._order.remove(key); self._order.append(key)
            self.hits += 1
            return self._cache[key]
        if not path.exists():
            print(f"  NOT FOUND: {path}", flush=True)
            return None
        print(f"  Loading {path.name} ...", flush=True)
        t0 = time.time()
        with fits.open(path, memmap=True) as h:
            # Find first extension that contains a 2D image (skip BINTABLEs)
            ext = 0
            for i, hdu in enumerate(h):
                if hdu.data is not None and len(hdu.data.shape) == 2:
                    ext = i
                    break
            data, header = h[ext].data.copy(), h[ext].header.copy()
        entry = {"data": data, "wcs": WCS(header)}
        print(f"  {data.shape}  {time.time()-t0:.1f}s", flush=True)
        self._cache[key] = entry; self._order.append(key); self.misses += 1
        while len(self._cache) > self.max_size:
            del self._cache[self._order.pop(0)]
        return entry

    def stats(self) -> str:
        total = self.hits + self.misses
        rate  = self.hits / total * 100 if total else 0
        return f"cache {rate:.0f}% hit ({self.hits}/{total})"


_cache = _MosaicCache(max_size=5)


# ── Core cutout writer ────────────────────────────────────────────────────────

def cut_and_save(
    mosaic_path: Path,
    ra: float,
    dec: float,
    size_arcsec: float,
    out_path: Path,
    label: str = "",
) -> bool:
    """
    Extract a stamp from mosaic_path centred on (ra, dec) and save to out_path.
    WCS of the cutout is preserved exactly. Returns True on success.
    """
    entry = _cache.get(mosaic_path)
    if entry is None:
        return False

    data, wcs_obj = entry["data"], entry["wcs"]

    # proj_plane_pixel_scales works correctly for both CDELT and CD-matrix WCS
    from astropy.wcs.utils import proj_plane_pixel_scales as _pps
    pix_scale = float(_pps(wcs_obj)[0]) * 3600.0   # arcsec/pixel
    size_pix  = int(size_arcsec / pix_scale)

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    px, py = wcs_obj.world_to_pixel(coord)

    # Skip if centre is outside the tile footprint entirely
    if not (0 <= float(px) < data.shape[1] and 0 <= float(py) < data.shape[0]):
        print(f"  {label} outside tile — skipping", flush=True)
        return False

    try:
        cutout = Cutout2D(
            data, (float(px), float(py)), size_pix,
            wcs=wcs_obj, mode="partial", fill_value=0.0,
        )
    except Exception as e:
        print(f"  {label} cutout error: {e}", flush=True)
        return False

    if 0 in cutout.data.shape or min(cutout.data.shape) < 100:
        print(f"  {label} cutout too small {cutout.data.shape} — outside tile?", flush=True)
        return False

    hdr = cutout.wcs.to_header()
    hdr["RA_CUT"]  = (ra,          "cutout centre RA [deg]")
    hdr["DEC_CUT"] = (dec,         "cutout centre Dec [deg]")
    hdr["SZ_ARCS"] = (size_arcsec, "cutout size [arcsec]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(
        data=cutout.data.astype(np.float32), header=hdr
    ).writeto(out_path, overwrite=True)
    print(f"  {label:12s} {out_path.name}  {cutout.data.shape}", flush=True)
    return True


# ── Per-instrument cutout helpers ─────────────────────────────────────────────

def cut_jwst(group_id: int, ra: float, dec: float, size: float,
             tile: str, out_dir: Path, res: int, overwrite: bool,
             jwst_dir: Path = DEFAULT_JWST_DIR) -> int:
    n = 0
    for filt in JWST_FILTERS:
        out = out_dir / f"{group_id}_{filt}.fits"
        if out.exists() and not overwrite:
            print(f"  {filt:12s} exists — skip", flush=True); n += 1; continue
        if cut_and_save(jwst_mosaic_path(filt, tile, res, jwst_dir),
                        ra, dec, size, out, filt):
            n += 1
    return n


def cut_hst(group_id: int, ra: float, dec: float, size: float,
            tile: str, out_dir: Path, res: int, overwrite: bool,
            hst_dir: Path = DEFAULT_HST_DIR) -> int:
    out = out_dir / f"{group_id}_F814W.fits"
    if out.exists() and not overwrite:
        print(f"  F814W        exists — skip", flush=True); return 1
    return 1 if cut_and_save(hst_mosaic_path(tile, res, hst_dir),
                              ra, dec, size, out, "F814W") else 0


def cut_xray(group_id: int, ra: float, dec: float, size: float,
             out_dir: Path, overwrite: bool,
             xray_dir: Path = DEFAULT_XRAY_DIR) -> int:
    n = 0
    for map_name, label, out_suffix in [
        (XRAY_LARGE_MAP, "xray-large", "large_scale"),
        (XRAY_SMALL_MAP, "xray-small", "small_scale"),
    ]:
        xray_map = xray_dir / map_name
        if not xray_map.exists():
            print(f"  {label:12s} map not found: {xray_map}", flush=True)
            continue
        out = out_dir / f"{group_id}_{out_suffix}.fits"
        if out.exists() and not overwrite:
            print(f"  {label:12s} exists — skip", flush=True); n += 1; continue
        # X-ray maps are full-survey: cut a generous radius (1.5× requested)
        if cut_and_save(xray_map, ra, dec, size * 1.5, out, label):
            n += 1
    return n


def cut_uvista(group_id: int, ra: float, dec: float, size: float,
               out_dir: Path, overwrite: bool,
               uvista_dir: Path = DEFAULT_UVISTA_DIR) -> int:
    """Cut UltraVista Y/J/H/Ks stamps. Returns number of bands successfully cut."""
    n = 0
    for band in UVISTA_FILTERS:
        out = out_dir / f"{group_id}_UVISTA_{band}.fits"
        if out.exists() and not overwrite:
            print(f"  UVISTA_{band:3s}  exists — skip", flush=True); n += 1; continue
        mosaic = uvista_mosaic_path(band, uvista_dir)
        if mosaic is None:
            print(f"  UVISTA_{band:3s}  mosaic not found in {uvista_dir}", flush=True)
            continue
        if cut_and_save(mosaic, ra, dec, size, out, f"UVISTA_{band}"):
            n += 1
    return n


def cut_hsc(group_id: int, ra: float, dec: float, size: float,
            out_dir: Path, overwrite: bool,
            cosmos2020_dir: Path = DEFAULT_COSMOS2020_DIR) -> int:
    """Cut HSC g/r/i stamps from COSMOS2020. Returns number of bands successfully cut."""
    n = 0
    for band, fname in HSC_BANDS.items():
        out = out_dir / f"{group_id}_HSC_{band}.fits"
        if out.exists() and not overwrite:
            print(f"  HSC_{band}       exists — skip", flush=True); n += 1; continue
        mosaic = cosmos2020_dir / fname
        if not mosaic.exists():
            print(f"  HSC_{band}       not found: {mosaic}", flush=True)
            continue
        if cut_and_save(mosaic, ra, dec, size, out, f"HSC_{band}"):
            n += 1
    return n


# ── Catalog loader ────────────────────────────────────────────────────────────

def load_catalog(path: Path) -> List[dict]:
    groups = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                # Accept multiple naming conventions
                gid = int(float(
                    row.get("Group_ID") or row.get("group_id") or
                    row.get("ID") or 0))
                ra  = float(
                    row.get("RA") or row.get("RA_MODEL") or 0)
                dec = float(
                    row.get("DEC") or row.get("Dec") or
                    row.get("DEC_MODEL") or 0)
                ra_x  = float(row.get("RA_xray_peak")  or ra)
                dec_x = float(row.get("Dec_xray_peak") or dec)
                z     = float(
                    row.get("Redshift") or row.get("z") or
                    row.get("LP_zfinal") or 0)
                val = row.get("cutout_arcsec", "").strip()
                size = float(val) if val else _cutout_arcsec(z)
                # Catalog name used as output subfolder (CW-All / CW-HCG)
                cat = (row.get("Catalog") or row.get("catalog") or "unknown").strip()
                cat = cat.replace(" ", "_")
                groups.append(dict(id=gid, ra=ra, dec=dec,
                                   ra_xray=ra_x, dec_xray=dec_x,
                                   z=z, size=size, catalog=cat))
            except (ValueError, KeyError):
                continue
    return groups


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    pa = argparse.ArgumentParser(
        description="Cut JWST / HST / X-ray stamps for COSMOS-Web galaxy groups",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # What to cut
    pa.add_argument("--jwst",       action="store_true", help="Cut JWST NIRCam (F115W F150W F277W F444W)")
    pa.add_argument("--hst",        action="store_true", help="Cut HST ACS F814W (COSMOS-Web tiles)")
    pa.add_argument("--clutch-hst", action="store_true", help="Cut CLUTCH HST F606W, F435W, F098M")
    pa.add_argument("--xray",       action="store_true", help="Cut Chandra/XMM X-ray maps (large + compact)")
    pa.add_argument("--uvista",     action="store_true", help="Cut UltraVista Y/J/H/Ks (ground-based NIR fill)")
    pa.add_argument("--hsc",        action="store_true", help="Cut HSC g/r/i from COSMOS2020 (optical gap-fill)")

    # Input / output
    pa.add_argument("--catalog",   required=True,  type=Path, help="Group catalog CSV")
    pa.add_argument("--output",    required=True,  type=Path,
                    help="Root output dir — created automatically if missing")

    # Mosaic directories (defaults to Candide paths)
    pa.add_argument("--jwst-dir",  type=Path, default=DEFAULT_JWST_DIR,
                    help=f"JWST NIRCam mosaic directory (default: {DEFAULT_JWST_DIR})")
    pa.add_argument("--hst-dir",    type=Path, default=DEFAULT_HST_DIR,
                    help=f"HST ACS mosaic directory       (default: {DEFAULT_HST_DIR})")
    pa.add_argument("--clutch-dir", type=Path, default=DEFAULT_CLUTCH_DIR,
                    help=f"CLUTCH HST mosaic directory    (default: {DEFAULT_CLUTCH_DIR})")
    pa.add_argument("--xray-dir",    type=Path, default=DEFAULT_XRAY_DIR,
                    help=f"X-ray map directory        (default: {DEFAULT_XRAY_DIR})")
    pa.add_argument("--uvista-dir",      type=Path, default=DEFAULT_UVISTA_DIR,
                    help=f"UltraVista mosaic directory     (default: {DEFAULT_UVISTA_DIR})")
    pa.add_argument("--cosmos2020-dir",  type=Path, default=DEFAULT_COSMOS2020_DIR,
                    help=f"COSMOS2020 images directory     (default: {DEFAULT_COSMOS2020_DIR})")
    pa.add_argument("--xray-large", type=str, default=XRAY_LARGE_MAP,
                    help=f"Large-scale X-ray map filename (default: {XRAY_LARGE_MAP})")
    pa.add_argument("--xray-small", type=str, default=XRAY_SMALL_MAP,
                    help=f"Small-scale X-ray map filename (default: {XRAY_SMALL_MAP})")

    # Options
    pa.add_argument("--ids",       nargs="*", type=int, default=None,
                    help="Process only these group IDs (default: all)")
    pa.add_argument("--size",      type=float, default=None,
                    help="Override cutout size [arcsec] for all groups (ignores redshift)")
    pa.add_argument("--hst-size",  type=float, default=None,
                    help="Override HST cutout size [arcsec] independently of --size "
                         "(useful for border groups where HST covers more sky than JWST)")
    pa.add_argument("--radius",    type=float, default=None,
                    help="Physical aperture radius [Mpc] to compute size from redshift "
                         "(default: 1.5 Mpc); ignored if --size is given")
    pa.add_argument("--res",       type=int,   default=30,
                    help="JWST/HST pixel scale in mas (default: 30)")
    pa.add_argument("--overwrite", action="store_true",
                    help="Re-cut files that already exist")
    pa.add_argument("--xray-centre", choices=["optical", "xray"], default="xray",
                    help="Centre X-ray cutout on optical (RA/Dec) or X-ray peak (default: xray)")
    args = pa.parse_args()

    if not (args.jwst or args.hst or args.clutch_hst or args.xray or args.uvista or args.hsc):
        pa.error("Specify at least one of --jwst, --hst, --clutch-hst, --xray, --uvista, --hsc")

    # Output root is created automatically — no manual mkdir needed
    args.output.mkdir(parents=True, exist_ok=True)

    groups = load_catalog(args.catalog)
    if args.ids:
        groups = [g for g in groups if g["id"] in args.ids]

    # Drop bad rows (Dec=0 or RA=0 almost certainly means missing data)
    bad = [g for g in groups if abs(g["dec"]) < 0.1 or g["ra"] == 0]
    if bad:
        print(f"WARNING: skipping {len(bad)} groups with invalid coordinates: "
              f"{[g['id'] for g in bad]}", flush=True)
        groups = [g for g in groups if g not in bad]

    print(f"Groups     : {len(groups)}", flush=True)
    print(f"Data types : "
          f"{'JWST ' if args.jwst else ''}"
          f"{'HST-F814W ' if args.hst else ''}"
          f"{'CLUTCH-HST ' if args.clutch_hst else ''}"
          f"{'X-ray ' if args.xray else ''}"
          f"{'UltraVista ' if args.uvista else ''}"
          f"{'HSC' if args.hsc else ''}", flush=True)
    if args.jwst:       print(f"JWST dir      : {args.jwst_dir}",    flush=True)
    if args.hst:        print(f"HST dir       : {args.hst_dir}",     flush=True)
    if args.clutch_hst: print(f"CLUTCH dir    : {args.clutch_dir}",  flush=True)
    if args.xray:       print(f"X-ray dir     : {args.xray_dir}",    flush=True)
    if args.uvista:     print(f"UVista dir    : {args.uvista_dir}",   flush=True)
    if args.hsc:        print(f"COSMOS2020 dir: {args.cosmos2020_dir}", flush=True)
    print(f"Output     : {args.output}/<catalog>/<group_id>/  (created automatically)", flush=True)

    # ── Pre-compute per-group metadata ────────────────────────────────────────
    for g in groups:
        coord = SkyCoord(ra=g["ra"] * u.deg, dec=g["dec"] * u.deg)
        g["tile"]    = find_tile(coord)
        g["out_dir"] = args.output / g["catalog"] / str(g["id"])
        if args.size:
            g["size"] = args.size
        elif args.radius:
            g["size"] = _cutout_arcsec(g.get("z", 0), r_mpc=args.radius)
        # else: use size already set from catalog / redshift in load_catalog

    # ── Mosaic-centric loop: one mosaic loaded → all groups cut from it ───────
    # Strategy: iterate (tile, filter) pairs; load mosaic once per pair,
    # cut every group that falls in that tile before moving on.
    # This matches the --max_cache_size=1 pattern and avoids re-loading.

    from collections import defaultdict
    by_tile: Dict[str, List[dict]] = defaultdict(list)
    for g in groups:
        by_tile[g["tile"]].append(g)

    t0 = time.time()
    results: Dict[int, int] = {g["id"]: 0 for g in groups}  # counts cuts made

    # ── JWST: tile → filter → cut all groups in tile ──────────────────────────
    if args.jwst:
        print(f"\n{'═'*50}\n  JWST NIRCam\n{'═'*50}", flush=True)
        for tile, tile_groups in sorted(by_tile.items()):
            print(f"\n  Tile {tile}  ({len(tile_groups)} groups)", flush=True)
            for filt in JWST_FILTERS:
                mosaic = jwst_mosaic_path(filt, tile, args.res, args.jwst_dir)
                entry  = _cache.get(mosaic)
                if entry is None:
                    print(f"  {filt} — mosaic not found, skipping", flush=True)
                    continue
                print(f"  {filt}", flush=True)
                for g in tile_groups:
                    out = g["out_dir"] / f"{g['id']}_{filt}.fits"
                    if out.exists() and not args.overwrite:
                        print(f"    [{g['id']}] exists — skip", flush=True)
                        results[g["id"]] += 1
                        continue
                    if cut_and_save(mosaic, g["ra"], g["dec"], g["size"],
                                    out, f"[{g['id']}] {filt}"):
                        results[g["id"]] += 1

    # ── HST: tile → cut all groups in tile ───────────────────────────────────
    if args.hst:
        print(f"\n{'═'*50}\n  HST ACS F814W\n{'═'*50}", flush=True)
        for tile, tile_groups in sorted(by_tile.items()):
            print(f"\n  Tile {tile}  ({len(tile_groups)} groups)", flush=True)
            mosaic = hst_mosaic_path(tile, args.res, args.hst_dir)
            entry  = _cache.get(mosaic)
            if entry is None:
                print(f"  F814W — mosaic not found, skipping", flush=True)
                continue
            for g in tile_groups:
                out = g["out_dir"] / f"{g['id']}_F814W.fits"
                if out.exists() and not args.overwrite:
                    print(f"    [{g['id']}] exists — skip", flush=True)
                    results[g["id"]] += 1
                    continue
                hst_sz = args.hst_size if args.hst_size else g["size"]
                if cut_and_save(mosaic, g["ra"], g["dec"], hst_sz,
                                out, f"[{g['id']}] F814W"):
                    results[g["id"]] += 1

    # ── CLUTCH HST (F606W, F435W, F098M): same tile grid as COSMOS-Web ──────
    if args.clutch_hst:
        print(f"\n{'═'*50}\n  CLUTCH HST (F606W F435W F098M)\n{'═'*50}", flush=True)
        for tile, tile_groups in sorted(by_tile.items()):
            print(f"\n  Tile {tile}  ({len(tile_groups)} groups)", flush=True)
            for filt in CLUTCH_HST_BANDS:
                mosaic = clutch_mosaic_path(filt, tile, args.res, args.clutch_dir)
                entry  = _cache.get(mosaic)
                if entry is None:
                    print(f"  {filt} — mosaic not found, skipping", flush=True)
                    continue
                print(f"  {filt}", flush=True)
                for g in tile_groups:
                    out = g["out_dir"] / f"{g['id']}_{filt}.fits"
                    if out.exists() and not args.overwrite:
                        print(f"    [{g['id']}] exists — skip", flush=True)
                        results[g["id"]] += 1
                        continue
                    hst_sz = args.hst_size if args.hst_size else g["size"]
                    if cut_and_save(mosaic, g["ra"], g["dec"], hst_sz,
                                    out, f"[{g['id']}] {filt}"):
                        results[g["id"]] += 1

    # ── X-ray: full-survey maps, no tile needed — cut all groups ─────────────
    if args.xray:
        print(f"\n{'═'*50}\n  X-ray (Chandra+XMM)\n{'═'*50}", flush=True)
        for map_name, suffix, label in [
            (args.xray_large, "large_scale", "large-scale"),
            (args.xray_small, "small_scale", "small-scale (wavelet)"),
        ]:
            xray_map = args.xray_dir / map_name
            entry = _cache.get(xray_map)
            if entry is None:
                print(f"  {label} — map not found: {xray_map}", flush=True)
                continue
            print(f"\n  {label}  ({map_name})", flush=True)
            for g in groups:
                out = g["out_dir"] / f"{g['id']}_{suffix}.fits"
                if out.exists() and not args.overwrite:
                    print(f"    [{g['id']}] exists — skip", flush=True)
                    results[g["id"]] += 1
                    continue
                cx = g["ra_xray"]  if args.xray_centre == "xray" else g["ra"]
                cy = g["dec_xray"] if args.xray_centre == "xray" else g["dec"]
                if cut_and_save(xray_map, cx, cy, g["size"] * 1.5,
                                out, f"[{g['id']}] {suffix}"):
                    results[g["id"]] += 1

    # ── UltraVista: single full-field mosaics, no tile needed ─────────────────
    if args.uvista:
        print(f"\n{'═'*50}\n  UltraVista DR4 (Y J H Ks)\n{'═'*50}", flush=True)
        for band in UVISTA_FILTERS:
            mosaic = uvista_mosaic_path(band, args.uvista_dir)
            if mosaic is None:
                print(f"  {band} — mosaic not found in {args.uvista_dir}", flush=True)
                continue
            entry = _cache.get(mosaic)
            if entry is None:
                continue
            print(f"  {band}", flush=True)
            for g in groups:
                out = g["out_dir"] / f"{g['id']}_UVISTA_{band}.fits"
                if out.exists() and not args.overwrite:
                    print(f"    [{g['id']}] exists — skip", flush=True)
                    results[g["id"]] += 1
                    continue
                if cut_and_save(mosaic, g["ra"], g["dec"], g["size"],
                                out, f"[{g['id']}] UVISTA_{band}"):
                    results[g["id"]] += 1

    # ── HSC g/r/i: COSMOS2020 full-field mosaics, no tile needed ─────────────
    if args.hsc:
        print(f"\n{'═'*50}\n  HSC g / r / i  (COSMOS2020)\n{'═'*50}", flush=True)
        for band, fname in HSC_BANDS.items():
            mosaic = args.cosmos2020_dir / fname
            entry = _cache.get(mosaic)
            if entry is None:
                print(f"  HSC_{band} — not found: {mosaic}", flush=True)
                continue
            print(f"  HSC_{band}", flush=True)
            for g in groups:
                out = g["out_dir"] / f"{g['id']}_HSC_{band}.fits"
                if out.exists() and not args.overwrite:
                    print(f"    [{g['id']}] exists — skip", flush=True)
                    results[g["id"]] += 1
                    continue
                if cut_and_save(mosaic, g["ra"], g["dec"], g["size"],
                                out, f"[{g['id']}] HSC_{band}"):
                    results[g["id"]] += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    ok  = sum(1 for v in results.values() if v > 0)
    err = len(results) - ok
    print(f"\n{'─'*50}", flush=True)
    print(f"Done — {ok} ok  {err} failed  {elapsed:.0f}s", flush=True)
    print(f"{_cache.stats()}", flush=True)


if __name__ == "__main__":
    main()
