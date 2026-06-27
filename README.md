# cosmos-group-rgb-xray

Publication-quality **RGB + X-ray composite images** for COSMOS-Web galaxy groups.

Combines JWST NIRCam + HST ACS F814W optical data with Chandra/XMM X-ray maps
to produce figures like the ESA image [potm2504b](https://esawebb.org/images/potm2504b/)
(COSMOS-Web group 15, Gozaliasl et al. in prep.).

---

## Science overview

```
JWST NIRCam (F115W → B, F277W → G, F444W → R)
  + F150W luminance overlay (PixInsight LRGB style)
  + HST ACS F814W (optional, improves optical colour)
          ↓  asinh stretch + CLAHE
        RGB  ──────────────────────────────────────┐
                                                   │  screen blend
Chandra + XMM X-ray (diffuse or compact map)  ────→  composite PNG
  magenta  z < 0.3
  cyan     z ≥ 0.3
```

---

## Repository structure

```
cosmos-group-rgb-xray/
├── cosmos_rgb_xray/          Python package
│   ├── stretch.py            asinh + CLAHE stretch (PixInsight STF)
│   ├── blend.py              screen / overlay blend modes
│   ├── io_fits.py            FITS load, reproject, cutout, WCS check
│   ├── rgb.py                JWST NIRCam + HST RGB builder
│   ├── xray.py               X-ray overlay (reproject → smooth → screen blend)
│   ├── batch.py              catalog-driven batch runner
│   └── single.py             single-group pipeline (pre-built TIFF workflow)
├── scripts/
│   ├── make_cutouts.py       standalone cutout script (JWST / HST / X-ray)
│   └── run_cutouts.sh        bash wrapper for make_cutouts.py
├── catalogs/                 top-20% X-ray group CSV catalogs (no FITS data)
├── configs/                  per-group and batch YAML configs
├── logs/                     cutout run logs (git-ignored)
├── data/                     FITS + images — git-ignored, local only
└── outputs/                  composite PNGs — git-ignored
```

---

## Installation

```bash
git clone https://github.com/gozaliasl/cosmos-group-rgb-xray.git
cd cosmos-group-rgb-xray
pip install -e .
```

Dependencies: `numpy astropy reproject scipy scikit-image Pillow shapely`

---

## Full workflow

### Step 1 — Cut postage stamps (run on Candide)

`scripts/make_cutouts.py` cuts per-group FITS stamps from the full survey mosaics.
`scripts/run_cutouts.sh` is a bash wrapper with logging.

**Mosaic paths (Candide):**

| Data | Path |
|------|------|
| JWST NIRCam | `/n23data2/cosmosweb/COSMOS-Web_Jan24/NIRCam/v0.8/` |
| HST ACS F814W | `/n17data/shuntov/COSMOS-Web/Images_HST-ACS/Jan24Tiles/` |
| X-ray maps | `/n23data2/gozaliasl/xray_maps/` |

**Python script — full control:**

```bash
# JWST NIRCam only
python scripts/make_cutouts.py --jwst \
    --catalog catalogs/top20_cutout_combined.csv \
    --output  /n23data2/gozaliasl/groups_cutouts/group_inputs

# JWST + HST
python scripts/make_cutouts.py --jwst --hst \
    --catalog catalogs/top20_cutout_combined.csv \
    --output  /n23data2/gozaliasl/groups_cutouts/group_inputs

# X-ray maps only
python scripts/make_cutouts.py --xray \
    --catalog catalogs/top20_cutout_combined.csv \
    --output  /n23data2/gozaliasl/groups_cutouts/group_inputs

# Everything for specific groups, override size
python scripts/make_cutouts.py --jwst --hst --xray \
    --catalog catalogs/top20_cutout_combined.csv \
    --output  /n23data2/gozaliasl/groups_cutouts/group_inputs \
    --ids 15 41 376 --size 300 --overwrite
```

**Bash wrapper — with automatic logging:**

```bash
# Cut everything (JWST + HST + X-ray)
bash scripts/run_cutouts.sh all

# JWST + HST only
bash scripts/run_cutouts.sh optical

# X-ray only, different catalog
bash scripts/run_cutouts.sh xray \
    --catalog catalogs/top20_cutout_cw_hcg.csv

# Specific groups, custom size
bash scripts/run_cutouts.sh optical \
    --ids "15 41 376" --size 300 --overwrite
```

Logs are saved to `logs/cutouts_<mode>_<timestamp>.log`.

**Output layout per group:**

```
group_inputs/<group_id>/
    <group_id>_F115W.fits       JWST NIRCam
    <group_id>_F150W.fits
    <group_id>_F277W.fits
    <group_id>_F444W.fits
    <group_id>_F814W.fits       HST ACS
    <group_id>_large_scale.fits X-ray diffuse (noem map)
    <group_id>_small_scale.fits X-ray compact (wavelet map)
```

---

### Step 2 — Build RGB + X-ray composites

#### Batch (catalog-driven, all groups)

```bash
python -m cosmos_rgb_xray.batch \
    --catalog   catalogs/top20_cutout_combined.csv \
    --data-root /n23data2/gozaliasl/groups_cutouts/group_inputs \
    --output-dir /n23data2/gozaliasl/groups_cutouts/rgb_xray_outputs \
    --jobs 8 --verbose
```

#### Single group (pre-built TIFF — group 15 workflow)

For group 15 a 10 000 × 10 000 TIFF already exists (the ESA potm2504b source):

```bash
python -m cosmos_rgb_xray.single \
    --rgb     data/rgb_inputs/group_15_rgb/jwst_hst_rgb_15.tiff \
    --wcs     data/group_inputs/gg15/15_F814W_reprojected.fits \
    --xray    data/group_inputs/gg15/15_large_scale.fits \
    --ra      150.395 --dec 2.404 --redshift 0.11 \
    --output  outputs/group15_rgb_xray.png \
    --verbose
```

---

## Catalogs

| File | Description |
|------|-------------|
| `top20_cutout_cw_all.csv`  | All COSMOS-Web groups, top 20% X-ray SNR |
| `top20_cutout_cw_hcg.csv`  | Hickson Compact Groups subset |
| `top20_cutout_combined.csv`| Union of the above |

Key columns used: `group_id`, `RA`, `Dec`, `RA_xray_peak`, `Dec_xray_peak`, `cutout_arcsec`, `z`, `SNR_xray`

---

## X-ray maps

| Map | Use |
|-----|-----|
| `cosmos_chaxmm14_noem_520.fits` | Diffuse ICM — point sources removed (`--xray` default) |
| `cosmos_chaxmm14_520_wv.3.fits` | Compact structure — wavelet scale 3 |

> **Never** use `cosmos_chaxmm14_520.fits` (raw combined) as overlay source —
> photon-count noise mimics real structure when Gaussian-smoothed.

---

## Reference

Gozaliasl et al. (in prep.) — *X-ray properties of galaxy groups in the COSMOS-Web field*
