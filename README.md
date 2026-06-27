# cosmos-group-rgb-xray

Publication-quality **RGB + X-ray composite images** for COSMOS-Web galaxy groups.

Combines JWST NIRCam + optional HST F814W optical data with Chandra/XMM X-ray maps
to produce figures like the ESA image [potm2504b](https://esawebb.org/images/potm2504b/)
(COSMOS-Web group 15, Gozaliasl et al.).

---

## Pipeline overview

```
JWST NIRCam FITS          HST F814W (optional)
  F115W → B                  reprojected
  F277W → G          →   RGB  ─────────────────┐
  F444W → R                                    │  screen blend
  F150W → luminance                            ▼
                           Chandra/XMM X-ray ──→  final composite PNG
```

**Stretching**: asinh + CLAHE (PixInsight STF-style)  
**Blend mode**: screen (`1-(1-a)(1-b)`) — preserves galaxy colours through X-ray glow  
**X-ray colour**: magenta (z<0.3) · cyan (z≥0.3)

---

## Installation

```bash
git clone https://github.com/ghassem-gozaliasl/cosmos-group-rgb-xray.git
cd cosmos-group-rgb-xray
pip install -e .
```

---

## Data layout

Data files are **not tracked in git** (too large). Place them under `data/`:

```
data/
  group_inputs/
    <group_id>/
      <id>_F115W.fits          JWST NIRCam cutout (correct sky position)
      <id>_F150W.fits
      <id>_F277W.fits
      <id>_F444W.fits
      <id>_F814W.fits          HST ACS (optional, improves optical colour)
      <id>_large_scale.fits    Per-group X-ray cutout (optional)
  xray_maps/
    cosmos_chaxmm14_noem_520.fits    diffuse/extended X-ray map
    cosmos_chaxmm14_520_wv.3.fits    compact/wavelet X-ray map
  rgb_inputs/
    group_15_rgb/
      jwst_hst_rgb_15.tiff           pre-built 10000×10000 TIFF (group 15)
```

### Generating JWST cutouts on Candide

The COSMOS-Web mosaic lives on the Candide cluster. For each group, cut
`<size>×<size>` arcsec postage stamps centred on the group X-ray peak:

```bash
# Example using astropy's Cutout2D via a batch script
python scripts/make_cutouts.py \
    --mosaic /candide/path/to/cosmosweb_F444W.fits \
    --catalog catalogs/top20_cutout_combined.csv \
    --output  data/group_inputs/ \
    --filter  F444W
```

---

## Usage

### Single group (pre-built TIFF — group 15)

```bash
python -m cosmos_rgb_xray.single \
    --rgb      data/rgb_inputs/group_15_rgb/jwst_hst_rgb_15.tiff \
    --wcs      data/group_inputs/gg15/15_F814W_reprojected.fits \
    --xray     data/group_inputs/gg15/15_large_scale.fits \
    --ra       150.395 --dec 2.404 --redshift 0.11 \
    --output   outputs/group15_rgb_xray.png \
    --verbose
```

### Batch (catalog-driven)

```bash
python -m cosmos_rgb_xray.batch \
    --catalog   catalogs/top20_cutout_combined.csv \
    --data-root data/group_inputs/ \
    --output-dir outputs/ \
    --jobs 4 --verbose
```

---

## Catalogs

`catalogs/` contains the top-20% X-ray group samples (CSV, no FITS data):

| File | Description |
|------|-------------|
| `top20_cutout_cw_all.csv`  | All COSMOS-Web groups, top 20% by X-ray SNR |
| `top20_cutout_cw_hcg.csv`  | Hickson Compact Groups subset |
| `top20_cutout_combined.csv`| Union of the above |

---

## X-ray maps

Two pre-processed COSMOS Chandra+XMM maps are supported:

| Map | Use case |
|-----|----------|
| `cosmos_chaxmm14_noem_520.fits` | Diffuse/extended ICM (point sources removed) |
| `cosmos_chaxmm14_520_wv.3.fits` | Compact/wavelet scale 3 |

> ⚠️ Never use `cosmos_chaxmm14_520.fits` (raw combined) as an overlay source —
> it contains photon-count noise that mimics real structure when smoothed.

---

## Modules

| Module | Purpose |
|--------|---------|
| `stretch.py` | asinh + CLAHE stretching |
| `blend.py`   | screen and overlay blend modes |
| `io_fits.py` | FITS load, reproject, cutout, WCS coverage check |
| `rgb.py`     | Multi-band JWST RGB builder |
| `xray.py`    | X-ray overlay (reproject → smooth → screen blend) |
| `batch.py`   | Catalog-driven batch runner |
| `single.py`  | Single-group pipeline (pre-built TIFF) |

---

## Reference

Gozaliasl et al. (in prep.) — *X-ray properties of galaxy groups in the COSMOS-Web field*
