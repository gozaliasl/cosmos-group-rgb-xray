# Session Notes — COSMOS-Web RGB + X-ray Pipeline

## Project goal
Publication-quality RGB + X-ray composite images of COSMOS-Web galaxy groups,
modelled after ESA image potm2504b (group 15, magenta/purple X-ray overlay).

---

## Pipeline overview

```
Catalog CSV
    ↓
make_cutouts.py          ← cut JWST / HST / X-ray stamps from survey mosaics
    ↓
batch.py / single.py     ← build RGB, overlay X-ray, save PNG + TIFF
    ↓
annotate.py (optional)   ← add RA/Dec axes + physical scale bar
```

---

## Key design decisions

### Cutout loop order — mosaic-centric
Load each mosaic **once**, cut all groups that fall in it, then move to the next.
Order: **tile → filter → all groups in tile** (NOT group-by-group).

```
Tile A1 → F115W → cut groups 15, 41, …
         → F150W → cut groups 15, 41, …
         → F277W → …
         → F444W → …
Tile B7 → F115W → …
X-ray large → load once → cut ALL groups
X-ray small → load once → cut ALL groups
```

### Output directory structure
```
group_inputs/
  CW-All/          ← from "Catalog" column in CSV
    15/
      15_F115W.fits
      15_F150W.fits
      15_F277W.fits
      15_F444W.fits
      15_F814W.fits
      15_large_scale.fits
      15_small_scale.fits
    41/
      …
  CW-HCG/          ← Hickson compact groups — separate subfolder
    15/
      …
```
The `Catalog` column in the input CSV drives the subfolder name.
Groups from different samples never collide even if IDs overlap.

### X-ray maps used
| File | Purpose |
|------|---------|
| `cosmos_chaxmm14_520.fits` | Large-scale Chandra+XMM map (diffuse emission) |
| `cosmos_chaxmm14_520_wv.3.fits` | Wavelet scale-3 (compact/point sources) |
| `cosmos_chaxmm14_noem_520.fits` | **NOT used** — this is a masked/emission-removed map |

### X-ray colour coding
| Redshift | Colour | RGB |
|----------|--------|-----|
| z < 0.3 | Magenta / purple | (0.85, 0, 1) |
| z ≥ 0.3 | Cyan | (0, 1, 1) |

### Blend mode
Screen blend: `1 - (1-a)(1-b)` — preserves galaxy colours through X-ray glow.

### RGB band assignment
| Channel | Band | Scale |
|---------|------|-------|
| R | F444W | 1.6× |
| G | F277W | 1.5× |
| B | F115W | 1.0× |
| Luminance | F150W | 1.25× |
| Blue blend | HST F814W | 45% |

### Redshift-aware cutout sizes
Physical radius 1.5 Mpc → arcsec at group redshift (Planck 2018 cosmology).
Typical values: z=0.1 → 1800", z=0.5 → 430", z=1.0 → 240".
Floor = 180", ceiling = 1800".

---

## Cluster (Candide) setup

### Environment
```bash
module load intelpython/3-2024.2.0   # Python 3.9.19
# No pip install — sys.path injection handles package import
```

### Key paths on Candide
```
JWST mosaics : /automnt/n23data2/cosmosweb/COSMOS-Web_Jan24/NIRCam/v0.8
HST mosaics  : /automnt/n17data/shuntov/COSMOS-Web/Images_HST-ACS/Jan24Tiles
X-ray maps   : /automnt/n23data2/gozaliasl/xray_maps
Cutout output: /automnt/n23data2/gozaliasl/groups_cutouts/group_inputs
Repo         : /automnt/n23data2/gozaliasl/cosmos-group-rgb-xray
```

### Submit cutout job
```bash
# All data types
sbatch scripts/run_cutouts_slurm.sh

# JWST only, specific groups
sbatch --export=MODE=jwst,IDS="15 41 376" scripts/run_cutouts_slurm.sh

# X-ray only
sbatch --export=MODE=xray scripts/run_cutouts_slurm.sh
```

### After git pull on Candide
```bash
cd /automnt/n23data2/gozaliasl/cosmos-group-rgb-xray
git pull
sbatch scripts/run_cutouts_slurm.sh
```

---

## Catalog column names expected
| Field | Accepted names |
|-------|---------------|
| Group ID | `Group_ID`, `group_id`, `ID` |
| RA | `RA`, `RA_MODEL` |
| Dec | `DEC`, `Dec`, `DEC_MODEL` |
| Redshift | `Redshift`, `z`, `LP_zfinal` |
| Catalog name | `Catalog`, `catalog` |
| X-ray RA peak | `RA_xray_peak` |
| X-ray Dec peak | `Dec_xray_peak` |
| Cutout size | `cutout_arcsec` (optional, computed from z if absent) |

---

## CLI quick reference

### Single group (pre-built TIFF input)
```bash
python -m cosmos_rgb_xray.single \
    --rgb      path/to/rgb.tiff \
    --wcs      path/to/ref_F814W.fits \
    --xray     path/to/15_large_scale.fits \
    --ra 150.395 --dec 2.404 --redshift 0.11 \
    --output   outputs/group15.png

# With RA/Dec axes + scale bar (paper figure)
python -m cosmos_rgb_xray.single ... --annotate
```

### Batch (from cutout FITS files)
```bash
python -m cosmos_rgb_xray.batch \
    --catalog   catalogs/top20_cutout_combined.csv \
    --data-root /path/to/group_inputs \
    --output-dir outputs/ \
    --jobs 4

# Paper figures with annotation
python -m cosmos_rgb_xray.batch ... --annotate --tiff --scale-kpc 500
```

### Cutouts only
```bash
python scripts/make_cutouts.py \
    --jwst --hst --xray \
    --catalog catalogs/top20_cutout_combined.csv \
    --output  /path/to/group_inputs

# Specific groups, override size
python scripts/make_cutouts.py --jwst \
    --catalog ... --output ... \
    --ids 15 41 376 --size 300 --overwrite
```

---

## Known issues / future improvements

- **X-ray core saturation**: at z~0.1 (group 15) the X-ray alpha is too high,
  washing out galaxy colours in the BCG region. Target is the Photoshop version
  (`jwst_hst_rgb_Xray_15_v4.png`) where galaxy colours remain visible through
  the glow. Fix: apply sqrt/log tone-map to X-ray layer before screen blend,
  or reduce `alpha` from 0.75 → ~0.55 for extended groups.

- **Cutout too large at low z**: group 15 at z=0.11 gets 1800" → mostly empty
  sky. Consider a tighter crop (600–800") for the RGB output even if the cutout
  is larger.

- **Black strip on annotated images**: matplotlib WCSAxes Dec label pushes the
  image inward. Fix in `annotate.py`: reduce left margin or move Dec label.

- **Scale bar position**: currently at image bottom edge, overlapping RA axis
  ticks. Should be pushed up ~5% inside the image frame.

- **STIFF pipeline**: available as `--rgb-method stiff` but requires STIFF
  binary in PATH. Falls back to asinh+CLAHE automatically if not found.

- **tifffile dependency**: 16-bit TIFF saving uses `tifffile` (not in default
  conda env). Falls back to 8-bit PIL TIFF if not installed.
  Install: `pip install tifffile`

---

## Reference image
`/Users/gozalig1/Projects/PHOTOSHOP_scripting/Real-ESRGAN/outputs/jwst_hst_rgb_Xray_15_v4.png`
— Group 15 composite made in Photoshop. This is the quality target.
Key features: natural X-ray falloff, galaxy colours visible through glow,
deep purple background, tight crop, no axes (outreach style).
