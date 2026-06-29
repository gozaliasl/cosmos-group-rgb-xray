#!/usr/bin/env python3
"""
Batch RGB + X-ray composite pipeline for COSMOS-Web group cutouts.

Improvements vs v1:
  - HST F814W background fill: where JWST has no coverage, fill with HST
    rendered as a muted warm-gray so border-groups still show galaxies.
  - Optical coverage mask: X-ray overlay and contours are clipped to the
    JWST footprint; no X-ray spill into blank sky.
  - Tighter smoothing: noise_floor_pct=65, haze_sigma=50, norm_power=2.2.

Run on Candide:
    python batch_xray.py [CW-All|CW-HCG|both]
    python batch_xray.py --cutouts <dir> --output <dir> --samples CW-All CW-HCG
"""
import sys, argparse, numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, label, binary_erosion
from reproject import reproject_interp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image as _PIL

# ── defaults (Candide paths) ──────────────────────────────────────────────────
DEFAULT_CUTOUTS = Path('/automnt/n23data2/gozaliasl/groups_cutouts/group_inputs')
DEFAULT_OUT     = Path('/automnt/n23data2/gozaliasl/cosmos-group-rgb-xray/outputs')


def load_band(p):
    with fits.open(p, memmap=False) as h:
        d = np.asarray(h[0].data, dtype=np.float32)
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


def asinh(x, s):
    return np.arcsinh(s * np.clip(x, 0, None)) / np.arcsinh(s)


def build_rgb(gdir, gid):
    b = load_band(gdir/f"{gid}_F115W.fits")
    l = load_band(gdir/f"{gid}_F150W.fits")
    g = load_band(gdir/f"{gid}_F277W.fits")
    r = load_band(gdir/f"{gid}_F444W.fits")
    hst = gdir/f"{gid}_F814W.fits"
    h_d = load_band(hst) if hst.exists() else None
    use_hst = h_d is not None and h_d.shape == b.shape and h_d.max() > 0

    R = np.clip(asinh(r*2.2, 14)*0.85 + asinh(l*1.8, 12)*0.15, 0, 1)
    G = np.clip(asinh(g,     12)*0.25 + asinh(l*1.8, 12)*0.65 + asinh(r*2.2, 14)*0.10, 0, 1)
    B = (np.clip(asinh(b, 7)*0.62 + asinh(h_d*1.2, 7)*0.32 + asinh(l*1.8, 12)*0.06, 0, 1)
         if use_hst else
         np.clip(asinh(b, 7)*0.74 + asinh(l*1.8, 12)*0.26, 0, 1))
    G = np.clip(G*(1 - 0.10*np.exp(-0.5*((G-0.30)/0.20)**2)), 0, 1)
    rgb = np.clip(np.stack([R, G, B], -1) + [0.002, 0.002, 0.022], 0, 1)
    return rgb[::-1, :, :]   # flip north-up (FITS row-0=south → PIL row-0=north)


def scratch_wcs(nw, nh, ra, dec, scale_deg):
    hdr = fits.Header()
    for k, v in [('NAXIS', 2), ('NAXIS1', nw), ('NAXIS2', nh),
                 ('CTYPE1', 'RA---TAN'), ('CTYPE2', 'DEC--TAN'),
                 ('CRVAL1', ra), ('CRVAL2', dec),
                 ('CRPIX1', nw/2), ('CRPIX2', nh/2),
                 ('CD1_1', -scale_deg), ('CD1_2', 0.),
                 ('CD2_1', 0.),         ('CD2_2', -scale_deg)]:
        hdr[k] = v
    return hdr


def largest_mask(arr, lvl):
    labeled, n = label(arr >= lvl)
    if n == 0:
        return np.zeros_like(arr)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == sizes.argmax()).astype(float)


def hst_background(gdir, gid, ref_hdr, nw, nh, coverage):
    """
    Load HST F814W (larger cutout if available), reproject to output WCS,
    return a warm-gray RGB array (H, W, 3) normalised to [0, 1].
    Only fills pixels where coverage == 0.
    Returns None if HST is unavailable or has no signal outside JWST footprint.
    """
    # Prefer the larger HST cutout written by make_cutouts --hst-size
    for name in (f"{gid}_F814W_large.fits", f"{gid}_F814W.fits"):
        hst_path = gdir / name
        if hst_path.exists():
            break
    else:
        return None

    with fits.open(hst_path, memmap=False) as hh:
        hst_data = np.nan_to_num(np.asarray(hh[0].data, np.float32))
        hst_hdr  = hh[0].header

    reproj, _ = reproject_interp((hst_data, WCS(hst_hdr)),
                                 WCS(ref_hdr), shape_out=(nh, nw))
    reproj = np.maximum(np.nan_to_num(reproj), 0.0)

    # Only care about regions outside the JWST footprint
    outside = (coverage < 0.5)
    if not reproj[outside].any():
        return None

    pos = reproj[reproj > 0]
    if len(pos) < 10:
        return None

    peak = np.percentile(pos, 99.5)
    norm = np.clip(np.arcsinh(3 * reproj / (peak + 1e-12)) / np.arcsinh(3), 0, 1)

    # Warm-gray tint: slightly warmer than neutral to distinguish from JWST sky
    rgb_hst = norm[..., None] * np.array([0.72, 0.68, 0.60], dtype=np.float32)
    return rgb_hst


def process_group(gid, gdir, out_path):
    # ── Load JWST WCS reference ───────────────────────────────────────────────
    with fits.open(gdir/f"{gid}_F277W.fits", memmap=False) as h:
        hdr0 = h[0].header
    w0 = WCS(hdr0)
    ny0, nx0 = hdr0['NAXIS2'], hdr0['NAXIS1']
    ra, dec = [float(v) for v in w0.all_pix2world(nx0/2, ny0/2, 0)]

    print(f"  {gid}: RGB ({nx0}×{ny0})...", end=' ', flush=True)
    rgb = build_rgb(gdir, gid)

    nw = nh = 3000
    sf = nw / max(nx0, ny0)
    native_scale = abs(hdr0.get('CD1_1', hdr0.get('CDELT1', 30e-3/3600)))
    out_scale = native_scale / sf
    ref_hdr = scratch_wcs(nw, nh, ra, dec, out_scale)

    rgb_out = np.asarray(
        _PIL.fromarray((rgb*255).astype(np.uint8)).resize((nw, nh), _PIL.LANCZOS),
        dtype=np.float32) / 255.0

    # ── Optical coverage mask (erode 8 px to avoid edge artefacts) ────────────
    coverage = binary_erosion(rgb_out.max(axis=-1) > 0.004,
                              iterations=8).astype(np.float32)

    # ── HST background fill where JWST has no data ────────────────────────────
    hst_bg = hst_background(gdir, gid, ref_hdr, nw, nh, coverage)
    if hst_bg is not None:
        no_cov = (coverage < 0.5)[..., None]
        rgb_out = rgb_out * (1 - no_cov) + hst_bg * no_cov

    # ── X-ray overlay ─────────────────────────────────────────────────────────
    xray_path = gdir/f"{gid}_large_scale.fits"
    with fits.open(xray_path, memmap=False) as hx:
        xdata = np.nan_to_num(np.asarray(hx[0].data, np.float32))
        xhdr  = hx[0].header

    reproj, _ = reproject_interp((xdata, WCS(xhdr)), WCS(ref_hdr), shape_out=(nh, nw))
    raw = np.maximum(np.nan_to_num(reproj), 0.0) * coverage  # clip to JWST footprint

    pos = raw[raw > 0]
    if len(pos) < 10:
        print("no X-ray — RGB only", flush=True)
        _PIL.fromarray((rgb_out*255).astype(np.uint8)).save(out_path)
        return

    # Tighter smoothing: higher noise floor, less haze spread, stronger power
    cleaned  = np.maximum(raw - np.percentile(pos, 65), 0.0)
    sc       = gaussian_filter(cleaned, sigma=20.0)
    sh       = gaussian_filter(cleaned, sigma=50.0)
    smoothed = (sc + 0.15 * sh) * coverage   # re-apply mask after blur

    pos2 = smoothed[smoothed > 0]
    if len(pos2) < 10:
        print("no X-ray after mask — RGB only", flush=True)
        _PIL.fromarray((rgb_out*255).astype(np.uint8)).save(out_path)
        return

    peak = np.percentile(pos2, 98)
    norm = np.clip(np.log1p(8*np.clip(smoothed/(peak+1e-12), 0, 1))/np.log1p(8), 0, 1)
    norm = norm * coverage
    norm_c = gaussian_filter(norm, sigma=10) * coverage
    if norm_c.max() > 0:
        norm_c /= norm_c.max()

    colors_rgba = [(0,0,0,0), (1.0,0.5,0.7,0.08), (1.0,0.1,0.6,0.19), (0.85,0.0,1.0,0.30)]
    cmap = mcolors.LinearSegmentedColormap.from_list('xp', colors_rgba)
    xray_rgba = cmap(norm**2.2)
    xray_rgba[..., 3] *= coverage   # zero alpha outside JWST footprint
    alpha_ch = xray_rgba[..., 3:4]
    comp = np.clip(xray_rgba[..., :3]*alpha_ch + rgb_out*(1-alpha_ch), 0, 1)

    fig, ax = plt.subplots(figsize=(nw/300, nh/300), dpi=300)
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)
    ax.imshow(comp, origin='upper', interpolation='nearest', zorder=1)

    for lvl, col, lw in [(0.20,'#FFB6C1',0.4), (0.35,'#FF69B4',0.55),
                          (0.58,'#FF1493',0.75), (0.82,'#C71585',1.0)]:
        m = norm_c * coverage
        if lvl < m.max():
            ax.contour(largest_mask(m, lvl), levels=[0.5],
                       colors=[col], linewidths=[lw], alpha=0.85, zorder=3)

    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    sz = out_path.stat().st_size / 1e6
    print(f"✓ ({sz:.1f} MB)", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('samples', nargs='*', default=['CW-All','CW-HCG'],
                    help='CW-All, CW-HCG, both, or positional list')
    pa.add_argument('--cutouts', type=Path, default=DEFAULT_CUTOUTS)
    pa.add_argument('--output',  type=Path, default=DEFAULT_OUT)
    args = pa.parse_args()

    samples = args.samples
    if samples == ['both']:
        samples = ['CW-All', 'CW-HCG']

    for sample in samples:
        base    = args.cutouts / sample
        out_dir = args.output  / sample
        out_dir.mkdir(parents=True, exist_ok=True)
        groups  = sorted([d.name for d in base.iterdir() if d.is_dir()])
        print(f"\n{'='*50}\n{sample}: {len(groups)} groups\n{'='*50}")
        for gid in groups:
            out_path = out_dir / f"group{gid}_xray.png"
            if out_path.exists():
                print(f"  {gid}: skip (exists)")
                continue
            try:
                process_group(gid, base/gid, out_path)
            except Exception as e:
                import traceback
                print(f"  {gid}: ERROR — {e}")
                traceback.print_exc()


if __name__ == '__main__':
    main()
