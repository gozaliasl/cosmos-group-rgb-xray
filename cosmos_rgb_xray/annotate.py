"""
Add RA/Dec axes and a physical scale bar to RGB + X-ray composite images.

Uses matplotlib WCSAxes so tick labels are in proper sky coordinates.
Scale bar shows a fixed physical size (default 500 kpc) computed from
the group redshift using Planck 2018 cosmology.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from astropy.io import fits
from astropy.wcs import WCS
from astropy.cosmology import FlatLambdaCDM

COSMO = FlatLambdaCDM(H0=67.4, Om0=0.315)


# ── Physical scale bar ────────────────────────────────────────────────────────

def physical_to_arcsec(size_kpc: float, redshift: float) -> float:
    """Convert physical size [kpc] to angular size [arcsec] at redshift z."""
    if redshift <= 0:
        return 60.0
    d_a = COSMO.angular_diameter_distance(redshift).to("kpc").value
    return size_kpc / d_a * 206_265.0


def arcsec_to_pixels(arcsec: float, wcs: WCS) -> float:
    """Angular size in arcsec → pixels for a given WCS."""
    pix_scale = abs(wcs.wcs.cdelt[0]) * 3600.0   # arcsec/pixel
    return arcsec / pix_scale


# ── Main annotation function ──────────────────────────────────────────────────

def annotate_and_save(
    rgb: np.ndarray,
    ref_hdr: fits.Header,
    output_path: Path,
    redshift: float = 0.0,
    scale_kpc: float = 500.0,
    save_tiff: bool = False,
    max_px: int = 4000,
    dpi: int = 300,
    fontsize: int = 17,
    tick_color: str = "white",
    scalebar_color: str = "white",
    margin_color: str = "white",   # background outside the image axes
    verbose: bool = False,
) -> None:
    """
    Render RGB array with WCS RA/Dec axes and a physical scale bar, save to disk.

    Parameters
    ----------
    rgb          : float32 HxWx3 array in [0,1]
    ref_hdr      : FITS header with valid WCS
    output_path  : PNG output path (.tiff saved alongside if save_tiff=True)
    redshift     : group redshift (used for scale bar physical size)
    scale_kpc    : physical length of scale bar in kpc (default 500)
    save_tiff    : also write 16-bit LZW TIFF
    max_px       : resample if image larger than this on either axis
    dpi          : output DPI (300 for publication)
    fontsize     : axis label / tick font size
    tick_color   : colour of tick marks and labels
    scalebar_color: colour of the scale bar and its label
    """
    from PIL import Image as PILImage

    # ── Resize if needed ──────────────────────────────────────────────────────
    h, w = rgb.shape[:2]
    if max(h, w) > max_px:
        scale  = max_px / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_rs = PILImage.fromarray((rgb * 255).astype(np.uint8)).resize(
            (new_w, new_h), PILImage.LANCZOS)
        rgb = np.asarray(img_rs, dtype=np.float32) / 255.0
        # Scale WCS pixel scale accordingly
        ref_hdr = ref_hdr.copy()
        ref_hdr["NAXIS1"] = new_w
        ref_hdr["NAXIS2"] = new_h
        ref_hdr["CRPIX1"] = ref_hdr.get("CRPIX1", w / 2) * scale
        ref_hdr["CRPIX2"] = ref_hdr.get("CRPIX2", h / 2) * scale
        for key in ["CDELT1", "CDELT2"]:
            if key in ref_hdr:
                ref_hdr[key] = ref_hdr[key] / scale
        if "CD1_1" in ref_hdr:
            for key in ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]:
                if key in ref_hdr:
                    ref_hdr[key] = ref_hdr[key] / scale
        h, w = new_h, new_w

    wcs = WCS(ref_hdr)

    # ── Figure with WCSAxes ───────────────────────────────────────────────────
    # Add margins (in inches) for axis labels so image fills the rest cleanly.
    # Without this, matplotlib reserves space for the Dec label but the figure
    # width doesn't account for it, producing a black strip on the left.
    label_margin_in = fontsize / dpi * 6   # ~label height in inches
    fig_w = w / dpi + label_margin_in * 1.5   # left: Dec label + ticks
    fig_h = h / dpi + label_margin_in * 1.2   # bottom: RA label + ticks

    left_frac   = label_margin_in * 1.5 / fig_w
    bottom_frac = label_margin_in * 1.2 / fig_h

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(margin_color)
    ax  = fig.add_axes(
        [left_frac, bottom_frac, 1 - left_frac, 1 - bottom_frac],
        projection=wcs,
    )

    ax.imshow(rgb, origin="lower", interpolation="nearest",
              aspect="equal", vmin=0, vmax=1)
    ax.set_facecolor("black")

    # ── RA / Dec axes ─────────────────────────────────────────────────────────
    ra_ax  = ax.coords["ra"]
    dec_ax = ax.coords["dec"]

    # Labels and tick values sit on the white margin → use dark colour
    label_color = "black" if margin_color == "white" else tick_color

    ra_ax.set_axislabel("RA (J2000)",  color=label_color, fontsize=fontsize)
    dec_ax.set_axislabel("Dec (J2000)", color=label_color, fontsize=fontsize)

    ra_ax.set_ticklabel(color=label_color, fontsize=fontsize - 1,
                        exclude_overlapping=True)
    dec_ax.set_ticklabel(color=label_color, fontsize=fontsize - 1,
                         exclude_overlapping=True)

    ra_ax.set_ticks_position("b")
    dec_ax.set_ticks_position("l")

    ra_ax.display_minor_ticks(True)
    dec_ax.display_minor_ticks(True)

    ra_ax.set_major_formatter("hh:mm:ss")
    dec_ax.set_major_formatter("dd:mm:ss")

    # Spines and tick marks on the image border
    for spine in ax.spines.values():
        spine.set_edgecolor(label_color)
    ax.tick_params(colors=label_color, which="both")

    # ── Scale bar ─────────────────────────────────────────────────────────────
    bar_arcsec = physical_to_arcsec(scale_kpc, redshift)
    bar_pix    = arcsec_to_pixels(bar_arcsec, wcs)
    bar_pix    = min(bar_pix, w * 0.25)   # never wider than 25% of image

    # Position: lower-left corner with padding — pushed up enough to clear RA ticks
    pad_x = w * 0.05
    pad_y = h * 0.08
    x0    = pad_x
    x1    = pad_x + bar_pix
    y_bar = pad_y

    outline = [pe.withStroke(linewidth=3, foreground="black")]

    ax.plot([x0, x1], [y_bar, y_bar],
            color=scalebar_color, linewidth=2.5,
            transform=ax.get_transform("pixel"),
            path_effects=outline)
    # End ticks
    for x in [x0, x1]:
        ax.plot([x, x], [y_bar - h * 0.008, y_bar + h * 0.008],
                color=scalebar_color, linewidth=2.0,
                transform=ax.get_transform("pixel"),
                path_effects=outline)

    # Label
    label = f"{scale_kpc:.0f} kpc" if scale_kpc < 1000 else \
            f"{scale_kpc/1000:.1f} Mpc"
    ax.text((x0 + x1) / 2, y_bar + h * 0.025, label,
            color=scalebar_color, fontsize=fontsize,
            ha="center", va="bottom",
            transform=ax.get_transform("pixel"),
            path_effects=outline)

    # Redshift annotation (upper right)
    if redshift > 0:
        ax.text(w * 0.97, h * 0.97, f"z = {redshift:.3f}",
                color="white", fontsize=fontsize,
                ha="right", va="top",
                transform=ax.get_transform("pixel"),
                path_effects=outline)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                pad_inches=0.02, facecolor=margin_color)
    plt.close(fig)
    if verbose:
        sz = output_path.stat().st_size / 1e6
        print(f"  PNG  → {output_path}  ({sz:.1f} MB)", flush=True)

    # 16-bit TIFF — render figure to array then save
    if save_tiff:
        tiff_path = output_path.with_suffix(".tiff")
        # Re-read PNG, scale 8-bit → 16-bit per channel, save as TIFF
        png_arr   = np.asarray(PILImage.open(output_path).convert("RGB"))  # uint8
        img16_arr = (png_arr.astype(np.uint32) * 257).astype(np.uint16)    # 0-255 → 0-65535
        # PIL doesn't support uint16 RGB directly — write via tifffile if available
        try:
            import tifffile
            tifffile.imwrite(str(tiff_path), img16_arr,
                             photometric="rgb", compression="lzw",
                             resolution=(dpi, dpi))
        except ImportError:
            # Fallback: save channels separately and merge, or just save 8-bit TIFF
            PILImage.open(output_path).save(
                tiff_path, compression="tiff_lzw", dpi=(dpi, dpi))
        if verbose:
            sz = tiff_path.stat().st_size / 1e6
            print(f"  TIFF → {tiff_path}  ({sz:.1f} MB)", flush=True)
