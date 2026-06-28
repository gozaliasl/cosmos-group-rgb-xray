"""
cosmos_rgb_xray
===============
Pipeline for building publication-quality RGB + X-ray composite images
of COSMOS-Web galaxy groups.

Components
----------
stretch   — asinh + CLAHE per-channel stretching (webbster / PixInsight STF)
blend     — screen, overlay, and luminance blend modes
xray      — X-ray cutout, reprojection, RGBA overlay, coverage mask, HST fill
rgb       — JWST NIRCam + HST F814W RGB construction (asinh + legacy CLAHE)
batch     — catalog-driven batch processing of the full group sample
single    — single-group pipeline (TIFF or FITS input)
"""

__version__ = "0.1.0"
__author__  = "Ghassem Gozaliasl"
