"""
cosmos_rgb_xray
===============
Pipeline for building publication-quality RGB + X-ray composite images
of COSMOS-Web galaxy groups.

Components
----------
stretch   — asinh + CLAHE per-channel stretching (webbster / PixInsight STF)
blend     — screen, overlay, and luminance blend modes
xray      — X-ray cutout, reprojection, and alpha overlay
rgb       — JWST NIRCam + HST F814W RGB construction
batch     — catalog-driven batch processing of the full group sample
single    — single-group pipeline (for group 15 / pre-built TIFF workflow)
"""

__version__ = "0.1.0"
__author__  = "Ghassem Gozaliasl"
