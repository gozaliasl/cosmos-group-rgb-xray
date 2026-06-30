"""
cosmos_rgb_xray_v2 — next-generation RGB + X-ray pipeline.

V2 is developed independently of v1. V1 remains accessible via git tag v1.0.0.
If v2 has any issues, run: git checkout v1.0.0

Key improvements over v1:
  - Per-band sigma-clipped sky subtraction
  - Generalized Hyperbolic Stretch (GHS) replacing asinh
  - Auto band selection from all available instruments
  - Ground-based fill (HSC/UltraVISTA) at survey edges
  - YAML config system — no hardcoded parameters
  - Improved X-ray: adaptive noise floor + better colormap
  - Proper logging throughout
"""
__version__ = "2.0.0"
