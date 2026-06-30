"""
V2 configuration system — YAML-based with validated defaults.

Usage:
    from cosmos_rgb_xray_v2.config import load_config, V2Config
    cfg = load_config("my_config.yaml")  # or load_config() for defaults
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml


# ---------------------------------------------------------------------------
# Band priority order: when multiple bands are available, use first match
# ---------------------------------------------------------------------------
# R channel: redder → cooler/older stellar populations
BAND_PRIORITY_R = ["F444W", "F410M", "F356W", "F277W"]
# G channel: middle wavelength
BAND_PRIORITY_G = ["F277W", "F200W", "F150W", "F115W", "F098M"]
# B channel: bluer → star-forming, younger populations
BAND_PRIORITY_B = ["F115W", "F606W", "F435W", "F814W", "F098M", "F150W"]
# Luminance (sharpening layer): sharpest / highest-resolution band
BAND_PRIORITY_L = ["F150W", "F115W", "F200W"]


@dataclass
class StretchConfig:
    method: str = "ghs"           # ghs | asinh | trilogy
    # GHS parameters
    ghs_b: float = 4.5            # stretch intensity
    ghs_D: float = 1.0            # local stretch at inflection point
    ghs_SP_pct: float = 0.5       # shadow protection percentile
    ghs_HP_pct: float = 97.0      # highlight — lower so galaxy disks land in mid-range
    ghs_LP: float = 0.12          # linear point — higher preserves faint disk outskirts
    # Luminance layer
    use_luminance: bool = True     # blend sharpest band as luminance layer
    luminance_weight: float = 0.25
    # Tone curve after stretch
    tone_gamma: float = 0.75       # x^gamma applied above sky floor (< 1 brightens midtones)


@dataclass
class BackgroundConfig:
    subtract: bool = True
    method: str = "sigma_clip"    # sigma_clip | plane_fit | none
    sigma: float = 3.0
    iterations: int = 5
    box_size: int = 256           # for local background estimation
    filter_size: int = 3          # median filter on background map


@dataclass
class XrayConfig:
    # Smoothing
    smooth_sigma_extended: float = 160.0   # px, for extended/bright groups
    smooth_sigma_compact: float = 25.0     # px, for compact groups
    smooth_haze_sigma: float = 200.0
    smooth_haze_weight: float = 0.10
    # Hole filling
    fill_holes: bool = True
    fill_kernel: int = 15
    # Normalization
    bg_percentile: float = 58.0           # background reference percentile
    norm_power: float = 1.0               # post-log power
    # Display
    alpha_peak_extended: float = 0.55
    alpha_peak_compact: float = 0.40
    # Contours
    contour_levels: Tuple[float, ...] = (0.20, 0.38, 0.62, 0.85)
    contour_linewidths: Tuple[float, ...] = (0.7, 0.9, 1.1, 1.3)
    contour_alpha: float = 0.85
    # No-contour version uses heavier smoothing
    nocontour_smooth_factor: float = 1.2
    nocontour_bg_percentile: float = 52.0


@dataclass
class GroundFillConfig:
    """Ground-based data to fill JWST survey-edge gaps."""
    enable: bool = True
    bands: List[str] = field(default_factory=lambda: ["F606W", "F435W", "F098M",
                                                       "Ks", "H", "J", "Y",
                                                       "i", "r", "g"])
    coverage_threshold: float = 0.05   # pixel is "covered" if > this fraction of max
    blend_width_px: int = 50           # feather edge of ground fill blend


@dataclass
class OutputConfig:
    dpi: int = 300
    max_px: int = 6000
    save_png: bool = True
    save_tiff: bool = False
    save_nocontours: bool = True    # always produce both versions
    jpeg_quality: int = 95
    colorspace: str = "sRGB"       # sRGB | P3 | AdobeRGB


@dataclass
class V2Config:
    stretch: StretchConfig = field(default_factory=StretchConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    xray: XrayConfig = field(default_factory=XrayConfig)
    ground_fill: GroundFillConfig = field(default_factory=GroundFillConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Band priorities (overridable per-run)
    band_r: List[str] = field(default_factory=lambda: list(BAND_PRIORITY_R))
    band_g: List[str] = field(default_factory=lambda: list(BAND_PRIORITY_G))
    band_b: List[str] = field(default_factory=lambda: list(BAND_PRIORITY_B))
    band_l: List[str] = field(default_factory=lambda: list(BAND_PRIORITY_L))

    # X-ray extended threshold
    xray_extended_snr: float = 10.0
    xray_extended_cutout_arcsec: float = 240.0


def load_config(path: Optional[Path] = None) -> V2Config:
    """Load a YAML config file, falling back to defaults for any missing key."""
    cfg = V2Config()
    if path is None or not Path(path).exists():
        return cfg
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    def _update(dc, d):
        for k, v in d.items():
            if hasattr(dc, k):
                attr = getattr(dc, k)
                if hasattr(attr, '__dataclass_fields__') and isinstance(v, dict):
                    _update(attr, v)
                else:
                    setattr(dc, k, v)
    _update(cfg, raw)
    return cfg


def save_config(cfg: V2Config, path: Path) -> None:
    """Save current config to YAML."""
    with open(path, "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False, sort_keys=False)
