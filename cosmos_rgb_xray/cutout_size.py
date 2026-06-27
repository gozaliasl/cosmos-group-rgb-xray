"""
Redshift-aware cutout size calculator for galaxy groups.

Strategy
--------
The cutout is set to cover a fixed physical radius around the group
(default 1.5 Mpc, encompassing the virial region for most groups),
converted to arcseconds at the group redshift using a flat ΛCDM cosmology.

A floor and ceiling are applied so that:
  - Very low-z groups (large apparent size) are not cropped too tightly
  - Very high-z groups (tiny apparent size) still get a meaningful stamp

Typical outputs
---------------
  z=0.1  → ~1800"  (30')   nearby extended group
  z=0.3  → ~650"   (11')   intermediate
  z=0.5  → ~430"   ( 7')
  z=1.0  → ~240"   ( 4')   high-z compact

Default floor/ceiling keep stamps between 3' and 30'.

Usage
-----
  from cosmos_rgb_xray.cutout_size import cutout_arcsec
  size = cutout_arcsec(z=0.35)               # → ~580"
  size = cutout_arcsec(z=0.11, r_mpc=2.0)   # group 15, larger radius
"""
from __future__ import annotations

from astropy.cosmology import FlatLambdaCDM

# Planck 2018 (Aghanim+2020) — consistent with COSMOS photo-z analysis
COSMO = FlatLambdaCDM(H0=67.4, Om0=0.315)

# Physical aperture radius in Mpc — covers most of the X-ray emitting ICM
DEFAULT_R_MPC = 1.5

# Angular size limits [arcsec]
SIZE_FLOOR   = 180.0   # 3'  — minimum even for high-z compact groups
SIZE_CEILING = 1800.0  # 30' — maximum for very nearby extended groups


def cutout_arcsec(
    z: float,
    r_mpc: float = DEFAULT_R_MPC,
    floor: float = SIZE_FLOOR,
    ceiling: float = SIZE_CEILING,
) -> float:
    """
    Physical radius r_mpc [Mpc] → angular diameter [arcsec] at redshift z.

    Parameters
    ----------
    z       : group redshift
    r_mpc   : physical radius to cover [Mpc]  (diameter = 2 × r_mpc)
    floor   : minimum cutout size [arcsec]
    ceiling : maximum cutout size [arcsec]

    Returns
    -------
    Cutout size (diameter) in arcseconds, clamped to [floor, ceiling].
    """
    if z <= 0:
        return ceiling
    # Angular diameter distance [Mpc]
    d_a = COSMO.angular_diameter_distance(z).value
    # 1 radian = d_a Mpc  →  r_mpc Mpc = r_mpc/d_a rad = r_mpc/d_a * 206265 "
    diameter_arcsec = 2.0 * r_mpc / d_a * 206_265.0
    return float(max(floor, min(ceiling, diameter_arcsec)))


def assign_cutout_sizes(
    catalog_path,
    r_mpc: float = DEFAULT_R_MPC,
    z_col: str = "z",
    out_col: str = "cutout_arcsec",
) -> "pd.DataFrame":
    """
    Read a group catalog CSV and fill / update the cutout_arcsec column.

    If cutout_arcsec is already present it is overwritten with the
    redshift-computed value (use this to standardise an existing catalog).

    Returns the updated DataFrame (does not write to disk).
    """
    import pandas as pd
    df = pd.read_csv(catalog_path)
    if z_col not in df.columns:
        raise ValueError(f"Column '{z_col}' not found in {catalog_path}")
    df[out_col] = df[z_col].apply(lambda z: cutout_arcsec(z, r_mpc))
    return df


if __name__ == "__main__":
    # Quick sanity check
    print(f"{'z':>6}  {'size [arcsec]':>14}  {'size [arcmin]':>14}")
    print("-" * 40)
    for z in [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50]:
        s = cutout_arcsec(z)
        print(f"{z:6.2f}  {s:14.1f}  {s/60:14.1f}")
