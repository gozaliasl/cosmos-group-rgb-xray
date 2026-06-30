#!/bin/bash
#SBATCH --job-name=ground_cutouts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/automnt/n23data2/gozaliasl/groups_cutouts/logs/ground_cutouts_%j.log

# =============================================================================
# SLURM job: cut HSC g/r/i + UltraVista J/H/Ks stamps for groups that have
# partial or no JWST coverage (border groups).
#
# These stamps are used to fill black regions in the RGB+X-ray composites.
#
# Submit:
#   sbatch scripts/run_ground_cutouts.sh
#
# Run only HSC or only UltraVista:
#   sbatch --export=MODE=hsc    scripts/run_ground_cutouts.sh
#   sbatch --export=MODE=uvista scripts/run_ground_cutouts.sh
#
# Run on a specific catalog:
#   sbatch --export=CATALOG=/path/to/my.csv scripts/run_ground_cutouts.sh
#
# Overwrite existing cutouts:
#   sbatch --export=OVERWRITE=1 scripts/run_ground_cutouts.sh
# =============================================================================

set -euo pipefail

module load intelpython/3-2024.2.0

REPO=/automnt/n23data2/gozaliasl/cosmos-group-rgb-xray
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# ── Paths ─────────────────────────────────────────────────────────────────────
CATALOG="${CATALOG:-$REPO/catalogs/top20_cutout_combined.csv}"
OUTPUT="${OUTPUT:-/automnt/n23data2/gozaliasl/groups_cutouts/group_inputs}"
COSMOS2020_DIR="/n08data/COSMOS2020/images"
UVISTA_DIR="/automnt/n23data1/UltraVista/DR4-RC2"
LOG_DIR="/automnt/n23data2/gozaliasl/groups_cutouts/logs"

mkdir -p "$LOG_DIR"

# ── Options ───────────────────────────────────────────────────────────────────
MODE="${MODE:-both}"        # hsc | uvista | hst | clutch-hst | both | all
OVERWRITE="${OVERWRITE:-0}"

OVERWRITE_FLAG=""
[[ "$OVERWRITE" == "1" ]] && OVERWRITE_FLAG="--overwrite"

case "$MODE" in
    hsc)        BAND_FLAGS="--hsc" ;;
    uvista)     BAND_FLAGS="--uvista" ;;
    hst)        BAND_FLAGS="--hst" ;;
    clutch-hst) BAND_FLAGS="--clutch-hst" ;;
    both)       BAND_FLAGS="--hsc --uvista" ;;
    all)        BAND_FLAGS="--hsc --uvista --hst --clutch-hst" ;;
    *)          echo "Unknown MODE=$MODE"; exit 1 ;;
esac

echo "======================================================"
echo "  Ground-based cutout pipeline — SLURM job ${SLURM_JOB_ID:-local}"
echo "  Mode          : $MODE"
echo "  Catalog       : $CATALOG"
echo "  Output        : $OUTPUT"
echo "  COSMOS2020 dir: $COSMOS2020_DIR"
echo "  UltraVista dir: $UVISTA_DIR"
echo "  Overwrite     : $OVERWRITE"
echo "  Start         : $(date)"
echo "======================================================"

python "$REPO/scripts/make_cutouts.py" \
    $BAND_FLAGS \
    --catalog        "$CATALOG"       \
    --output         "$OUTPUT"        \
    --cosmos2020-dir "$COSMOS2020_DIR" \
    --uvista-dir     "$UVISTA_DIR"    \
    $OVERWRITE_FLAG

echo ""
echo "Done: $(date)"
