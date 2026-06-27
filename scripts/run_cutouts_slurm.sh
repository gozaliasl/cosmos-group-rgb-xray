#!/bin/bash
#SBATCH --job-name=rgb_cutouts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/automnt/n23data2/gozaliasl/groups_cutouts/logs/cutouts_%j.log

# =============================================================================
# SLURM job: cut JWST + HST + X-ray stamps for COSMOS-Web galaxy groups
#
# Submit:
#   sbatch scripts/run_cutouts_slurm.sh
#
# Override mode at submission time:
#   sbatch --export=MODE=optical scripts/run_cutouts_slurm.sh
#   sbatch --export=MODE=xray,IDS="15 41 376" scripts/run_cutouts_slurm.sh
#
# Modes: jwst | hst | xray | optical | all (default)
# =============================================================================

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
module load intelpython/3-2024.2.0

REPO=/automnt/n23data2/gozaliasl/cosmos-group-rgb-xray
cd "$REPO"

pip install -e . -q --no-build-isolation 2>/dev/null || true

# ── Paths ─────────────────────────────────────────────────────────────────────
CATALOG="$REPO/catalogs/top20_cutout_combined.csv"
OUTPUT="/automnt/n23data2/gozaliasl/groups_cutouts/group_inputs"
JWST_DIR="/automnt/n23data2/cosmosweb/COSMOS-Web_Jan24/NIRCam/v0.8"
HST_DIR="/automnt/n17data/shuntov/COSMOS-Web/Images_HST-ACS/Jan24Tiles"
XRAY_DIR="/automnt/n23data2/gozaliasl/xray_maps"
LOG_DIR="/automnt/n23data2/gozaliasl/groups_cutouts/logs"

mkdir -p "$LOG_DIR" "$OUTPUT"

# ── Mode (override via --export at sbatch time) ───────────────────────────────
MODE="${MODE:-all}"
IDS="${IDS:-}"

# ── Build flags ───────────────────────────────────────────────────────────────
FLAGS=""
case "$MODE" in
    jwst)    FLAGS="--jwst" ;;
    hst)     FLAGS="--hst" ;;
    xray)    FLAGS="--xray" ;;
    optical) FLAGS="--jwst --hst" ;;
    all)     FLAGS="--jwst --hst --xray" ;;
    *)       echo "Unknown MODE=$MODE"; exit 1 ;;
esac

[[ -n "$IDS" ]] && FLAGS="$FLAGS --ids $IDS"

# ── Log header ────────────────────────────────────────────────────────────────
echo "======================================================"
echo "  COSMOS-Web RGB cutout pipeline — SLURM job $SLURM_JOB_ID"
echo "  Mode      : $MODE"
echo "  Catalog   : $CATALOG"
echo "  Output    : $OUTPUT"
echo "  JWST dir  : $JWST_DIR"
echo "  HST dir   : $HST_DIR"
echo "  X-ray dir : $XRAY_DIR"
echo "  CPUs      : $SLURM_CPUS_PER_TASK"
echo "  Start     : $(date)"
echo "======================================================"

# ── Run ───────────────────────────────────────────────────────────────────────
python "$REPO/scripts/make_cutouts.py" \
    $FLAGS \
    --catalog  "$CATALOG"  \
    --output   "$OUTPUT"   \
    --jwst-dir "$JWST_DIR" \
    --hst-dir  "$HST_DIR"  \
    --xray-dir "$XRAY_DIR"

echo ""
echo "Done: $(date)"
