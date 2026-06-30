#!/bin/bash
#SBATCH --job-name=top20_rgb_xray
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/automnt/n23data2/gozaliasl/groups_cutouts/top20/logs/top20_rgb_%j.log

# =============================================================================
# SLURM job: RGB + X-ray composites for top-20% SNR groups
#
# Submit (both samples):
#   sbatch scripts/run_top20_rgb_xray.sh
#
# Submit one sample only:
#   sbatch --export=SAMPLES=CW-All  scripts/run_top20_rgb_xray.sh
#   sbatch --export=SAMPLES=CW-HCG  scripts/run_top20_rgb_xray.sh
#
# Override at submission time:
#   sbatch --export=OVERWRITE=1 scripts/run_top20_rgb_xray.sh
# =============================================================================

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
module load intelpython/3-2024.2.0

REPO=/automnt/n23data2/gozaliasl/cosmos-group-rgb-xray
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# ── Paths ─────────────────────────────────────────────────────────────────────
CUTOUT_BASE=/automnt/n23data2/gozaliasl/groups_cutouts/top20
OUTPUT_BASE=/automnt/n23data2/gozaliasl/groups_cutouts/top20_rgb_v2
CATALOG="$REPO/catalogs/top20_cutout_combined.csv"
LOG_DIR="$CUTOUT_BASE/logs"

mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

# ── Options (override via --export at submission time) ────────────────────────
SAMPLES="${SAMPLES:-CW-All CW-HCG}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK}}"
OVERWRITE="${OVERWRITE:-0}"
RGB_METHOD="${RGB_METHOD:-trilogy}"

# ── Build optional flags ──────────────────────────────────────────────────────
EXTRA_FLAGS=""
[[ "$OVERWRITE" == "1" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --overwrite"

# ── Log header ────────────────────────────────────────────────────────────────
echo "======================================================"
echo "  top20 RGB+X-ray pipeline — SLURM job ${SLURM_JOB_ID:-local}"
echo "  Samples   : $SAMPLES"
echo "  Input base: $CUTOUT_BASE"
echo "  Output    : $OUTPUT_BASE"
echo "  Catalog   : $CATALOG"
echo "  Jobs      : $JOBS"
echo "  Overwrite : $OVERWRITE"
echo "  Start     : $(date)"
echo "======================================================"

# ── Run each sample ───────────────────────────────────────────────────────────
for SAMPLE in $SAMPLES; do
    DATA_DIR="$CUTOUT_BASE/$SAMPLE"
    OUT_DIR="$OUTPUT_BASE/$SAMPLE"
    mkdir -p "$OUT_DIR"

    N=$(ls -d "$DATA_DIR"/*/ 2>/dev/null | wc -l)
    echo ""
    echo "── $SAMPLE ── $N groups → $OUT_DIR"

    python -m cosmos_rgb_xray_v2.batch \
        --catalog    "$CATALOG"      \
        --data-root  "$DATA_DIR"     \
        --output-dir "$OUT_DIR"      \
        --jobs       "$JOBS"         \
        --rgb-method "$RGB_METHOD"   \
        --verbose    \
        $EXTRA_FLAGS

    echo "── $SAMPLE done: $(date)"
done

echo ""
echo "======================================================"
echo "  All done: $(date)"
echo "======================================================"
