#!/bin/bash
#SBATCH --job-name=hz_rgb_xray
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/automnt/n23data2/gozaliasl/groups_cutouts/hz_detected/logs/hz_rgb_%j.log

# =============================================================================
# SLURM job: RGB + X-ray composites for high-z detected groups
#
# Submit (both samples):
#   sbatch scripts/run_hz_rgb_xray.sh
#
# Submit one sample only:
#   sbatch --export=SAMPLES=CW-All scripts/run_hz_rgb_xray.sh
#   sbatch --export=SAMPLES=CW-HCG scripts/run_hz_rgb_xray.sh
#
# Override at submission time:
#   sbatch --export=OVERWRITE=1 scripts/run_hz_rgb_xray.sh
#   sbatch --export=JOBS=4,ANNOTATE=1 scripts/run_hz_rgb_xray.sh
# =============================================================================

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
module load intelpython/3-2024.2.0

REPO=/automnt/n23data2/gozaliasl/cosmos-group-rgb-xray
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# ── Paths ─────────────────────────────────────────────────────────────────────
CUTOUT_BASE=/automnt/n23data2/gozaliasl/groups_cutouts/hz_detected
OUTPUT_BASE=/automnt/n23data2/gozaliasl/groups_cutouts/hz_detected_rgb
CATALOG="$REPO/catalogs/hz_detected_cutout.csv"
LOG_DIR="$CUTOUT_BASE/logs"

mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

# ── Options (override via --export at submission time) ────────────────────────
SAMPLES="${SAMPLES:-CW-All CW-HCG}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK}}"
OVERWRITE="${OVERWRITE:-0}"
ANNOTATE="${ANNOTATE:-0}"
TIFF="${TIFF:-0}"
MAX_PX="${MAX_PX:-4000}"

# ── Build optional flags ──────────────────────────────────────────────────────
EXTRA_FLAGS=""
[[ "$OVERWRITE" == "1" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --overwrite"
[[ "$ANNOTATE"  == "1" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --annotate"
[[ "$TIFF"      == "1" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --tiff"

# ── Log header ────────────────────────────────────────────────────────────────
echo "======================================================"
echo "  hz RGB+X-ray pipeline — SLURM job ${SLURM_JOB_ID:-local}"
echo "  Samples   : $SAMPLES"
echo "  Input base: $CUTOUT_BASE"
echo "  Output    : $OUTPUT_BASE"
echo "  Catalog   : $CATALOG"
echo "  Jobs      : $JOBS"
echo "  Overwrite : $OVERWRITE"
echo "  Annotate  : $ANNOTATE"
echo "  Max px    : $MAX_PX"
echo "  Start     : $(date)"
echo "======================================================"

# ── Run each sample ───────────────────────────────────────────────────────────
for SAMPLE in $SAMPLES; do
    DATA_DIR="$CUTOUT_BASE/$SAMPLE"
    OUT_DIR="$OUTPUT_BASE/$SAMPLE"
    mkdir -p "$OUT_DIR"

    N=$(ls -d "$DATA_DIR"/*/  2>/dev/null | wc -l)
    echo ""
    echo "── $SAMPLE ── $N groups → $OUT_DIR"

    python -m cosmos_rgb_xray.batch \
        --catalog    "$CATALOG"  \
        --data-root  "$DATA_DIR" \
        --output-dir "$OUT_DIR"  \
        --jobs       "$JOBS"     \
        --max-px     "$MAX_PX"   \
        --verbose    \
        $EXTRA_FLAGS

    echo "── $SAMPLE done: $(date)"
done

echo ""
echo "======================================================"
echo "  All done: $(date)"
echo "======================================================"
