#!/usr/bin/env bash
# =============================================================================
# run_cutouts.sh  —  COSMOS-Web group cutout pipeline (Candide)
#
# Usage:
#   bash scripts/run_cutouts.sh [MODE] [OPTIONS]
#
# Modes (pick one):
#   jwst          Cut JWST NIRCam only
#   hst           Cut HST F814W only
#   xray          Cut X-ray maps only
#   optical       Cut JWST + HST
#   all           Cut JWST + HST + X-ray  (default)
#
# Options:
#   --catalog PATH   Catalog CSV (default: catalogs/top20_cutout_combined.csv)
#   --output  PATH   Output root (default: /n23data2/gozaliasl/groups_cutout/group_inputs)
#   --size    ARCSEC Override cutout size for all groups
#   --res     MAS    Pixel scale in mas (default: 30)
#   --ids     "id1 id2 ..."  Process only these group IDs
#   --overwrite      Re-cut existing files
#
# Examples:
#   bash scripts/run_cutouts.sh all
#   bash scripts/run_cutouts.sh jwst --ids "15 41 376"
#   bash scripts/run_cutouts.sh optical --size 300 --overwrite
#   bash scripts/run_cutouts.sh xray --catalog catalogs/top20_cutout_cw_hcg.csv
# =============================================================================
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

MODE="${1:-all}"
shift || true   # consume mode arg (may not be set)

CATALOG="$REPO_ROOT/catalogs/top20_cutout_combined.csv"
OUTPUT="/n23data2/gozaliasl/groups_cutout/group_inputs"
JWST_DIR="/n23data2/cosmosweb/COSMOS-Web_Jan24/NIRCam/v0.8"
HST_DIR="/n17data/shuntov/COSMOS-Web/Images_HST-ACS/Jan24Tiles"
XRAY_DIR="/n23data2/gozaliasl/xray_maps"
SIZE=""
RES="30"
IDS=""
OVERWRITE=""

# ── Parse options ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --catalog)   CATALOG="$2";   shift 2 ;;
        --output)    OUTPUT="$2";    shift 2 ;;
        --jwst-dir)  JWST_DIR="$2";  shift 2 ;;
        --hst-dir)   HST_DIR="$2";   shift 2 ;;
        --xray-dir)  XRAY_DIR="$2";  shift 2 ;;
        --size)      SIZE="$2";      shift 2 ;;
        --res)       RES="$2";       shift 2 ;;
        --ids)       IDS="$2";       shift 2 ;;
        --overwrite) OVERWRITE="--overwrite"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Build flag list ───────────────────────────────────────────────────────────
FLAGS=""
case "$MODE" in
    jwst)    FLAGS="--jwst" ;;
    hst)     FLAGS="--hst" ;;
    xray)    FLAGS="--xray" ;;
    optical) FLAGS="--jwst --hst" ;;
    all)     FLAGS="--jwst --hst --xray" ;;
    *)       echo "Unknown mode: $MODE  (use: jwst | hst | xray | optical | all)"; exit 1 ;;
esac

[[ -n "$SIZE" ]]     && FLAGS="$FLAGS --size $SIZE"
[[ -n "$RES" ]]      && FLAGS="$FLAGS --res $RES"
[[ -n "$IDS" ]]      && FLAGS="$FLAGS --ids $IDS"
[[ -n "$OVERWRITE" ]] && FLAGS="$FLAGS $OVERWRITE"

# ── Log ───────────────────────────────────────────────────────────────────────
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOG_DIR/cutouts_${MODE}_${TIMESTAMP}.log"

echo "=============================================="
echo "  COSMOS-Web group cutout pipeline"
echo "  Mode     : $MODE"
echo "  Catalog  : $CATALOG"
echo "  Output   : $OUTPUT  (auto-created)"
echo "  JWST dir : $JWST_DIR"
echo "  HST dir  : $HST_DIR"
echo "  X-ray dir: $XRAY_DIR"
echo "  Log      : $LOGFILE"
echo "=============================================="

# ── Run ───────────────────────────────────────────────────────────────────────
python "$SCRIPT_DIR/make_cutouts.py" \
    $FLAGS \
    --catalog  "$CATALOG" \
    --output   "$OUTPUT"  \
    --jwst-dir "$JWST_DIR" \
    --hst-dir  "$HST_DIR"  \
    --xray-dir "$XRAY_DIR" \
    2>&1 | tee "$LOGFILE"

echo ""
echo "Done. Log saved to $LOGFILE"
