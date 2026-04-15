#!/usr/bin/env bash
# scripts/06_visualize.sh
# ─────────────────────────────────────────────────────────────────────
# Step 6: Generate presentation figures from all completed runs.
#
# Run any time — skips figures for which data is not yet available.
# Re-running overwrites previous figures with latest results.
#
# Usage:
#   bash scripts/06_visualize.sh
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

OUTPUT_DIR="${OUTPUT_DIR:-figures}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-results}"
HEATMAP_SCALE="${HEATMAP_SCALE:-1920}"

echo "=== Step 6: Generate Figures ==="
echo "  Checkpoints : $CKPT_DIR/"
echo "  Results     : $RESULTS_DIR/"
echo "  Output      : $OUTPUT_DIR/"

python src/visualize.py \
    --ckpt_dir      "$CKPT_DIR"      \
    --results_dir   "$RESULTS_DIR"   \
    --output_dir    "$OUTPUT_DIR"    \
    --heatmap_scale "$HEATMAP_SCALE"

echo ""
echo "Done.  Open $OUTPUT_DIR/ to review figures."
