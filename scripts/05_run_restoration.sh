#!/usr/bin/env bash
# scripts/05_run_restoration.sh
# ─────────────────────────────────────────────────────────────────────
# Step 5: Post-hoc parameter restoration sweep.
#
# For each fine-tuned checkpoint, restores the top-k% most-changed
# parameters (k ∈ {1,3,5,10,20,40,60}%) and evaluates CBQA accuracy.
# Reproduces Table 4 of Ye et al. (2025).
#
# Usage:
#   bash scripts/05_run_restoration.sh
#
# Defaults to the two critical checkpoints.
# Set FULL=1 to sweep all available checkpoints.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
CONFIG="${CONFIG:-configs/base.yaml}"
CKPT_DIR="checkpoints"
RESULTS_DIR="results/restoration"
TEST_INDOMAIN="data/splits/test_indomain.jsonl"
TEST_OOD="data/splits/test_ood.jsonl"
FULL="${FULL:-0}"

run_restoration() {
  local ckpt_name=$1
  local finetuned_dir="$CKPT_DIR/$ckpt_name/final_model"
  local out_dir="$RESULTS_DIR/$ckpt_name"

  if [[ ! -d "$finetuned_dir" ]]; then
    echo "  SKIP (no checkpoint): $finetuned_dir"
    return
  fi

  echo ""
  echo "  ── Restoration sweep: $ckpt_name (in-domain) ──"
  python src/restore.py \
    --finetuned_dir   "$finetuned_dir"               \
    --pretrained_name "$MODEL"                        \
    --test_jsonl      "$TEST_INDOMAIN"               \
    --output_root     "$out_dir/indomain"            \
    --config          "$CONFIG"

  echo ""
  echo "  ── Restoration sweep: $ckpt_name (OOD) ──"
  python src/restore.py \
    --finetuned_dir   "$finetuned_dir"               \
    --pretrained_name "$MODEL"                        \
    --test_jsonl      "$TEST_OOD"                    \
    --output_root     "$out_dir/ood"                 \
    --config          "$CONFIG"
}

echo "=== Step 5: Parameter Restoration Sweep ==="

if [[ "$FULL" == "1" ]]; then
  echo "  Mode: ALL checkpoints in $CKPT_DIR/"
  for ckpt_dir in "$CKPT_DIR"/*/; do
    ckpt_name="$(basename "$ckpt_dir")"
    run_restoration "$ckpt_name"
  done
else
  echo "  Mode: Critical checkpoints (cat=2, scales=240+1920, seed=42)"
  run_restoration "Dtrain_2_240_seed42"
  run_restoration "Dtrain_2_1920_seed42"
fi

echo ""
echo "Done.  Results → $RESULTS_DIR/"
