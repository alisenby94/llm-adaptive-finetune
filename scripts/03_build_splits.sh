#!/usr/bin/env bash
# scripts/03_build_splits.sh
# ─────────────────────────────────────────────────────────────────────
# Step 3: Build the categorised train / test splits (CPU-only).
# Run from the repo root:   bash scripts/03_build_splits.sh
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/base.yaml}"
SCORED_JSONL="data/processed/scored.jsonl"
SPLITS_DIR="data/splits"

if [[ ! -f "$SCORED_JSONL" ]]; then
  echo "ERROR: $SCORED_JSONL not found. Run scripts/02_score_mastery.sh first."
  exit 1
fi

echo "=== Step 3: Build categorical splits ==="
python data/build_splits.py \
  --scored_jsonl "$SCORED_JSONL" \
  --splits_dir   "$SPLITS_DIR"   \
  --config       "$CONFIG"

echo ""
echo "Done.  Splits → $SPLITS_DIR/"
echo "Next:  bash scripts/04_run_sft.sh"
