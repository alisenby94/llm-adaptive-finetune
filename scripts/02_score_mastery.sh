#!/usr/bin/env bash
# scripts/02_score_mastery.sh
# ─────────────────────────────────────────────────────────────────────
# Step 2: Score pretrained-model mastery for every training fact.
# Requires a GPU.  Roughly 5–15 min on RTX 5090 for LLaMA-3-8B.
# Run from the repo root:   bash scripts/02_score_mastery.sh
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
CONFIG="${CONFIG:-configs/base.yaml}"
RAW_JSONL="data/raw/entity_questions.jsonl"
SCORED_JSONL="data/processed/scored.jsonl"

if [[ ! -f "$RAW_JSONL" ]]; then
  echo "ERROR: $RAW_JSONL not found. Run scripts/01_download_data.sh first."
  exit 1
fi

mkdir -p data/processed

echo "=== Step 2: Score pretrained mastery ==="
echo "  Model : $MODEL"
echo "  Input : $RAW_JSONL"
echo "  Output: $SCORED_JSONL"

python data/mastery_scorer.py \
  --raw_jsonl    "$RAW_JSONL"    \
  --output_jsonl "$SCORED_JSONL" \
  --model_name   "$MODEL"        \
  --config       "$CONFIG"

echo ""
echo "Done.  Scored data → $SCORED_JSONL"
echo "Next:  bash scripts/03_build_splits.sh"
