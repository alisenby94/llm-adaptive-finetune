#!/usr/bin/env bash
# scripts/01_download_data.sh
# ─────────────────────────────────────────────────────────────────────
# Step 1: Download EntityQuestions and build the processed JSONL.
# Run from the repo root:   bash scripts/01_download_data.sh
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Step 1: Download EntityQuestions ==="
python data/download.py --output_dir data/raw

echo ""
echo "Done.  Raw data → data/raw/entity_questions.jsonl"
echo "Next:  bash scripts/02_score_mastery.sh"
