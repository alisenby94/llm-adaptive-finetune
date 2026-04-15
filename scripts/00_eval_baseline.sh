#!/usr/bin/env bash
# scripts/00_eval_baseline.sh
# ─────────────────────────────────────────────────────────────────────
# Evaluate the pretrained LLaMA-3-8B (zero-shot, no fine-tuning) on
# both test splits.  Results are saved to
#   checkpoints/pretrained_baseline/eval_results.json      ← in-domain
#   checkpoints/pretrained_baseline/eval_ood_results.json  ← OOD
# which visualize.py reads as the baseline reference line.
#
# Run ONCE, after GPU is free from training.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export WANDB_MODE="disabled"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
SPLITS_DIR="data/splits"
OUTDIR="checkpoints/pretrained_baseline"
EVAL_INDOMAIN="$SPLITS_DIR/test_indomain.jsonl"
EVAL_OOD="$SPLITS_DIR/test_ood.jsonl"

mkdir -p "$OUTDIR"

echo "=== Pretrained Baseline Evaluation (zero-shot LLaMA-3-8B) ==="
echo "  Model : $MODEL"
echo "  Output: $OUTDIR"
echo ""

if [[ -f "$OUTDIR/eval_results.json" ]]; then
  echo "  SKIP in-domain (already done): $OUTDIR/eval_results.json"
else
  echo "  ── In-domain eval ──"
  python src/evaluate.py \
    --model_path     "$MODEL"             \
    --test_jsonl     "$EVAL_INDOMAIN"     \
    --output_json    "$OUTDIR/eval_results.json" \
    --batch_size     32                   \
    --max_new_tokens 16
fi

if [[ -f "$OUTDIR/eval_ood_results.json" ]]; then
  echo "  SKIP OOD (already done): $OUTDIR/eval_ood_results.json"
else
  echo "  ── OOD eval ──"
  python src/evaluate.py \
    --model_path     "$MODEL"             \
    --test_jsonl     "$EVAL_OOD"          \
    --output_json    "$OUTDIR/eval_ood_results.json" \
    --batch_size     32                   \
    --max_new_tokens 16
fi

echo ""
echo "Done. Baseline results saved to $OUTDIR/"
echo "Re-run visualize.py to add the zero-shot reference line to figures."
