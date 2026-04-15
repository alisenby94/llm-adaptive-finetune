#!/usr/bin/env bash
# scripts/04b_run_fullparam_short.sh
# ─────────────────────────────────────────────────────────────────────
# Short full-param runs for presentation: cats 0,1,3,4 × scales 60,120,240
# (12 runs, ≤30 steps each, completes before cat=2 long runs resume)
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export DS_BUILD_OPS=0
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export WANDB_MODE="disabled"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
rm -rf ~/.cache/torch_extensions/py312_cu130/cpu_adam 2>/dev/null || true
rm -rf ~/.cache/torch_extensions/py312_cu130/fused_adam 2>/dev/null || true

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
CONFIG="configs/fullparam.yaml"
SPLITS_DIR="data/splits"
CKPT_DIR="checkpoints"
EVAL_INDOMAIN="$SPLITS_DIR/test_indomain.jsonl"
EVAL_OOD="$SPLITS_DIR/test_ood.jsonl"

echo "=== Full-Param SHORT runs (cats 0,1,3,4 × scales 60,120,240) ==="

run_fullparam() {
  local cat=$1 scale=$2 seed=$3
  local split="$SPLITS_DIR/Dtrain_${cat}_${scale}_seed${seed}.jsonl"
  local outdir="$CKPT_DIR/fullparam_Dtrain_${cat}_${scale}_seed${seed}"

  if [[ ! -f "$split" ]]; then
    echo "  SKIP (no split file): $split"
    return
  fi
  if [[ -f "$outdir/eval_ood_results.json" && -f "$outdir/eval_results.json" ]]; then
    echo "  SKIP (already done): $outdir"
    return
  fi

  echo "  ── Training: cat=$cat  scale=$scale  seed=$seed ──"
  if [[ ! -f "$outdir/training_complete" ]]; then
    torchrun --nproc_per_node=1 --master_port=29500 src/train.py \
      --split      "$split"         \
      --output_dir "$outdir"        \
      --config     "$CONFIG"        \
      --eval_split "$EVAL_INDOMAIN"
  else
    echo "  Training already complete (sentinel found), skipping torchrun."
  fi

  echo "  ── In-domain eval: $outdir ──"
  python src/evaluate.py \
    --model_path     "$outdir/final_model" \
    --test_jsonl     "$EVAL_INDOMAIN"       \
    --output_json    "$outdir/eval_results.json" \
    --batch_size     32                    \
    --max_new_tokens 16

  echo "  ── OOD eval: $outdir ──"
  python src/evaluate.py \
    --model_path     "$outdir/final_model" \
    --test_jsonl     "$EVAL_OOD"           \
    --output_json    "$outdir/eval_ood_results.json" \
    --batch_size     32                    \
    --max_new_tokens 16
}

for scale in 60 120 240; do
  for cat in 0 1 3 4; do
    run_fullparam "$cat" "$scale" 42
  done
done

echo ""
echo "Done. Short runs complete — ready for presentation."
echo "Resume long runs (480/960/1920 + remaining cats) with 04_run_fullparam.sh"
