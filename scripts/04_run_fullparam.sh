#!/usr/bin/env bash
# scripts/04_run_fullparam.sh
# ─────────────────────────────────────────────────────────────────────
# Full-parameter SFT sweep for direct comparison with Ye et al. (2025).
#
# Runs cat=2 × {60,120,240,480,960,1920} × seed=42 (6 runs).
# Uses DeepSpeed ZeRO-2 + CPU optimizer offload to fit LLaMA-3-8B on 32 GB.
#
# Checkpoints saved to checkpoints/fullparam_Dtrain_... (separate from LoRA
# results in checkpoints/Dtrain_... — no risk of overwriting).
#
# Usage:
#   bash scripts/04_run_fullparam.sh
#
# Requires:
#   ~37 GB free system RAM (for FP32 optimizer states offloaded to CPU)
#   ~32 GB GPU VRAM (weights 16 GB + gradients 16 GB)
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# Activate virtual environment so torchrun/python are on PATH
source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export DS_BUILD_OPS=0
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export WANDB_MODE="disabled"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Clear any stale DeepSpeed JIT build cache from previous failed attempts.
rm -rf ~/.cache/torch_extensions/py312_cu130/cpu_adam 2>/dev/null || true
rm -rf ~/.cache/torch_extensions/py312_cu130/fused_adam 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
CONFIG="configs/fullparam.yaml"
SPLITS_DIR="data/splits"
CKPT_DIR="checkpoints"
EVAL_INDOMAIN="$SPLITS_DIR/test_indomain.jsonl"
EVAL_OOD="$SPLITS_DIR/test_ood.jsonl"

echo "=== Full-Parameter SFT (paper replication) ==="
echo "  Config : $CONFIG"
echo "  Scales : 60 120 240 480 960 1920"
echo "  Cat    : 2 (50-75% mastery)  Seed: 42"
echo ""

run_fullparam() {
  local cat=$1 scale=$2 seed=$3
  local split="$SPLITS_DIR/Dtrain_${cat}_${scale}_seed${seed}.jsonl"
  local outdir="$CKPT_DIR/fullparam_Dtrain_${cat}_${scale}_seed${seed}"

  if [[ ! -f "$split" ]]; then
    echo "  SKIP (no split file): $split"
    return
  fi
  if [[ -f "$outdir/eval_ood_results.json" ]]; then
    echo "  SKIP (already done): $outdir"
    return
  fi

  echo "  ── Training: cat=$cat  scale=$scale  seed=$seed (full-param) ──"

  # Only run torchrun if training hasn't already completed cleanly.
  if [[ ! -f "$outdir/training_complete" ]]; then
    # No resume — auto-saves are disabled under DeepSpeed to avoid the 90 GB
    # optimizer-state checkpoint write that OOM-kills the process.  If a run
    # crashes, we restart from scratch (each full-param run is ≤ ~90 min).
    torchrun --nproc_per_node=1 --master_port=29500 src/train.py \
      --split      "$split"         \
      --output_dir "$outdir"        \
      --config     "$CONFIG"        \
      --eval_split "$EVAL_INDOMAIN"
  else
    echo "  Training already complete (sentinel found), skipping torchrun."
  fi

  # train.py now saves final_model in-process (BF16 GPU shard gather, no big
  # optimizer-state write).  No post-training conversion step needed.

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

for scale in 60 120 240 480 960 1920; do
  run_fullparam 2 "$scale" 42
done

echo ""
echo "Done.  Full-param checkpoints → $CKPT_DIR/fullparam_Dtrain_*"
echo "Compare with LoRA results in  → $CKPT_DIR/Dtrain_*"
