#!/usr/bin/env bash
# scripts/04_run_sft.sh
# ─────────────────────────────────────────────────────────────────────
# Step 4: Run SFT experiments.
#
# Three modes (set via env var MODE=):
#
#   critical (default)  — 2 runs: cat=2 × {240,1920} × seed=42
#                         Fastest sanity check that the pipeline works.
#
#   demo                — 75 runs: 5 cats × 5 scales × 3 seeds
#                         Enough to reproduce both Phenomena for a
#                         presentation. ~5–7 h on RTX 5090. Run overnight.
#
#   full                — 50 runs: 5 cats × {240,1920} × 5 seeds
#                         Full paper replication with max seeds.
#
# Usage:
#   bash scripts/04_run_sft.sh                 # critical (2 runs)
#   MODE=demo bash scripts/04_run_sft.sh       # presentation sweep
#   MODE=full bash scripts/04_run_sft.sh       # paper replication
#
# Env overrides:
#   MODEL=meta-llama/Meta-Llama-3-8B
#   CONFIG=configs/base.yaml
#   DEEPSPEED=1   (use DeepSpeed ZeRO-2 CPU offload — requires ~48 GB RAM)
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
# Prevent deepspeed from trying to compile ops (no nvcc on this system)
export DS_BUILD_OPS=0
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# Disable wandb unless a key is already configured
export WANDB_MODE="${WANDB_MODE:-disabled}"
# Reduce CUDA memory fragmentation (helps fit 8B model + backward pass in 32 GB)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
CONFIG="${CONFIG:-configs/base.yaml}"
SPLITS_DIR="data/splits"
CKPT_DIR="checkpoints"
EVAL_INDOMAIN="$SPLITS_DIR/test_indomain.jsonl"
EVAL_OOD="$SPLITS_DIR/test_ood.jsonl"
MODE="${MODE:-critical}"

# Optionally inject DeepSpeed config
DS_ARGS=""
if [[ "${DEEPSPEED:-0}" == "1" ]]; then
  DS_ARGS="--deepspeed_config configs/deepspeed_zero2.json"
  echo "  [DeepSpeed ZeRO-2 enabled]"
fi

# ── Helper ─────────────────────────────────────────────────────────
run_sft() {
  local cat=$1 scale=$2 seed=$3
  local split="$SPLITS_DIR/Dtrain_${cat}_${scale}_seed${seed}.jsonl"
  local outdir="$CKPT_DIR/Dtrain_${cat}_${scale}_seed${seed}"

  if [[ ! -f "$split" ]]; then
    echo "  SKIP (no split file): $split"
    return
  fi
  if [[ -f "$outdir/eval_ood_results.json" ]]; then
    echo "  SKIP (already done): $outdir"
    return
  fi

  echo ""
  echo "  ── Training: cat=$cat  scale=$scale  seed=$seed ──"
  python src/train.py \
    --split      "$split"         \
    --output_dir "$outdir"        \
    --config     "$CONFIG"        \
    --eval_split "$EVAL_INDOMAIN" \
    $DS_ARGS

  # Run OOD evaluation separately (train.py only evaluates in-domain)
  echo "  ── OOD eval: $outdir ──"
  python src/evaluate.py \
    --model_path     "$outdir/final_model" \
    --test_jsonl     "$EVAL_OOD"           \
    --output_json    "$outdir/eval_ood_results.json" \
    --batch_size     32                    \
    --max_new_tokens 16
}

echo "=== Step 4: SFT Training (mode=$MODE) ==="

case "$MODE" in
  demo)
    # 5 cats × [60, 120, 240, 480, 960, 1920] × 3 seeds = 90 runs
    # Covers full scale curve + all mastery categories for presentation plots.
    echo "  Mode: DEMO  (5 categories × 6 scales × 3 seeds = 90 runs)"
    for cat in 0 1 2 3 4; do
      for scale in 60 120 240 480 960 1920; do
        for seed in 42 43 44; do
          run_sft "$cat" "$scale" "$seed"
        done
      done
    done
    ;;
  full)
    # 5 cats × {240,1920} × 5 seeds = 50 runs — full paper replication
    echo "  Mode: FULL  (5 categories × 2 scales × 5 seeds = 50 runs)"
    for cat in 0 1 2 3 4; do
      for scale in 240 1920; do
        for seed in 42 43 44 45 46; do
          run_sft "$cat" "$scale" "$seed"
        done
      done
    done
    ;;
  critical|*)
    echo "  Mode: CRITICAL  (cat=2, scales=240+1920, seed=42 — 2 runs)"
    run_sft 2 240  42
    run_sft 2 1920 42
    ;;
esac

echo ""
echo "Done.  Checkpoints → $CKPT_DIR/"
echo "Next:  bash scripts/05_run_restoration.sh"
