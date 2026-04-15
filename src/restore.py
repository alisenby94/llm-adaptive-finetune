"""
Post-hoc parameter restoration baseline.

Implements the parameter restoration procedure from Ye et al. (2025):

  After SFT, rank all scalar parameters by |θ_ft - θ_0|.
  Reset the top-k% most-changed parameters back to their pretrained values.

This is the *post-hoc* baseline we need to reproduce before building the
in-training adaptive variant (AFT-P).

Algorithm:
  1. Compute Δ_i = |θ_ft_i - θ_0_i| for every scalar parameter i.
  2. Find the threshold T such that exactly k% of parameters have Δ_i ≥ T
     (using np.partition for O(n) selection instead of full sort).
  3. For each parameter i where Δ_i ≥ T, set θ_restored_i = θ_0_i.
  4. Save the restored model.

Memory note: for LLaMA-3-8B, collecting all 8B deltas requires ~32 GB of
float32 RAM (8B × 4 bytes).  The script processes parameters layer-by-layer
but still concatenates deltas for threshold estimation.  Ensure you have
sufficient system RAM.  A future optimisation could use reservoir sampling
for an approximate threshold (~1% error).

Usage:
    python src/restore.py \
        --finetuned_dir   checkpoints/Dtrain_2_1920_seed42/final_model \
        --pretrained_name meta-llama/Meta-Llama-3-8B \
        --output_root     results/restoration \
        --config          configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluate import evaluate_cbqa

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Core restoration function
# ──────────────────────────────────────────────────────────────────────────────

def _in_transformer_region(name: str) -> bool:
    """Match the paper's region filter: transformer layers excluding norms."""
    return "layers" in name and "norm" not in name


def restore_top_k(
    finetuned_state_dict:  dict[str, torch.Tensor],
    pretrained_state_dict: dict[str, torch.Tensor],
    k_percent:             float,
) -> tuple[dict[str, torch.Tensor], dict]:
    """
    Restore the top-k% most-changed parameters to their pretrained values.

    Parameters
    ----------
    finetuned_state_dict  : Model state dict after SFT.
    pretrained_state_dict : Original pretrained model state dict.
    k_percent             : Percentage of parameters to restore (e.g. 10.0 = 10%).

    Returns
    -------
    restored_state_dict : Modified state dict with top-k% params reset.
    stats               : Dict with restoration statistics.
    """
    if k_percent <= 0:
        return {k: v.clone() for k, v in finetuned_state_dict.items()}, {
            "k_percent": 0, "n_restored": 0, "n_total": 0
        }

    # ── Pass 1: collect all |Δ| values to find the threshold ──────────
    logger.info("Computing parameter deltas for k=%.2f%% …", k_percent)
    delta_chunks: list[np.ndarray] = []
    n_total = 0

    EPS = 1e-8
    for name in tqdm(sorted(finetuned_state_dict), desc="Pass 1 (deltas)", leave=False):
        if not _in_transformer_region(name):
            continue
        pt = pretrained_state_dict.get(name)
        if pt is None:
            continue
        ft    = finetuned_state_dict[name].float().flatten().numpy()
        pt_np = pt.float().flatten().numpy()
        # Use relative difference (per paper's official implementation):
        # |ft - pt| / (|pt| + eps)
        delta = np.abs(ft - pt_np) / (np.abs(pt_np) + EPS)
        delta_chunks.append(delta)
        n_total += len(delta)

    all_deltas = np.concatenate(delta_chunks)  # [n_total]
    n_restore  = max(1, int(n_total * k_percent / 100))

    # O(n) threshold selection via partition (no full sort needed)
    # np.partition(arr, -k)[-k] = kth largest element
    threshold = float(np.partition(all_deltas, -n_restore)[-n_restore])
    logger.info("Threshold for top %.2f%%: %.6e  (%d / %d params)", k_percent,
                threshold, n_restore, n_total)

    # ── Pass 2: apply restoration ──────────────────────────────────────
    restored: dict[str, torch.Tensor] = {}
    n_restored_actual = 0

    for name in tqdm(sorted(finetuned_state_dict), desc="Pass 2 (restore)", leave=False):
        ft_tensor = finetuned_state_dict[name]
        pt_tensor = pretrained_state_dict.get(name)

        # Non-transformer params (embeddings, norms, lm_head) are never touched
        if not _in_transformer_region(name):
            restored[name] = ft_tensor.clone()
            continue

        if pt_tensor is None:
            restored[name] = ft_tensor.clone()
            continue

        ft   = ft_tensor.float()
        pt   = pt_tensor.float()
        # Relative difference mask (matches paper's official implementation)
        EPS = 1e-8
        mask = (ft - pt).abs() / (pt.abs() + EPS) >= threshold

        restored[name] = torch.where(mask, pt, ft).to(ft_tensor.dtype)
        n_restored_actual += int(mask.sum().item())

    stats = {
        "k_percent":        k_percent,
        "threshold":        threshold,
        "n_restored":       n_restored_actual,
        "n_total":          n_total,
        "actual_k_percent": 100 * n_restored_actual / max(n_total, 1),
    }
    logger.info("Restored %d / %d params  (%.2f%%)",
                n_restored_actual, n_total, stats["actual_k_percent"])
    return restored, stats


# ──────────────────────────────────────────────────────────────────────────────
# Full restoration sweep
# ──────────────────────────────────────────────────────────────────────────────

def restoration_sweep(
    finetuned_dir:   str,
    pretrained_name: str,
    output_root:     str,
    test_jsonl:      str,
    k_values:        list[float],
    eval_batch_size: int = 32,
    eval_max_tokens: int = 16,
) -> list[dict]:
    """
    Run the restoration sweep over all k_values and evaluate CBQA accuracy.

    Returns a list of result dicts (one per k value).
    """
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(finetuned_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load fine-tuned weights to CPU (will be passed to restore_top_k)
    logger.info("Loading fine-tuned model from %s …", finetuned_dir)
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        finetuned_dir,
        torch_dtype = torch.bfloat16,
        device_map  = "cpu",
    )
    finetuned_sd = {k: v.clone() for k, v in finetuned_model.state_dict().items()}

    # Load pretrained weights to CPU
    logger.info("Loading pretrained weights from %s …", pretrained_name)
    pretrained_sd = {
        k: v.cpu() for k, v in
        AutoModelForCausalLM.from_pretrained(
            pretrained_name,
            torch_dtype = torch.bfloat16,
        ).state_dict().items()
    }

    all_results: list[dict] = []

    for k in k_values:
        logger.info("─── k = %.2f%% ───", k)
        restored_sd, stats = restore_top_k(finetuned_sd, pretrained_sd, k)

        # Load restored weights into the model for evaluation
        finetuned_model.load_state_dict(restored_sd)
        finetuned_model.to(device)

        eval_results = evaluate_cbqa(
            model          = finetuned_model,
            tokenizer      = tokenizer,
            jsonl_path     = test_jsonl,
            batch_size     = eval_batch_size,
            max_new_tokens = eval_max_tokens,
            device         = device,
        )
        finetuned_model.to("cpu")  # move back to CPU to free VRAM

        result = {
            "k_percent": k,
            "accuracy":  eval_results["accuracy"],
            "per_category": eval_results["per_category"],
            "n_correct": eval_results["n_correct"],
            "n_total":   eval_results["n_total"],
            **{f"stats_{kk}": vv for kk, vv in stats.items()},
        }
        all_results.append(result)
        logger.info("k=%.2f%%  accuracy=%.4f", k, result["accuracy"])

    # Save summary
    summary_path = out_root / "restoration_sweep.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Summary → %s", summary_path)
    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Post-hoc parameter restoration sweep.")
    parser.add_argument("--finetuned_dir",   required=True,
                        help="Fine-tuned model directory (from train.py)")
    parser.add_argument("--pretrained_name", default=None,
                        help="Pretrained model name or path (overrides config)")
    parser.add_argument("--test_jsonl",      required=True,
                        help="Test JSONL to evaluate on")
    parser.add_argument("--output_root",     required=True,
                        help="Directory to save restoration results")
    parser.add_argument("--config",          default="configs/base.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pretrained_name = args.pretrained_name or cfg["model"]["name"]
    k_values        = cfg["restoration"]["k_values"]
    k_values_pct    = [k * 100 for k in k_values]  # config stores as fractions

    restoration_sweep(
        finetuned_dir   = args.finetuned_dir,
        pretrained_name = pretrained_name,
        output_root     = args.output_root,
        test_jsonl      = args.test_jsonl,
        k_values        = k_values_pct,
        eval_batch_size = cfg["evaluation"]["batch_size"],
        eval_max_tokens = cfg["evaluation"]["max_new_tokens"],
    )
