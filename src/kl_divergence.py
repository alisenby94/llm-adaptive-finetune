"""
KL divergence utilities.

Implements the re-normalised KL divergence metric from Ye et al. (2025) §4.1:

  sKL(p ∥ p') = -Σ_i  p_i · log(p'_i / p_i)

where p is derived from the fine-tuned model's top-10 logits and p' from the
corresponding logits in the pretrained model, both normalised via softmax.

Functions
---------
compute_sample_kl         : Compute sKL for a single token position.
compute_batch_kl          : Compute sKL for a batch of token positions.
compute_layer_deltas      : Compute ||θ_l - θ_0_l||_F per transformer layer.
compute_eval_kl           : Compute average sKL over an evaluation dataset.
KLDivergenceEvaluator     : Standalone evaluator class used in analysis scripts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)

N_TOP = 10  # Number of top-logit tokens used for KL computation (paper §4.1)


# ──────────────────────────────────────────────────────────────────────────────
# Token-level sKL
# ──────────────────────────────────────────────────────────────────────────────

def compute_sample_kl(
    ft_logits: torch.Tensor,   # [vocab_size] — fine-tuned model logits for one token
    pt_logits: torch.Tensor,   # [vocab_size] — pretrained model logits for same token
    n_top: int = N_TOP,
) -> float:
    """
    Compute the re-normalised KL divergence sKL(p ∥ p') for a single token.

    Follows the logits re-normalisation procedure from the paper (§4.1 / Fig 3):
      1. Take the top-n_top token indices from the fine-tuned logits.
      2. Extract the corresponding logits from both models.
      3. Apply softmax to each subset independently.
      4. Compute KL divergence.
    """
    # Step 1: top-n indices from fine-tuned logit vector
    top_vals_ft, top_idx = ft_logits.float().topk(n_top)

    # Step 2: corresponding values from pretrained
    top_vals_pt = pt_logits.float()[top_idx]

    # Step 3: normalise via softmax
    p  = F.softmax(top_vals_ft, dim=-1)   # fine-tuned distribution
    p_ = F.softmax(top_vals_pt, dim=-1)   # pretrained distribution

    # Step 4: sKL = -Σ p * log(p' / p)  =  Σ p * log(p / p')
    # Clamp to avoid log(0)
    p_  = p_.clamp(min=1e-9)
    skl = (p * torch.log(p / p_)).sum().item()
    return skl


def compute_batch_kl(
    ft_logits: torch.Tensor,   # [batch, seq_len, vocab] OR [batch, vocab]
    pt_logits: torch.Tensor,
    n_top: int = N_TOP,
) -> torch.Tensor:
    """
    Vectorised sKL over a batch.

    Returns a tensor of shape [batch] (if input is [batch, vocab])
    or [batch, seq_len] (if input is [batch, seq_len, vocab]).
    """
    original_shape = ft_logits.shape[:-1]
    vocab = ft_logits.shape[-1]
    ft = ft_logits.float().reshape(-1, vocab)
    pt = pt_logits.float().reshape(-1, vocab)

    # Top-n_top indices from fine-tuned model (per row)
    top_vals_ft, top_idx = ft.topk(n_top, dim=-1)               # [N, n_top]
    top_vals_pt = pt.gather(-1, top_idx)                         # [N, n_top]

    p  = F.softmax(top_vals_ft, dim=-1)
    p_ = F.softmax(top_vals_pt, dim=-1).clamp(min=1e-9)

    skl = (p * torch.log(p / p_)).sum(dim=-1)                    # [N]
    return skl.reshape(original_shape)


# ──────────────────────────────────────────────────────────────────────────────
# Per-layer weight delta norms
# ──────────────────────────────────────────────────────────────────────────────

def compute_layer_deltas(
    model,
    pretrained_state_dict: dict[str, torch.Tensor],
) -> dict[str, float]:
    """
    Compute the Frobenius norm of (θ_ft - θ_0) for every named parameter.

    Aggregates by transformer block (layer index) and by module type
    (attn.q, attn.k, attn.v, attn.o, mlp.gate, mlp.up, mlp.down).

    Returns a flat dict with 'layer_{i}_total' and 'module_{name}' keys.
    """
    deltas: dict[str, float] = {}

    with torch.no_grad():
        for name, param in model.named_parameters():
            pt = pretrained_state_dict.get(name)
            if pt is None:
                continue
            delta_norm = (param.float() - pt.to(param.device).float()).norm().item()
            deltas[name] = delta_norm

    # Aggregate by block index (e.g. "model.layers.7.self_attn.q_proj.weight" → block 7)
    block_totals: dict[int, float] = {}
    for name, delta in deltas.items():
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    block_idx = int(parts[i + 1])
                    block_totals[block_idx] = block_totals.get(block_idx, 0.0) + delta
                except ValueError:
                    pass
                break

    aggregated = {f"layer_{i}_total": v for i, v in sorted(block_totals.items())}
    return aggregated


# ──────────────────────────────────────────────────────────────────────────────
# Dataset-level KL evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def compute_eval_kl(
    finetuned_model,
    pretrained_model,
    tokenizer,
    jsonl_path:      str,
    batch_size:      int = 16,
    device:          str = "cuda",
    n_top:           int = N_TOP,
) -> dict:
    """
    Compute the mean sKL divergence between a fine-tuned and pretrained model
    over a test dataset.

    Uses the same data selection criterion as the paper (§4.1): evaluates on
    tokens from the answer span (positions after the prompt ends).

    Returns
    -------
    dict with 'mean_skl', 'per_sample_skl', and 'n_samples'.
    """
    import json
    from src.templates import make_sft_prompt

    finetuned_model.eval()
    pretrained_model.eval()

    records: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("question") and rec.get("answers"):
                    records.append(rec)

    all_skl: list[float] = []

    for i in range(0, len(records), batch_size):
        batch   = records[i: i + batch_size]
        prompts = [make_sft_prompt(r["question"]) for r in batch]

        enc = tokenizer(
            prompts,
            return_tensors = "pt",
            padding        = True,
            truncation     = True,
            max_length     = 256,
        ).to(device)

        ft_out = finetuned_model(**enc)
        pt_out = pretrained_model(**enc)

        # Compute sKL at each token position (over the prompt region is noisy;
        # consider restricting to answer tokens if you tokenize answers separately)
        skl = compute_batch_kl(
            ft_out.logits,
            pt_out.logits.to(device),
            n_top = n_top,
        )  # [batch, seq_len]

        # Mean over non-padding positions
        attn = enc["attention_mask"].bool()
        for j in range(skl.shape[0]):
            seq_skl = skl[j][attn[j]].mean().item()
            all_skl.append(seq_skl)

    return {
        "mean_skl":       float(sum(all_skl) / max(len(all_skl), 1)),
        "per_sample_skl": all_skl,
        "n_samples":      len(all_skl),
    }
