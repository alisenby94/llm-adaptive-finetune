"""
Mastery scorer: compute R_k^M for every training fact in the raw dataset.

For each fact k = (subject, relation, object), the pretrained model M is scored
using N_map = 21 cloze-style prompt templates × N_sample = 10 sampled completions
at temperature 0.7 (Ye et al., 2025 §3.1 / Appendix F.1).

    R_k^M = (correct completions) / (N_map × N_sample)

A completion is "correct" if any synonym of the gold answer appears as a
substring in the generated text (case-insensitive).

Output: the input JSONL with an added `mastery_score` field (float ∈ [0, 1]).

Usage:
    python data/mastery_scorer.py \
        --raw_jsonl   data/raw/entity_questions.jsonl \
        --output_jsonl data/processed/scored.jsonl \
        --model_name   meta-llama/Meta-Llama-3-8B \
        --config       configs/base.yaml
"""

import json
import argparse
from pathlib import Path
from typing import List

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import cloze templates and synonym utilities (defined in src/)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.templates import get_cloze_templates
from src.synonyms import get_all_synonyms


# ──────────────────────────────────────────────────────────────────────────────
# Core scoring function
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def score_fact(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    subject: str,
    relation: str,
    answers: List[str],
    n_templates: int = 21,
    n_samples: int = 10,
    temperature: float = 0.7,
    max_new_tokens: int = 16,
    device: str = "cuda",
) -> float:
    """
    Score a single (subject, relation) → answers fact.

    Returns R_k^M ∈ [0, 1]: fraction of template-sample pairs where a synonym
    of the gold answer appears in the generated completion.
    """
    templates = get_cloze_templates(relation, n_templates)
    all_synonyms = get_all_synonyms(answers)

    prompts = [t.format(subject=subject) for t in templates]

    # Tokenize all templates as a batch (left-pad so generation starts aligned)
    tokenizer.padding_side = "left"
    encoding = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    # Generate n_samples continuations per template
    outputs = model.generate(
        **encoding,
        do_sample=True,
        temperature=temperature,
        num_return_sequences=n_samples,
        max_new_tokens=max_new_tokens,
        max_length=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    # outputs: [n_templates * n_samples, seq_len]

    prompt_len = encoding["input_ids"].shape[1]
    correct = 0
    total = len(templates) * n_samples

    for i, output_ids in enumerate(outputs):
        generated = tokenizer.decode(
            output_ids[prompt_len:], skip_special_tokens=True
        ).strip()
        if any(syn.lower() in generated.lower() for syn in all_synonyms):
            correct += 1

    return correct / total if total > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Batch scorer for an entire JSONL
# ──────────────────────────────────────────────────────────────────────────────

def score_dataset(
    raw_jsonl: str,
    output_jsonl: str,
    model_name: str,
    n_templates: int = 7,
    n_samples: int = 3,
    temperature: float = 0.7,
    max_new_tokens: int = 16,
    location_relations: List[str] = None,
    checkpoint_every: int = 50,  # flush checkpoint every N batches (~400 facts)
) -> None:
    """
    Score all training facts in raw_jsonl and write results to output_jsonl.

    Only train-split records from location_relations are scored (dev/test
    records and OOD records are passed through unchanged with mastery_score=None).

    Checkpointing: scores are flushed to <output_jsonl>.ckpt.jsonl every
    `checkpoint_every` batches.  If the process is interrupted, re-running
    will auto-resume from that checkpoint.
    """
    loc_set = set(location_relations or [])
    out_path = Path(output_jsonl)
    ckpt_path = out_path.with_suffix(".ckpt.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all records
    records: list[dict] = []
    with open(raw_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Filter to records that need scoring; use index as stable key
    to_score_idx = [
        i for i, r in enumerate(records)
        if r.get("split") == "train"
        and r.get("relation") in loc_set
        and r.get("subject") is not None
    ]

    # ── Resume from checkpoint if one exists ──────────────────────────────────
    score_map: dict[int, float] = {}
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    score_map[entry["idx"]] = entry["score"]
        print(f"Resumed {len(score_map):,} scores from checkpoint {ckpt_path}")

    remaining_idx = [i for i in to_score_idx if i not in score_map]
    print(
        f"Scoring {len(remaining_idx):,} remaining facts "
        f"({len(to_score_idx):,} total, {len(records):,} records) …"
    )

    if not remaining_idx:
        print("All facts already scored — writing final output.")
    else:
        # Load model only when there is actual work to do
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model {model_name} on {device} …")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "eager"
        print(f"Attention implementation: {attn_impl}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=device,
            attn_implementation=attn_impl,
        )
        model.eval()

        # Score and store by index (avoids id() aliasing bug after dict copy).
        # We batch FACT_BATCH facts per generate() call to increase GPU utilization.
        FACT_BATCH = 8  # 8 facts × 7 templates × 3 samples = 168 parallel seqs

        batches = list(range(0, len(remaining_idx), FACT_BATCH))
        with open(ckpt_path, "a") as ckpt_f:
            for batch_num, batch_start in enumerate(
                tqdm(batches, desc="Mastery scoring")
            ):
                batch_indices = remaining_idx[batch_start : batch_start + FACT_BATCH]
                batch_recs = [records[i] for i in batch_indices]

                # Build all prompts for this fact-batch
                all_prompts: list[str] = []
                per_fact_meta: list[tuple[list[str], int]] = []  # (synonyms, n_tmpl)
                for rec in batch_recs:
                    templates = get_cloze_templates(rec["relation"], n_templates)
                    prompts = [t.format(subject=rec["subject"]) for t in templates]
                    synonyms = get_all_synonyms(rec["answers"])
                    all_prompts.extend(prompts)
                    per_fact_meta.append((synonyms, len(templates)))

                # Tokenize all prompts together (left-pad)
                tokenizer.padding_side = "left"
                encoding = tokenizer(
                    all_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to(device)

                # Single generate call for the whole fact-batch
                outputs = model.generate(
                    **encoding,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=n_samples,
                    max_new_tokens=max_new_tokens,
                    max_length=None,
                    pad_token_id=tokenizer.eos_token_id,
                )
                # outputs: [(n_all_prompts * n_samples), seq_len]
                prompt_padded_len = encoding["input_ids"].shape[1]

                # Decode and tally per fact
                prompt_offset = 0
                for data_idx, (synonyms, n_tmpl) in zip(batch_indices, per_fact_meta):
                    correct = 0
                    for t in range(n_tmpl):
                        for s in range(n_samples):
                            out_idx = (prompt_offset + t) * n_samples + s
                            generated = tokenizer.decode(
                                outputs[out_idx][prompt_padded_len:],
                                skip_special_tokens=True,
                            ).strip()
                            if any(syn.lower() in generated.lower() for syn in synonyms):
                                correct += 1
                    score = correct / (n_tmpl * n_samples)
                    score_map[data_idx] = score
                    prompt_offset += n_tmpl

                    # Append new score to checkpoint file immediately
                    ckpt_f.write(json.dumps({"idx": data_idx, "score": score}) + "\n")

                # Flush to disk every checkpoint_every batches
                if (batch_num + 1) % checkpoint_every == 0:
                    ckpt_f.flush()

    # Write final output and remove checkpoint on success
    with open(out_path, "w") as fout:
        for i, rec in enumerate(records):
            out = dict(rec)
            out["mastery_score"] = score_map.get(i, None)
            fout.write(json.dumps(out) + "\n")

    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"Checkpoint removed: {ckpt_path}")

    print(f"\nWrote {len(records):,} records → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score pretrained model mastery of training facts.")
    parser.add_argument("--raw_jsonl",    required=True,
                        help="Raw EntityQuestions JSONL from data/download.py")
    parser.add_argument("--output_jsonl", required=True,
                        help="Output path for mastery-scored JSONL")
    parser.add_argument("--model_name",   default=None,
                        help="HuggingFace model name (overrides config)")
    parser.add_argument("--config",       default="configs/base.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = args.model_name or cfg["model"]["name"]

    score_dataset(
        raw_jsonl          = args.raw_jsonl,
        output_jsonl       = args.output_jsonl,
        model_name         = model_name,
        n_templates        = cfg["mastery_scoring"]["n_templates"],
        n_samples          = cfg["mastery_scoring"]["n_samples"],
        temperature        = cfg["mastery_scoring"]["temperature"],
        max_new_tokens     = cfg["mastery_scoring"]["max_new_tokens"],
        location_relations = cfg["data"]["location_relations"],
    )
