"""
CBQA accuracy evaluation with synonym expansion.

Implements the evaluation metric from Ye et al. (2025):
  - Greedy decoding, max_new_tokens = 16
  - A prediction is correct if any synonym of any gold answer appears
    as a substring of the generated text (case-insensitive)
  - Reports per-mastery-category accuracy and overall mean

Usage:
    python src/evaluate.py \
        --model_path  checkpoints/Dtrain_2_1920_seed42/final_model \
        --test_jsonl  data/splits/test_indomain.jsonl \
        --output_json results/Dtrain_2_1920_seed42_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.synonyms import get_all_synonyms
from src.templates import make_sft_prompt

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Minimal inference dataset (no labels, just questions)
# ──────────────────────────────────────────────────────────────────────────────

class EvalDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.records: list[dict] = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec.get("question") and rec.get("answers"):
                        self.records.append(rec)

    def __len__(self): return len(self.records)
    def __getitem__(self, idx): return self.records[idx]


def _collate(batch: list[dict]) -> dict:
    return {
        "questions":  [b["question"]             for b in batch],
        "answers":    [b["answers"]              for b in batch],
        "categories": [b.get("mastery_category") for b in batch],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation function
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate_cbqa(
    model:           "AutoModelForCausalLM",
    tokenizer:       "AutoTokenizer",
    jsonl_path:      str,
    batch_size:      int  = 32,
    max_new_tokens:  int  = 16,
    device:          Optional[str] = None,
) -> dict:
    """
    Evaluate a model on a CBQA JSONL test file.

    Returns
    -------
    dict with keys:
        accuracy           : float — overall accuracy (mean across answer categories)
        per_category       : dict  — accuracy per mastery_category (0–4)
        n_correct          : int
        n_total            : int
        predictions        : list  — per-sample dicts for error analysis
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset    = EvalDataset(jsonl_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=_collate)

    category_correct: dict[int | None, int] = {}
    category_total:   dict[int | None, int] = {}
    predictions: list[dict] = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        prompts    = [make_sft_prompt(q) for q in batch["questions"]]
        encodings  = tokenizer(
            prompts,
            return_tensors = "pt",
            padding        = True,
            truncation     = True,
            max_length     = 256,
        ).to(device)

        outputs = model.generate(
            **encodings,
            max_new_tokens = max_new_tokens,
            do_sample      = False,          # greedy decoding (paper §3.3)
            pad_token_id   = tokenizer.eos_token_id,
        )

        prompt_len = encodings["input_ids"].shape[1]
        generated  = tokenizer.batch_decode(
            outputs[:, prompt_len:], skip_special_tokens=True
        )

        for gen, gold_answers, cat in zip(generated, batch["answers"], batch["categories"]):
            all_syns = get_all_synonyms(gold_answers)
            correct  = any(syn.lower() in gen.lower() for syn in all_syns)

            category_correct[cat] = category_correct.get(cat, 0) + int(correct)
            category_total[cat]   = category_total.get(cat, 0) + 1

            predictions.append({
                "prediction": gen.strip(),
                "answers":    gold_answers,
                "correct":    correct,
                "category":   cat,
            })

    # Compute per-category and overall accuracy
    per_category = {
        cat: category_correct.get(cat, 0) / max(category_total.get(cat, 1), 1)
        for cat in sorted(set(category_total) | set(category_correct))
    }

    # Overall accuracy = mean across all test categories (Ye et al. §3.3)
    # Filter to the 5 standard mastery-category keys (0–4) for the main metric
    standard_cats = [c for c in per_category if c in (0, 1, 2, 3, 4)]
    if standard_cats:
        overall_accuracy = sum(per_category[c] for c in standard_cats) / len(standard_cats)
    else:
        n_total   = sum(category_total.values())
        n_correct = sum(category_correct.values())
        overall_accuracy = n_correct / max(n_total, 1)

    n_total   = sum(category_total.values())
    n_correct = sum(category_correct.values())

    return {
        "accuracy":     overall_accuracy,
        "per_category": {str(k): v for k, v in per_category.items()},
        "n_correct":    n_correct,
        "n_total":      n_total,
        "predictions":  predictions,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model on CBQA.")
    parser.add_argument("--model_path",   required=True,
                        help="Path to a saved model directory (from train.py)")
    parser.add_argument("--test_jsonl",   required=True,
                        help="Test split JSONL (e.g. data/splits/test_indomain.jsonl)")
    parser.add_argument("--output_json",  default=None,
                        help="Optional path to save results JSON")
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    args = parser.parse_args()

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Support both full checkpoints and LoRA adapter checkpoints
    adapter_config = Path(args.model_path) / "adapter_config.json"
    if adapter_config.exists():
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model_path,
            dtype      = torch.bfloat16,
            device_map = device,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype      = torch.bfloat16,
            device_map = device,
        )

    results = evaluate_cbqa(
        model          = model,
        tokenizer      = tokenizer,
        jsonl_path     = args.test_jsonl,
        batch_size     = args.batch_size,
        max_new_tokens = args.max_new_tokens,
        device         = device,
    )

    print(f"\nAccuracy: {results['accuracy']:.4f}  ({results['n_correct']}/{results['n_total']})")
    for cat, acc in sorted(results["per_category"].items()):
        print(f"  Category {cat}: {acc:.4f}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Omit per-sample predictions from the saved JSON unless needed
        save = {k: v for k, v in results.items() if k != "predictions"}
        with open(out, "w") as f:
            json.dump(save, f, indent=2)
        print(f"\nSaved → {out}")
