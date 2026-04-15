"""
PyTorch Dataset for supervised fine-tuning (SFT) on the CBQA task.

Reads a JSONL split file (output of data/build_splits.py) and returns
tokenized prompt+answer sequences ready for causal-LM training.

The loss is masked on the prompt tokens so the model only learns to
predict the answer span.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.templates import make_sft_prompt, make_sft_full


class CBQADataset(Dataset):
    """
    Tokenized CBQA dataset for SFT.

    Each item is a dict with:
        input_ids      : [seq_len]   full prompt + answer token IDs
        attention_mask : [seq_len]
        labels         : [seq_len]   same as input_ids but -100 on prompt tokens
        question       : str         original question string (for evaluation)
        answers        : list[str]   list of gold answer strings
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        split_filter: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        jsonl_path   : Path to a JSONL split file.
        tokenizer    : Loaded HuggingFace tokenizer (must have a pad token set).
        max_length   : Maximum sequence length; longer sequences are truncated.
        split_filter : If given ('train', 'dev', 'test'), skip records from
                       other splits.  Useful when the JSONL mixes splits.
        """
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.records: list[dict] = []

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if split_filter and rec.get("split") != split_filter:
                    continue
                if not rec.get("question") or not rec.get("answers"):
                    continue
                self.records.append(rec)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        rec      = self.records[idx]
        question = rec["question"]
        # Use the first answer as the training target (consistent with paper)
        answer   = rec["answers"][0]

        prompt   = make_sft_prompt(question)
        full_seq = make_sft_full(question, answer)

        # Tokenize the full sequence
        full_enc = self.tokenizer(
            full_seq,
            max_length  = self.max_length,
            truncation  = True,
            padding     = False,
            return_tensors = "pt",
        )
        input_ids      = full_enc["input_ids"].squeeze(0)       # [L]
        attention_mask = full_enc["attention_mask"].squeeze(0)  # [L]

        # Determine where the answer tokens start so we can mask labels
        prompt_enc = self.tokenizer(
            prompt,
            max_length  = self.max_length,
            truncation  = True,
            padding     = False,
            return_tensors = "pt",
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        # Labels: -100 on prompt tokens, actual ids on answer tokens
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            # Keep raw strings for evaluation
            "question": question,
            "answers":  rec["answers"],
        }

    # ------------------------------------------------------------------
    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """
        Pad a list of variable-length samples into a single batch.
        Called by DataLoader when used as collate_fn.
        """
        # Separate string fields from tensor fields
        questions = [b.pop("question") for b in batch]
        answers   = [b.pop("answers")  for b in batch]

        max_len = max(b["input_ids"].size(0) for b in batch)

        padded_input_ids      = []
        padded_attention_mask = []
        padded_labels         = []

        for b in batch:
            length = b["input_ids"].size(0)
            pad    = max_len - length

            # Right-pad input_ids with 0 (will be masked by attention_mask)
            padded_input_ids.append(
                torch.cat([b["input_ids"],
                           torch.zeros(pad, dtype=torch.long)])
            )
            padded_attention_mask.append(
                torch.cat([b["attention_mask"],
                           torch.zeros(pad, dtype=torch.long)])
            )
            # Right-pad labels with -100 (ignored by cross-entropy)
            padded_labels.append(
                torch.cat([b["labels"],
                           torch.full((pad,), -100, dtype=torch.long)])
            )

        out = {
            "input_ids":      torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels":         torch.stack(padded_labels),
            "question": questions,
            "answers":  answers,
        }
        return out
