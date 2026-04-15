"""
Main SFT training script.

Reproduces the full-parameter supervised fine-tuning setup from
Ye et al. (2025) "Analyzing the Effects of Supervised Fine-Tuning on Model
Knowledge from Token and Parameter Levels" (EMNLP 2025).

Training configuration (paper §3.3):
  - Full-parameter fine-tuning (no LoRA)
  - AdamW with cosine LR schedule, lr = 1e-5
  - Batch size 8, 1 epoch

Memory requirements for LLaMA-3-8B on a single 32 GB GPU:
  The script defaults to paged_adamw_8bit + gradient checkpointing which
  reduces peak VRAM to approximately 30-34 GB.  If VRAM is exceeded, switch
  to DeepSpeed ZeRO-2 with CPU offload by setting `deepspeed_config` in
  configs/base.yaml (requires ~48 GB system RAM for optimizer states).

Usage:
    python src/train.py \
        --split        data/splits/Dtrain_2_1920_seed42.jsonl \
        --output_dir   checkpoints/Dtrain_2_1920_seed42 \
        --config       configs/base.yaml

Optional overrides:
    --model            meta-llama/Meta-Llama-3-8B
    --learning_rate    1e-5
    --eval_split       data/splits/test_indomain.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model, TaskType

from src.dataset import CBQADataset
from src.kl_divergence import compute_layer_deltas, compute_eval_kl
from src.evaluate import evaluate_cbqa

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Custom Trainer: hooks for per-step analysis logging and AFT methods
# ──────────────────────────────────────────────────────────────────────────────

class SFTTrainer(Trainer):
    """
    Trainer subclass that adds:
      - Per-step per-layer weight delta logging.
      - [Phase 2] AFT-D: data-aware sample loss reweighting based on per-sample
        KL divergence spiking (enabled by training_args.aft_d_enabled).
      - [Phase 3] AFT-P: selective periodic micro-resets of drifted parameters
        to pretrained values (enabled by training_args.aft_p_enabled).
    """

    def __init__(self, *args, pretrained_state_dict: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Store pretrained weights on CPU for delta computation and AFT-P resets
        self.pretrained_state_dict = pretrained_state_dict

    # ------------------------------------------------------------------
    # Standard loss; AFT-D reweighting plugged in here when enabled
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Pop non-tensor fields that the model doesn't accept
        inputs.pop("question", None)
        inputs.pop("answers",  None)

        outputs = model(**inputs)
        loss    = outputs.loss

        # ── Phase 2 hook: data-aware reweighting (AFT-D) ──────────────
        if getattr(self.args, "aft_d_enabled", False):
            loss = self._aft_d_reweight(model, inputs, outputs, loss)

        return (loss, outputs) if return_outputs else loss

    def _aft_d_reweight(self, model, inputs, outputs, loss):
        """
        AFT-D: down-weight samples whose per-token KL divergence spikes above
        a threshold relative to the pretrained model distribution.

        NOTE: This requires the pretrained model reference. For Phase 2
        implementation, hook in per-sample KL computation here.
        Currently a stub — returns unmodified loss.
        """
        # TODO (Phase 2): compute per-sample sKL, build weight vector, return
        # weighted mean loss.
        return loss

    # ------------------------------------------------------------------
    # AFT-P: micro-resets applied periodically via optimizer step override
    # ------------------------------------------------------------------

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)

        # ── Phase 3 hook: selective micro-resets (AFT-P) ─────────────
        if (
            getattr(self.args, "aft_p_enabled", False)
            and self.pretrained_state_dict is not None
            and self.state.global_step > 0
            and self.state.global_step % self.args.aft_p_reset_every_n_steps == 0
        ):
            self._aft_p_micro_reset(model)

        return loss

    def _aft_p_micro_reset(self, model):
        """
        AFT-P: restore the top-k% most-drifted parameters in 'suspect' layers
        to their pretrained values.

        NOTE: full implementation in Phase 3.  Currently a stub.
        """
        # TODO (Phase 3): compute per-layer delta norms, identify suspect layers,
        # restore top-k% param subset to pretrained values.
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Per-step analysis callback
# ──────────────────────────────────────────────────────────────────────────────

class AnalysisCallback(TrainerCallback):
    """Log per-layer weight delta norms at each logging step."""

    def __init__(self, pretrained_state_dict: dict | None, log_dir: str):
        self.pretrained_state_dict = pretrained_state_dict
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_dir / "layer_deltas.jsonl", "w")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        logs: dict | None = None,
        **kwargs,
    ):
        if self.pretrained_state_dict is None or model is None:
            return
        # Under DeepSpeed ZeRO-3, parameters are sharded and have size 0 on
        # the local rank — gathering them for delta computation would require
        # a collective op. Skip logging in that case.
        first_param = next(iter(model.parameters()), None)
        if first_param is not None and 0 in first_param.shape:
            return
        deltas = compute_layer_deltas(model, self.pretrained_state_dict)
        record = {"step": state.global_step, "layer_deltas": deltas}
        self._log_file.write(json.dumps(record) + "\n")
        self._log_file.flush()

    def on_train_end(self, *args, **kwargs):
        self._log_file.close()


# ──────────────────────────────────────────────────────────────────────────────
# Training entry point
# ──────────────────────────────────────────────────────────────────────────────

def train(
    split_jsonl: str,
    output_dir:  str,
    cfg:         dict,
    eval_jsonl:  str | None = None,
    resume_from_checkpoint: str | None = None,
):
    """Run one SFT training experiment."""
    model_cfg    = cfg["model"]
    train_cfg    = cfg["training"]
    model_name   = model_cfg["name"]
    out_path     = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ─────────────────────────────────────────────────────────
    logger.info("Loading model: %s", model_name)
    # Use flash_attention_2 if installed, fall back to eager silently
    if model_cfg.get("use_flash_attention"):
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            logger.warning("flash-attn not installed — falling back to eager attention")
            attn_impl = "eager"
    else:
        attn_impl = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype                 = torch.bfloat16 if model_cfg.get("bf16") else torch.float32,
        attn_implementation   = attn_impl,
        device_map            = "auto",
    )
    if model_cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # ── LoRA ──────────────────────────────────────────────────────────
    lora_cfg = cfg.get("lora", {})
    if lora_cfg.get("enabled", False):
        logger.info("Applying LoRA (r=%d, alpha=%d) …", lora_cfg["r"], lora_cfg["alpha"])
        peft_config = LoraConfig(
            task_type       = TaskType.CAUSAL_LM,
            r               = lora_cfg["r"],
            lora_alpha      = lora_cfg["alpha"],
            lora_dropout    = lora_cfg.get("dropout", 0.05),
            target_modules  = lora_cfg["target_modules"],
            bias            = "none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Snapshot pretrained weights on CPU for delta logging + AFT-P.
    # Under DeepSpeed ZeRO-3 the on_log callback skips delta computation anyway
    # (params are sharded / size-0 locally), so skip the 16 GB snapshot to
    # keep memory headroom during training.
    ds_config = train_cfg.get("deepspeed_config")  # checked again here for clarity
    if ds_config:
        logger.info("DeepSpeed mode: skipping pretrained snapshot (not used under ZeRO-3).")
        pretrained_state_dict = None
    else:
        logger.info("Snapshotting pretrained weights to CPU …")
        pretrained_state_dict = {
            k: v.cpu().clone() for k, v in model.state_dict().items()
        }

    # ── Datasets ──────────────────────────────────────────────────────
    train_dataset = CBQADataset(split_jsonl, tokenizer,
                                max_length=train_cfg["max_seq_length"])
    eval_dataset  = None
    if eval_jsonl:
        eval_dataset = CBQADataset(eval_jsonl, tokenizer,
                                   max_length=train_cfg["max_seq_length"])

    logger.info("Train samples: %d", len(train_dataset))
    if eval_dataset:
        logger.info("Eval samples:  %d", len(eval_dataset))

    # ── Training arguments ────────────────────────────────────────────
    # (ds_config already set above, before the snapshot block)

    # Build kwargs incrementally so we never pass deepspeed=None
    # (passing None still triggers deepspeed import in accelerate on some versions)
    training_kwargs: dict = dict(
        output_dir                  = str(out_path),
        per_device_train_batch_size = train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps = train_cfg["gradient_accumulation_steps"],
        num_train_epochs            = train_cfg["num_train_epochs"],
        learning_rate               = train_cfg["learning_rate"],
        lr_scheduler_type           = train_cfg["lr_scheduler_type"],
        warmup_steps                = int(train_cfg.get("warmup_steps", 0) or
                                          train_cfg.get("warmup_ratio", 0.03) *
                                          (len(train_dataset) // (train_cfg["per_device_train_batch_size"] * train_cfg["gradient_accumulation_steps"]) * train_cfg["num_train_epochs"])),
        bf16                        = model_cfg.get("bf16", True),
        gradient_checkpointing      = model_cfg.get("gradient_checkpointing", True),
        optim                       = train_cfg["optimizer"],
        logging_steps               = train_cfg["logging_steps"],
        # Under DeepSpeed ZeRO-3, writing the ~90 GB optimizer-state checkpoint
        # mid-training causes torch.save to create contiguous copies →
        # SIGKILL from OOM.  Disable ALL automatic saves; we extract the model
        # weights directly from the GPU BF16 shards after training ends.
        save_strategy               = "no" if ds_config else "steps",
        save_steps                  = train_cfg["save_steps"],
        eval_steps                  = train_cfg.get("eval_steps"),
        # With DeepSpeed, skip the Trainer's NLL eval (memory pressure, and we
        # run our own CBQA accuracy eval post-training via evaluate_cbqa anyway).
        eval_strategy               = "no" if ds_config else ("steps" if eval_dataset else "no"),
        save_total_limit            = 2,
        load_best_model_at_end      = False,
        remove_unused_columns       = False,
        report_to                   = "none",
        run_name                    = Path(split_jsonl).stem,
    )
    if ds_config:
        training_kwargs["deepspeed"] = ds_config

    training_args = TrainingArguments(**training_kwargs)

    # Attach AFT flags as dynamic attributes (TrainingArguments accepts extras)
    training_args.aft_d_enabled           = train_cfg.get("aft_d_enabled", False)
    training_args.aft_p_enabled           = train_cfg.get("aft_p_enabled", False)
    training_args.aft_p_reset_every_n_steps = (
        train_cfg.get("aft_p", {}).get("reset_every_n_steps", 100)
    )

    # ── Trainer ───────────────────────────────────────────────────────
    callbacks = [
        AnalysisCallback(
            pretrained_state_dict = pretrained_state_dict,
            log_dir               = str(out_path / "logs"),
        )
    ]

    trainer = SFTTrainer(
        model                  = model,
        args                   = training_args,
        train_dataset          = train_dataset,
        eval_dataset           = eval_dataset,
        data_collator          = CBQADataset.collate_fn,
        pretrained_state_dict  = pretrained_state_dict,
        callbacks              = callbacks,
    )

    # ── Train ─────────────────────────────────────────────────────────
    logger.info("Starting training …")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ── Save ──────────────────────────────────────────────────────────
    # For LoRA models, save_model saves only the adapter weights.
    # For full-param models, it saves the full checkpoint.
    #
    # Under DeepSpeed ZeRO-3 we never write the full ZeRO checkpoint (would
    # torch.save ~90 GB of optimizer state, creating contiguous copies →
    # SIGKILL).  Instead, we gather BF16 working parameters one at a time
    # from the GPU ZeRO-3 shards using GatheredParameters.  This reads at
    # most ~500 MB at a time from GPU, builds only a 16 GB state dict on CPU,
    # and never touches the optimizer state tensors.
    final_model_dir = str(out_path / "final_model")
    if ds_config:
        import deepspeed
        import gc
        logger.info("Gathering BF16 params from ZeRO-3 GPU shards …")
        state_dict: dict = {}
        with torch.no_grad():
            for name, param in trainer.model.named_parameters():
                with deepspeed.zero.GatheredParameters([param], modifier_rank=None):
                    state_dict[name] = param.data.cpu().to(torch.bfloat16).clone()
        logger.info("Gathered %d tensors. Saving HF model …", len(state_dict))
        os.makedirs(final_model_dir, exist_ok=True)
        model.save_pretrained(final_model_dir, state_dict=state_dict)
        tokenizer.save_pretrained(final_model_dir)
        del state_dict
        gc.collect()
        # Sentinel: signals the shell script that training + save completed.
        (out_path / "training_complete").write_text("ok\n")
    else:
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

        # ── Post-training CBQA eval (greedy, non-DeepSpeed only) ──────────────
        if eval_jsonl:
            logger.info("Running post-training CBQA evaluation …")
            results = evaluate_cbqa(
                model          = model,
                tokenizer      = tokenizer,
                jsonl_path     = eval_jsonl,
                batch_size     = cfg["evaluation"]["batch_size"],
                max_new_tokens = cfg["evaluation"]["max_new_tokens"],
            )
            results_path = out_path / "eval_results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info("Eval accuracy: %.4f  → %s", results["accuracy"], results_path)

    logger.info("Done → %s", out_path)
    return final_model_dir


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="SFT training script for CBQA reproduction.")
    parser.add_argument("--split",       required=True,
                        help="Path to train split JSONL (e.g. data/splits/Dtrain_2_1920_seed42.jsonl)")
    parser.add_argument("--output_dir",  required=True,
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--config",      default="configs/base.yaml")
    parser.add_argument("--model",       default=None,
                        help="Override model name from config")
    parser.add_argument("--eval_split",  default=None,
                        help="Optional eval JSONL (e.g. data/splits/test_indomain.jsonl)")
    parser.add_argument("--resume_from_checkpoint", default=None,
                        help="Path to a checkpoint directory to resume from")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Override learning rate")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model:
        cfg["model"]["name"] = args.model
    if args.learning_rate:
        cfg["training"]["learning_rate"] = args.learning_rate

    train(
        split_jsonl             = args.split,
        output_dir              = args.output_dir,
        cfg                     = cfg,
        eval_jsonl              = args.eval_split,
        resume_from_checkpoint  = args.resume_from_checkpoint,
    )
