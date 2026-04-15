# Adaptive Fine-Tuning Strategies for Closed-Book QA

Experiments comparing LoRA and full-parameter SFT on llama-3-8B across training scales and difficulty strata, using the [EntityQuestions](https://github.com/princeton-nlp/EntityQuestions) benchmark.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing experiments

```bash
bash scripts/01_download_data.sh      # download EntityQuestions
bash scripts/02_score_mastery.sh      # compute mastery scores
bash scripts/03_build_splits.sh       # build stratified splits
bash scripts/04_run_sft.sh            # LoRA SFT sweep
bash scripts/04b_run_fullparam_short.sh  # full-param SFT sweep
bash scripts/00_eval_baseline.sh      # pretrained zero-shot baseline
bash scripts/06_visualize.sh          # generate figures → figures/
```

## Project structure

```
configs/        DeepSpeed & training configs
data/           Dataset scripts (raw data excluded; see scripts/01-03)
scripts/        End-to-end pipeline scripts
src/            Model training, evaluation, and visualization code
```

## Requirements

- CUDA GPU with ≥ 24 GB VRAM (32 GB recommended for full-param with ZeRO-3)
- HuggingFace access to `meta-llama/Meta-Llama-3-8B`
