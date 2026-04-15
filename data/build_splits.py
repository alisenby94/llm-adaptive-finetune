"""
Build categorized training / test splits from the scored EntityQuestions data.

Reads the mastery-scored JSONL (output of data/mastery_scorer.py) and produces
the five Dtrain-{0..4} category splits at multiple data scales, reproducing the
experimental setup of Ye et al. (2025).

Split definitions (Dtrain-i):
    i=0 : R_k = 0              (model knows nothing about this fact)
    i=1 : R_k ∈ (0,   0.25]
    i=2 : R_k ∈ (0.25, 0.50]
    i=3 : R_k ∈ (0.50, 0.75]
    i=4 : R_k ∈ (0.75, 1.00]

For each (category, scale, seed) triple we sample with balanced topic distribution
and write a separate JSONL file under `splits_dir`.

Usage:
    python data/build_splits.py \
        --scored_jsonl  data/processed/scored.jsonl \
        --splits_dir    data/splits \
        --config        configs/base.yaml
"""

import json
import random
import argparse
from collections import defaultdict
from pathlib import Path

import yaml

# ──────────────────────────────────────────────────────────────────────────────
# Mastery category boundaries (lower-exclusive, upper-inclusive)
# ──────────────────────────────────────────────────────────────────────────────
CATEGORY_BOUNDS = [
    (0.0,  0.0),   # cat 0: exactly 0
    (0.0,  0.25),  # cat 1: (0, 0.25]
    (0.25, 0.50),  # cat 2: (0.25, 0.50]
    (0.50, 0.75),  # cat 3: (0.50, 0.75]
    (0.75, 1.01),  # cat 4: (0.75, 1.0]  — 1.01 to include 1.0
]


def _assign_category(mastery_score: float) -> int:
    """Map a continuous mastery score to category index 0–4."""
    if not mastery_score:  # handles None, NaN, and 0.0
        return 0
    for i, (lo, hi) in enumerate(CATEGORY_BOUNDS[1:], start=1):
        if lo < mastery_score <= hi:
            return i
    return 4  # clamp to top category for any floating-point edge case


def _stratified_sample(
    records: list[dict],
    n: int,
    topic_key: str = "relation",
    rng: random.Random = None,
) -> list[dict]:
    """
    Sample n records with balanced distribution across topics.
    If a topic has fewer items than its fair share, all its items are included.
    """
    if rng is None:
        rng = random.Random()

    # Group by topic
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_topic[rec.get(topic_key, "unknown")].append(rec)

    topics = list(by_topic.keys())
    if not topics:
        return []

    base_per_topic = n // len(topics)
    remainder      = n % len(topics)

    sampled: list[dict] = []
    for idx, topic in enumerate(topics):
        quota = base_per_topic + (1 if idx < remainder else 0)
        pool  = by_topic[topic]
        take  = min(quota, len(pool))
        sampled.extend(rng.sample(pool, take))

    # If we fell short due to small topics, fill from the full pool
    if len(sampled) < n:
        sampled_ids = set(id(s) for s in sampled)
        remaining_pool = [r for r in records if id(r) not in sampled_ids]
        extra = min(n - len(sampled), len(remaining_pool))
        sampled.extend(rng.sample(remaining_pool, extra))

    rng.shuffle(sampled)
    return sampled


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def build_splits(
    scored_jsonl: str,
    splits_dir: str,
    location_relations: list[str],
    data_scales: list[int],
    random_seeds: list[int],
) -> None:
    """
    Build and save Dtrain-{i} splits at each (scale, seed) combination.

    Also writes:
      - ``test_indomain.jsonl``  : in-domain test set (same 10 location topics)
      - ``test_ood.jsonl``       : OOD test set (non-location topics)
    """
    splits_path = Path(splits_dir)
    splits_path.mkdir(parents=True, exist_ok=True)

    loc_set = set(location_relations)

    # Load all scored records
    all_records: list[dict] = []
    with open(scored_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    # Partition into splits
    train_pool: dict[int, list[dict]] = defaultdict(list)  # category → records
    test_indomain: list[dict] = []
    test_ood:      list[dict] = []

    for rec in all_records:
        split    = rec.get("split", "train")
        relation = rec.get("relation", "")

        if split == "train" and relation in loc_set:
            cat = _assign_category(rec.get("mastery_score", 0.0))
            rec["mastery_category"] = cat
            train_pool[cat].append(rec)

        elif split in ("dev", "test"):
            if relation in loc_set:
                # Tag test records with their relation so evaluate.py can group by relation
                rec["mastery_category"] = relation
                test_indomain.append(rec)
            else:
                rec["mastery_category"] = relation
                test_ood.append(rec)

    # Write test sets
    _write_jsonl(splits_path / "test_indomain.jsonl", test_indomain)
    _write_jsonl(splits_path / "test_ood.jsonl", test_ood)
    print(f"  test_indomain : {len(test_indomain):,} records")
    print(f"  test_ood      : {len(test_ood):,}      records")

    # Print category sizes for inspection
    for cat in range(5):
        print(f"  Dtrain-{cat} pool : {len(train_pool[cat]):,} records")

    # Build (category, scale, seed) splits
    for cat in range(5):
        pool = train_pool[cat]
        for scale in data_scales:
            for seed in random_seeds:
                rng = random.Random(seed)
                sampled = _stratified_sample(pool, scale, topic_key="relation", rng=rng)
                filename = f"Dtrain_{cat}_{scale}_seed{seed}.jsonl"
                _write_jsonl(splits_path / filename, sampled)
                print(f"  {filename}: {len(sampled)} records")

    print(f"\nAll splits written to {splits_path}/")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build categorical train/test splits.")
    parser.add_argument("--scored_jsonl", required=True,
                        help="Path to mastery-scored JSONL (output of mastery_scorer.py)")
    parser.add_argument("--splits_dir", default="data/splits",
                        help="Output directory for split files")
    parser.add_argument("--config", default="configs/base.yaml",
                        help="Experiment config YAML")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    build_splits(
        scored_jsonl       = args.scored_jsonl,
        splits_dir         = args.splits_dir,
        location_relations = cfg["data"]["location_relations"],
        data_scales        = cfg["data"]["data_scales"],
        random_seeds       = cfg["data"]["random_seeds"],
    )
