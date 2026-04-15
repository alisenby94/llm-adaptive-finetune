"""
Presentation visualization for CBQA SFT reproduction.

Scans the checkpoints/ and results/ directories for completed evaluation
results and generates the four key figures needed for the 8-minute presentation:

  Figure 1 — Accuracy vs. data scale (in-domain + OOD panels)
              One line per training mastery category, with ±1 std bands.
              Reproduces the core shape of Ye et al. Fig 2.

  Figure 2 — Mastery category effect at scale 1920
              Grouped bar chart: training category × in-domain / OOD accuracy.
              Shows Phenomenon 2 (mastery level determines degradation severity).

  Figure 3 — Per-test-category heatmap at scale 1920
              Rows = training category (0–4), Columns = test category (0–4).
              Visual equivalent of Table 2 in the paper.

  Figure 4 — Restoration sweep (k% vs. accuracy)
              One line per fine-tuned checkpoint, with pretrained baseline.
              Reproduces the post-hoc restoration result.

Usage:
    python src/visualize.py --ckpt_dir checkpoints --results_dir results \
                            --output_dir figures
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Colour palette (one per mastery category) ─────────────────────────────────
CATEGORY_COLORS = {
    0: "#e74c3c",   # red    — least mastered (Dtrain-0)
    1: "#e67e22",   # orange
    2: "#2ecc71",   # green  — mid-mastery (Dtrain-2, best overall)
    3: "#3498db",   # blue
    4: "#9b59b6",   # purple — most mastered (Dtrain-4)
}

CATEGORY_LABELS = {
    0: "Cat 0 (0–25% mastery)",
    1: "Cat 1 (25–50% mastery)",
    2: "Cat 2 (50–75% mastery)",
    3: "Cat 3 (75–100% mastery)",
    4: "Cat 4 (unscored)",
}


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_ckpt_name(name: str) -> tuple[int, int, int] | None:
    """Parse 'Dtrain_2_1920_seed42' → (category=2, scale=1920, seed=42)."""
    m = re.fullmatch(r"Dtrain_(\d)_(\d+)_seed(\d+)", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def load_ckpt_results(ckpt_dir: str) -> dict:
    """
    Walk checkpoints/ and collect eval_results.json (in-domain) and
    eval_ood_results.json (OOD) for every completed run.

    Returns
    -------
    records : list of dicts, each with keys:
        category, scale, seed, id_accuracy, ood_accuracy,
        id_per_cat (dict str→float), ood_per_cat (dict str→float)
    """
    records = []
    root = Path(ckpt_dir)
    for ckpt_path in sorted(root.iterdir()):
        parsed = _parse_ckpt_name(ckpt_path.name)
        if parsed is None:
            continue
        cat, scale, seed = parsed

        id_file  = ckpt_path / "eval_results.json"
        ood_file = ckpt_path / "eval_ood_results.json"

        if not id_file.exists():
            continue  # training not complete yet

        with open(id_file) as f:
            id_data = json.load(f)

        ood_data = None
        if ood_file.exists():
            with open(ood_file) as f:
                ood_data = json.load(f)

        records.append({
            "category":    cat,
            "scale":       scale,
            "seed":        seed,
            "id_accuracy": id_data.get("accuracy", 0.0),
            "ood_accuracy": ood_data.get("accuracy", 0.0) if ood_data else None,
            "id_per_cat":  id_data.get("per_category", {}),
            "ood_per_cat": ood_data.get("per_category", {}) if ood_data else {},
        })

    return records


def load_fullparam_results(ckpt_dir: str) -> list[dict]:
    """
    Walk checkpoints/ collecting fullparam_Dtrain_* eval results.
    Loads both eval_results.json (in-domain) and eval_ood_results.json (OOD)
    when available.
    """
    records = []
    root = Path(ckpt_dir)
    for ckpt_path in sorted(root.iterdir()):
        m = re.fullmatch(r"fullparam_Dtrain_(\d)_(\d+)_seed(\d+)", ckpt_path.name)
        if not m:
            continue
        cat, scale, seed = int(m.group(1)), int(m.group(2)), int(m.group(3))
        ood_file = ckpt_path / "eval_ood_results.json"
        id_file  = ckpt_path / "eval_results.json"
        if not ood_file.exists() and not id_file.exists():
            continue
        id_data  = json.load(open(id_file))  if id_file.exists()  else None
        ood_data = json.load(open(ood_file)) if ood_file.exists() else None
        records.append({
            "category":    cat,
            "scale":       scale,
            "seed":        seed,
            "id_accuracy":  id_data.get("accuracy", 0.0)  if id_data  else None,
            "ood_accuracy": ood_data.get("accuracy", 0.0) if ood_data else None,
            "id_per_cat":   id_data.get("per_category", {})  if id_data  else {},
            "ood_per_cat":  ood_data.get("per_category", {}) if ood_data else {},
        })
    return records


def load_baseline(ckpt_dir: str) -> dict | None:
    """
    Load pretrained (zero-shot) baseline from checkpoints/pretrained_baseline/.
    Returns {id_accuracy, ood_accuracy} or None if not yet evaluated.
    """
    p = Path(ckpt_dir) / "pretrained_baseline"
    result = {}
    for key, fname in [("id_accuracy", "eval_results.json"),
                       ("ood_accuracy", "eval_ood_results.json")]:
        f = p / fname
        if f.exists():
            with open(f) as fh:
                result[key] = json.load(fh).get("accuracy")
        else:
            result[key] = None
    if result["id_accuracy"] is None and result["ood_accuracy"] is None:
        return None
    return result


def load_restoration_results(results_dir: str) -> list[dict]:
    """
    Walk results/restoration/ and collect restoration_sweep.json files.

    Returns a list of dicts, each with keys:
        ckpt_name, split (indomain/ood),
        sweep: list of {k_percent, accuracy}
    """
    out = []
    root = Path(results_dir) / "restoration"
    if not root.exists():
        return out

    for ckpt_path in sorted(root.iterdir()):
        for split in ("indomain", "ood"):
            sweep_file = ckpt_path / split / "restoration_sweep.json"
            if not sweep_file.exists():
                continue
            with open(sweep_file) as f:
                sweep = json.load(f)
            out.append({
                "ckpt_name": ckpt_path.name,
                "split": split,
                "sweep": sweep,
            })
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _aggregate(records: list[dict], key: str) -> dict[tuple, tuple[float, float]]:
    """
    Given a list of records, group by (category, scale) and return
    mean ± std for the field `key`.

    Returns {(cat, scale): (mean, std)}
    """
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        v = r.get(key)
        if v is not None:
            grouped[(r["category"], r["scale"])].append(v)
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in grouped.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 helpers
# ──────────────────────────────────────────────────────────────────────────────

def _scale_line_panel(ax, records: list[dict], split_key: str,
                      marker: str = "o", linestyle: str = "-",
                      alpha: float = 1.0, label_prefix: str = "",
                      baseline: dict | None = None,
                      tick_scales: list | None = None) -> list:
    """Draw one accuracy-vs-scale panel onto ax. Returns (handles, labels)."""
    all_scales = tick_scales if tick_scales is not None else sorted({r["scale"] for r in records})
    # Map each scale value to an evenly-spaced integer position so the
    # geometric doubling (60→120→240…) doesn't require a log axis.
    scale_to_x = {s: i for i, s in enumerate(all_scales)}
    agg = _aggregate(records, split_key)
    for cat in sorted(CATEGORY_COLORS):
        means, stds, xs = [], [], []
        for scale in all_scales:
            if (cat, scale) in agg:
                m, s = agg[(cat, scale)]
                means.append(m * 100)
                stds.append(s * 100)
                xs.append(scale_to_x[scale])
        if not xs:
            continue
        color = CATEGORY_COLORS[cat]
        label = f"{label_prefix}{CATEGORY_LABELS[cat]}" if label_prefix else CATEGORY_LABELS[cat]
        ax.plot(xs, means, marker=marker, linestyle=linestyle, color=color,
                label=label, linewidth=2, markersize=6, alpha=alpha)
        if linestyle == "-":
            ax.fill_between(xs,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=color, alpha=0.12)
    # Pretrained zero-shot baseline horizontal line
    if baseline is not None:
        bval = baseline.get(split_key)
        if bval is not None:
            ax.axhline(bval * 100, color="black", linestyle=":", linewidth=2,
                       label="Pretrained (zero-shot)")
    ax.set_xticks(range(len(all_scales)))
    ax.set_xticklabels([str(s) for s in all_scales])
    ax.set_xlabel("Training samples", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.grid(True, alpha=0.3)
    return ax.get_legend_handles_labels()


def _finalize_fig(fig, axes, title: str, out_path: Path,
                  extra_handles: list | None = None) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate category handles (comparison fig draws each category twice)
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)

    if extra_handles:
        # Two-row legend: categories on top, style key below
        leg1 = fig.legend(h2, l2, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                          ncol=min(len(h2), 8), fontsize=8, frameon=True)
        fig.add_artist(leg1)
        extra_labels = [h.get_label() for h in extra_handles]
        fig.legend(extra_handles, extra_labels,
                   loc="upper center", bbox_to_anchor=(0.5, 0.93),
                   ncol=len(extra_handles), fontsize=8, frameon=True)
    else:
        fig.legend(h2, l2, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                   ncol=min(len(h2), 8), fontsize=8, frameon=True, borderaxespad=0.2)

    fig.suptitle(title, fontsize=12, y=1.14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1a — LoRA: Accuracy vs. data scale
# ──────────────────────────────────────────────────────────────────────────────

def fig1a_lora_accuracy_vs_scale(records: list[dict], output_dir: Path,
                                  baseline: dict | None = None) -> None:
    """LoRA only — In-Domain + OOD panels, solid lines with shaded std bands."""
    if not records:
        print("  No LoRA records — skipping Fig 1a")
        return

    has_ood = any(r["ood_accuracy"] is not None for r in records)
    panels = [("id_accuracy", "In-Domain Accuracy")]
    if has_ood:
        panels.append(("ood_accuracy", "Out-of-Domain Accuracy"))

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5), sharey=False)
    if len(panels) == 1:
        axes = [axes]

    for ax, (key, title) in zip(axes, panels):
        _scale_line_panel(ax, records, key, baseline=baseline)
        ax.set_title(title, fontsize=13, fontweight="bold")

    _finalize_fig(fig, axes,
                  "LoRA — CBQA Accuracy vs. Fine-tuning Data Scale\n(LLaMA-3-8B, Ye et al. 2025)",
                  output_dir / "fig1a_lora_accuracy_vs_scale.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1b — Full-param: OOD Accuracy vs. data scale
# ──────────────────────────────────────────────────────────────────────────────

def fig1b_fp_accuracy_vs_scale(fullparam_records: list[dict], output_dir: Path,
                                baseline: dict | None = None) -> None:
    """Full-param only — In-Domain + OOD panels (mirrors fig1a structure)."""
    if not fullparam_records:
        print("  No full-param records — skipping Fig 1b")
        return

    has_id  = any(r["id_accuracy"]  is not None for r in fullparam_records)
    has_ood = any(r["ood_accuracy"] is not None for r in fullparam_records)

    panels = []
    if has_id:  panels.append(("id_accuracy",  "In-Domain Accuracy"))
    if has_ood: panels.append(("ood_accuracy", "Out-of-Domain Accuracy"))

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5), sharey=False)
    if len(panels) == 1:
        axes = [axes]

    for ax, (key, title) in zip(axes, panels):
        _scale_line_panel(ax, fullparam_records, key, baseline=baseline)
        ax.set_title(title, fontsize=13, fontweight="bold")

    _finalize_fig(fig, axes,
                  "Full-Param SFT — CBQA Accuracy vs. Fine-tuning Data Scale\n(LLaMA-3-8B, Ye et al. 2025)",
                  output_dir / "fig1b_fp_accuracy_vs_scale.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1c — LoRA vs. Full-param comparison (OOD)
# ──────────────────────────────────────────────────────────────────────────────

def fig1c_comparison_accuracy_vs_scale(records: list[dict],
                                       fullparam_records: list[dict],
                                       output_dir: Path,
                                       baseline: dict | None = None) -> None:
    """LoRA (solid) vs Full-param (dashed) on the OOD panel, same colour = same category."""
    from matplotlib.lines import Line2D

    has_lora = any(r["ood_accuracy"] is not None for r in records)
    has_fp   = bool(fullparam_records)

    if not has_lora and not has_fp:
        print("  No OOD data for comparison — skipping Fig 1c")
        return

    # Union of scales so both series share the same x-axis ticks
    all_scales = sorted(
        {r["scale"] for r in records} | {r["scale"] for r in fullparam_records}
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    if has_lora:
        _scale_line_panel(ax, records, "ood_accuracy",
                          marker="o", linestyle="-", alpha=1.0, baseline=baseline,
                          tick_scales=all_scales)
    if has_fp:
        _scale_line_panel(ax, fullparam_records, "ood_accuracy",
                          marker="s", linestyle="--", alpha=0.85,
                          tick_scales=all_scales)

    # Style entries — only include baseline entry when data actually exists
    style_handles = [
        Line2D([0], [0], color="grey", linewidth=2, linestyle="-",  label="LoRA"),
        Line2D([0], [0], color="grey", linewidth=2, linestyle="--", label="Full-param SFT"),
    ]
    if baseline is not None and baseline.get("ood_accuracy") is not None:
        style_handles.append(
            Line2D([0], [0], color="black", linewidth=2, linestyle=":", label="Pretrained (zero-shot)")
        )

    ax.set_title("OOD Accuracy: LoRA vs. Full-Param SFT", fontsize=13, fontweight="bold")

    _finalize_fig(fig, [ax],
                  "LLaMA-3-8B — LoRA (solid) vs Full-Param SFT (dashed)\nOOD Accuracy vs. Fine-tuning Data Scale",
                  output_dir / "fig1c_lora_vs_fp_ood.png",
                  extra_handles=style_handles)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: Mastery category effect (bar chart at scale=1920)
# ──────────────────────────────────────────────────────────────────────────────

def fig_mastery_category_effect(records: list[dict], output_dir: Path,
                                  scale: int = 1920) -> None:
    """Grouped bar chart: training category vs. in-domain / OOD accuracy at given scale."""
    subset = [r for r in records if r["scale"] == scale]
    if not subset:
        print(f"  No records at scale={scale} — skipping Fig 2")
        return

    cats = sorted(CATEGORY_COLORS)
    id_means, id_stds = [], []
    ood_means, ood_stds = [], []

    for cat in cats:
        cat_records = [r for r in subset if r["category"] == cat]
        id_vals  = [r["id_accuracy"] for r in cat_records]
        ood_vals = [r["ood_accuracy"] for r in cat_records if r["ood_accuracy"] is not None]

        id_means.append(np.mean(id_vals) * 100 if id_vals else 0)
        id_stds.append(np.std(id_vals) * 100 if len(id_vals) > 1 else 0)
        ood_means.append(np.mean(ood_vals) * 100 if ood_vals else 0)
        ood_stds.append(np.std(ood_vals) * 100 if len(ood_vals) > 1 else 0)

    x = np.arange(len(cats))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, id_means,  width, yerr=id_stds,
                   label="In-Domain", color="#3498db", alpha=0.85, capsize=4)
    bars2 = ax.bar(x + width / 2, ood_means, width, yerr=ood_stds,
                   label="OOD",       color="#e74c3c", alpha=0.85, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in cats], rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Effect of Training Data Mastery Category (scale = {scale})\n"
                 f"LLaMA-3-8B + LoRA — Phenomenon 2: Low-mid mastery (Cat 1) trains best",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    # Annotate bar tops
    for bar in (*bars1, *bars2):
        h = bar.get_height()
        if h > 1:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    # Horizontal legend strip under the title
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(handles), fontsize=11, frameon=True,
               borderaxespad=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = output_dir / f"fig2_mastery_category_effect_scale{scale}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: Per-test-category heatmap (Table 2 equivalent)
# ──────────────────────────────────────────────────────────────────────────────

def fig_per_category_heatmap(records: list[dict], output_dir: Path,
                               scale: int = 1920, split: str = "id") -> None:
    """
    Heatmap: rows = training category (0–4), columns = test relation (P17, P131…).
    Reproduces Table 2 from Ye et al.  Per-category keys are Wikidata relation IDs.
    """
    key = "id_per_cat" if split == "id" else "ood_per_cat"
    subset = [r for r in records if r["scale"] == scale and r.get(key)]
    if not subset:
        print(f"  No per-category data at scale={scale} — skipping Fig 3")
        return

    # Discover all relation keys that actually appear in the data (exclude "None")
    all_relations: set[str] = set()
    for r in subset:
        all_relations.update(k for k in r[key].keys() if k != "None")
    if not all_relations:
        print(f"  No per-relation breakdown in {key} — skipping Fig 3 ({split})")
        return
    test_relations = sorted(all_relations)

    train_cats = sorted(CATEGORY_COLORS)

    # Average per (train_cat, relation) over seeds
    matrix = np.zeros((len(train_cats), len(test_relations)))
    for i, train_cat in enumerate(train_cats):
        cat_recs = [r for r in subset if r["category"] == train_cat]
        for j, rel in enumerate(test_relations):
            vals = [r[key].get(rel, 0.0) for r in cat_recs if rel in r[key]]
            matrix[i, j] = np.mean(vals) * 100 if vals else 0.0

    fig, ax = plt.subplots(figsize=(max(10, len(test_relations) * 1.1), 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="Accuracy (%)")

    ax.set_xticks(range(len(test_relations)))
    ax.set_xticklabels(test_relations, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(train_cats)))
    ax.set_yticklabels([f"Train {CATEGORY_LABELS[c]}" for c in train_cats], fontsize=10)
    ax.set_xlabel("Test Relation (Wikidata property)", fontsize=11)
    ax.set_ylabel("Training Data Category", fontsize=11)
    split_label = "In-Domain" if split == "id" else "OOD"
    ax.set_title(f"{split_label} Per-Relation Accuracy at Scale {scale}\n"
                 f"(Table 2 equivalent — LLaMA-3-8B + LoRA)",
                 fontsize=12, fontweight="bold")

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                    fontsize=8, color="black" if 20 < matrix[i, j] < 80 else "white")

    fig.tight_layout()
    out = output_dir / f"fig3_per_category_heatmap_{split}_scale{scale}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4: Restoration sweep
# ──────────────────────────────────────────────────────────────────────────────

def fig_restoration_sweep(restoration_records: list[dict], output_dir: Path) -> None:
    """
    Line chart: restoration k% vs. accuracy for each checkpoint.
    Shows that restoring the top-k% most-changed params improves performance.
    """
    id_records = [r for r in restoration_records if r["split"] == "indomain"]
    if not id_records:
        print("  No restoration results found — skipping Fig 4")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(id_records)))  # type: ignore[attr-defined]

    for idx, rec in enumerate(id_records):
        name  = rec["ckpt_name"]
        sweep = sorted(rec["sweep"], key=lambda x: x["k_percent"])
        ks    = [s["k_percent"] for s in sweep]
        accs  = [s["accuracy"] * 100 for s in sweep]

        # k=0 is the baseline fine-tuned model
        baseline_k0 = accs[0] if ks[0] == 0 else None

        ax.plot(ks, accs, marker="o", label=name, color=colors[idx],
                linewidth=2, markersize=5)

        if baseline_k0 is not None:
            ax.axhline(baseline_k0, color=colors[idx], linestyle=":",
                       linewidth=1, alpha=0.5)

    ax.set_xlabel("Parameters restored to pretrained values (k%)", fontsize=12)
    ax.set_ylabel("In-Domain Accuracy (%)", fontsize=12)
    ax.set_title("Post-hoc Parameter Restoration Sweep\n"
                 "(Ye et al. §5: up to 90% of updates are unnecessary)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    fig.tight_layout()
    out = output_dir / "fig4_restoration_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate presentation figures.")
    parser.add_argument("--ckpt_dir",     default="checkpoints",
                        help="Checkpoints directory (output of 04_run_sft.sh)")
    parser.add_argument("--results_dir",  default="results",
                        help="Results directory (output of 05_run_restoration.sh)")
    parser.add_argument("--output_dir",   default="figures",
                        help="Where to save generated PNG figures")
    parser.add_argument("--heatmap_scale", type=int, default=1920,
                        help="Data scale to use for Fig 3 heatmap (default 1920)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading LoRA checkpoint eval results …")
    records = load_ckpt_results(args.ckpt_dir)
    print(f"  Found {len(records)} completed LoRA runs")

    print("Loading full-param eval results …")
    fullparam_records = load_fullparam_results(args.ckpt_dir)
    print(f"  Found {len(fullparam_records)} completed full-param runs")

    print("Loading pretrained baseline …")
    baseline = load_baseline(args.ckpt_dir)
    if baseline:
        print(f"  Baseline: ID={baseline.get('id_accuracy')}, OOD={baseline.get('ood_accuracy')}")
    else:
        print("  No baseline found — run: python src/evaluate.py --model_path meta-llama/Meta-Llama-3-8B ...")

    print("Loading restoration results …")
    restoration_records = load_restoration_results(args.results_dir)
    print(f"  Found {len(restoration_records)} restoration sweeps")

    if not records and not fullparam_records:
        print("\nNo completed training runs found yet.")
        return

    print("\nGenerating figures …")
    fig1a_lora_accuracy_vs_scale(records, out, baseline=baseline)
    fig1b_fp_accuracy_vs_scale(fullparam_records, out, baseline=baseline)
    fig1c_comparison_accuracy_vs_scale(records, fullparam_records, out, baseline=baseline)
    fig_mastery_category_effect(records, out, scale=args.heatmap_scale)
    fig_per_category_heatmap(records, out, scale=args.heatmap_scale, split="id")
    if any(r["ood_accuracy"] is not None for r in records):
        fig_per_category_heatmap(records, out, scale=args.heatmap_scale, split="ood")
    fig_restoration_sweep(restoration_records, out)

    print(f"\nAll figures saved to {out}/")


if __name__ == "__main__":
    main()
