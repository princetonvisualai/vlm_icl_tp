"""Control Accuracy Visualization — accuracy scaling plots with clean vs corrupt conditions."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from figures._plotting.models import QWEN3_MODELS
from figures._plotting.style import setup_grid_style

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_FAMILY_MAP = {
    "qwen3": QWEN3_MODELS,
    "all": QWEN3_MODELS,
}

CONDITION_STYLES = {
    "clean": {"color": "#2196F3", "linestyle": "-", "label": "Clean"},
    "corrupt": {"color": "#F44336", "linestyle": "--", "label": "Corrupt"},
}

# ── Data Loading ─────────────────────────────────────────────────────────────


def discover_summaries(base_path: str, model_family: str) -> List[Dict]:
    """Discover accuracy_summary.json files and return structured records."""
    base = Path(base_path)
    family_models = MODEL_FAMILY_MAP.get(model_family, MODEL_FAMILY_MAP["all"])
    records = []

    for summary_path in base.rglob("accuracy_summary.json"):
        parts = summary_path.relative_to(base).parts
        # Expected: task/model/experiment/condition/kN/accuracy_summary.json
        if len(parts) < 5:
            continue

        task = parts[0]
        model = parts[1]
        condition = parts[-3]
        k_dir = parts[-2]

        if model not in family_models:
            continue
        if condition not in ("clean", "corrupt"):
            continue

        k_match = re.match(r"k(\d+)", k_dir)
        if not k_match:
            continue
        k = int(k_match.group(1))

        try:
            with open(summary_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

        acc = data.get("accuracy", {})
        mean = acc.get("mean")
        std = acc.get("std", 0.0)
        if mean is None:
            continue

        records.append({
            "task": task,
            "model": model,
            "condition": condition,
            "k": k,
            "accuracy_mean": mean * 100,
            "accuracy_std": std * 100,
        })

    return records


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_control_accuracy(ax, records: List[Dict], title: str):
    """Plot accuracy vs k for clean and corrupt conditions."""
    by_condition = defaultdict(dict)
    for r in records:
        by_condition[r["condition"]][r["k"]] = (r["accuracy_mean"], r["accuracy_std"])

    for condition in ("clean", "corrupt"):
        if condition not in by_condition:
            continue
        style = CONDITION_STYLES[condition]
        data = by_condition[condition]
        ks = sorted(data)
        means = [data[k][0] for k in ks]
        stds = [data[k][1] for k in ks]

        ax.plot(
            ks, means, marker="o", linewidth=2, markersize=8,
            color=style["color"], linestyle=style["linestyle"],
            label=style["label"],
        )
        ax.fill_between(
            ks,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15, color=style["color"],
        )

    ax.set_xlabel("Number of ICL Demos (k)", fontweight="bold", fontsize=16)
    ax.set_ylabel("Accuracy (%)", fontweight="bold", fontsize=16)
    ax.set_title(title, fontweight="bold", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_ylim(0, 105)
    all_ks = sorted(set(r["k"] for r in records))
    ax.set_xticks(all_ks)
    ax.legend(loc="best", fontsize=12)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Control Accuracy Visualization")
    parser.add_argument(
        "--base_path", type=str,
        default=None,
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--model_family", type=str, choices=["qwen3", "all"], default="all",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--results_dir", type=str, default="output",
                        help="Root of the experiment results")
    args = parser.parse_args()
    if args.base_path is None:
        args.base_path = str(Path(args.results_dir) / "control_accuracy/visual_reasoning")

    setup_grid_style(font_size=16)

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        f"output_vis/control_accuracy/{args.model_family}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_summaries(args.base_path, args.model_family)
    if not records:
        print("No accuracy_summary.json files found.")
        return

    tasks = sorted(set(r["task"] for r in records))
    models = sorted(set(r["model"] for r in records))
    print(f"Found {len(records)} records: tasks={tasks}, models={models}")

    saved = []

    for task in tasks:
        for model in models:
            recs = [r for r in records if r["task"] == task and r["model"] == model]
            if not recs:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            task_title = task.replace("_", " ").title()
            plot_control_accuracy(ax, recs, f"Control Accuracy: {task_title} ({model})")
            plt.tight_layout()

            path = output_dir / f"{task}_{model}.png"
            fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
            print(f"Saved: {path}")

    print(f"\nAll {len(saved)} figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
