"""Variance Ablation Visualization — generates accuracy vs ICL demos plots with shaded variance bands."""

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

from figures._plotting.models import API_MODELS, QWEN25_MODELS, QWEN3_MODELS
from figures._plotting.style import setup_grid_style

# ── Constants ────────────────────────────────────────────────────────────────

DISPLAY_ORDER = ["counting", "direction", "relation", "mental_rotation"]
MODEL_ORDER = QWEN3_MODELS + QWEN25_MODELS + API_MODELS

MODEL_FAMILY_MAP = {
    "qwen3": QWEN3_MODELS,
    "qwen2p5": QWEN25_MODELS,
    "api": API_MODELS,
    "all": MODEL_ORDER,
}

RANDOM_GUESSING = {
    "counting_noprior": 1 / 8,
    "counting_simple": 1 / 8,
    "counting_complex": 1 / 6,
    "direction_noprior": 1 / 6,
    "direction_simple": 1 / 6,
    "direction_complex": 1 / 6,
    "relation_noprior": 1 / 6,
    "relation_simple": 1 / 6,
    "relation_complex": 1 / 6,
}

# ── Data Loading ─────────────────────────────────────────────────────────────


def parse_variance_dir(dirname: str) -> Tuple[int, int]:
    """Parse variance experiment directory name into (sps, icl_count)."""
    sps_match = re.match(r"sps(\d+)_", dirname)
    sps = int(sps_match.group(1)) if sps_match else 0

    icl_match = re.search(r"icl(\d+)", dirname)
    icl_count = int(icl_match.group(1)) if icl_match else 0

    return sps, icl_count


def load_variance_summary(path: Path) -> Optional[Dict]:
    """Load and return parsed variance_summary.json, or None on error."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {path}: {e}")
        return None


def collect_all_data(base_path: str) -> List[Dict]:
    """Discover all variance_summary.json files and return a flat list of accuracy records."""
    base = Path(base_path)
    records = []

    for summary_path in base.rglob("variance_summary.json"):
        parts = summary_path.relative_to(base).parts
        # Expected: task/model/experiment/variance_summary.json
        if len(parts) != 4:
            continue
        task, model, experiment, _ = parts

        _, shots = parse_variance_dir(experiment)

        data = load_variance_summary(summary_path)
        if data is None:
            continue

        # by_split entries
        by_split = data.get("by_split", {})
        for split_name, split_metrics in by_split.items():
            acc = split_metrics.get("accuracy", {})
            mean = acc.get("mean")
            std = acc.get("std")
            if mean is not None and std is not None:
                records.append(
                    {
                        "task": task,
                        "model": model,
                        "shots": shots,
                        "metric_type": split_name,
                        "accuracy_mean": mean * 100,
                        "accuracy_std": std * 100,
                    }
                )

        # by_originality entries (scaffolded — not present in current data)
        by_originality = data.get("by_originality", {})
        for json_key, metric_type in [
            ("original", "in_dist"),
            ("perturbed", "out_dist"),
        ]:
            orig_data = by_originality.get(json_key, {})
            acc = orig_data.get("accuracy", {})
            mean = acc.get("mean")
            std = acc.get("std")
            if mean is not None and std is not None:
                records.append(
                    {
                        "task": task,
                        "model": model,
                        "shots": shots,
                        "metric_type": metric_type,
                        "accuracy_mean": mean * 100,
                        "accuracy_std": std * 100,
                    }
                )

    return records


def filter_by_model_family(records: List[Dict], family: str) -> List[Dict]:
    """Filter records by model family."""
    if family == "all":
        return records
    if family == "qwen3":
        return [r for r in records if r["model"].startswith("Qwen3")]
    if family == "qwen2p5":
        return [r for r in records if r["model"].startswith("Qwen2.5")]
    if family == "api":
        api_names = {"gemini", "gpt", "gpt_noreasoning"}
        return [r for r in records if r["model"].lower() in api_names]
    return records


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_variance_scaling(
    ax, records, title, model_colors, model_order, baseline=None
):
    """Plot ICL scaling curves with shaded variance bands on the given axis."""
    by_model = defaultdict(dict)
    for r in records:
        by_model[r["model"]][r["shots"]] = (r["accuracy_mean"], r["accuracy_std"])

    # Plot in model_order, then any extras
    models = [m for m in model_order if m in by_model]
    models += [m for m in sorted(by_model) if m not in models]

    for model in models:
        color = model_colors.get(model, "black")
        data = by_model[model]
        shots = sorted(data)
        means = [data[s][0] for s in shots]
        stds = [data[s][1] for s in shots]

        ax.plot(
            shots, means, marker="o", linewidth=2, color=color, markersize=8, label=model
        )
        ax.fill_between(
            shots,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.2,
            color=color,
        )

    if baseline is not None:
        pct = baseline * 100
        ax.axhline(
            y=pct,
            color="gray",
            linestyle="--",
            linewidth=2,
            label=f"Random Guessing ({pct:.1f}%)",
            alpha=0.7,
            zorder=1,
        )

    ax.set_xlabel("Number of ICL Examples", fontweight="bold", fontsize=16)
    ax.set_ylabel("Accuracy (%)", fontweight="bold", fontsize=16)
    ax.set_title(title, fontweight="bold", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_ylim(0, 105)
    # Set xticks from actual data points
    all_shots = sorted(set(s for data in by_model.values() for s in data))
    ax.set_xticks(all_shots)
    ax.legend(loc="best", fontsize=12)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Variance Ablation Visualization")
    parser.add_argument(
        "--base_path",
        type=str,
        default=None,
        help="Root directory containing task/model/experiment/variance_summary.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for saved figures (default: output_vis/variance/visual_reasoning/variance_scaling/{model_family})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Filter to specific models (default: all)",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        choices=["qwen3", "qwen2p5", "api", "all"],
        default="all",
        help="Filter by model family (default: all)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures (default: 300)",
    )
    parser.add_argument("--results_dir", type=str, default="output",
                        help="Root of the experiment results")
    args = parser.parse_args()
    if args.base_path is None:
        args.base_path = str(Path(args.results_dir) / "variance/visual_reasoning")

    # Resolve model order for this run
    active_model_order = MODEL_FAMILY_MAP[args.model_family]

    # Style
    setup_grid_style(font_size=16)
    model_colors = {
        m: c
        for m, c in zip(active_model_order, sns.color_palette("husl", len(active_model_order)))
    }

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(
            f"output_vis/variance/visual_reasoning/variance_scaling/{args.model_family}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    all_records = collect_all_data(args.base_path)

    # Filter by model family
    all_records = filter_by_model_family(all_records, args.model_family)

    # Filter by specific models (further narrows if both flags used)
    if args.models:
        all_records = [r for r in all_records if r["model"] in args.models]

    # Summary
    tasks_found = sorted(set(r["task"] for r in all_records))
    models_found = sorted(set(r["model"] for r in all_records))
    print(f"Loaded {len(all_records)} records (model_family={args.model_family})")
    print(f"Tasks: {tasks_found}")
    print(f"Models: {models_found}")
    for task in tasks_found:
        task_recs = [r for r in all_records if r["task"] == task]
        types = sorted(set(r["metric_type"] for r in task_recs))
        shots = sorted(set(r["shots"] for r in task_recs))
        print(f"  {task}: metric_types={types}, shots={shots}")

    saved = []

    def save_fig(fig, name):
        path = output_dir / f"{name}.png"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        print(f"Saved: {path}")

    # Discover all task-split combinations (excluding originality types)
    originality_types = {"in_dist", "out_dist"}
    split_types = sorted(
        set(
            r["metric_type"]
            for r in all_records
            if r["metric_type"] not in originality_types
        )
    )

    # Per-split plots: one plot per (task, split) combination
    for task in DISPLAY_ORDER:
        for split in split_types:
            recs = [
                r
                for r in all_records
                if r["task"] == task and r["metric_type"] == split
            ]
            if not recs:
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            baseline = RANDOM_GUESSING.get(f"{task}_{split}")
            split_title = split.replace("_", " ").title()
            plot_variance_scaling(
                ax,
                recs,
                f"Variance Ablation: {task.replace('_', ' ').title()} ({split_title})",
                model_colors,
                active_model_order,
                baseline=baseline,
            )
            plt.tight_layout()
            save_fig(fig, f"{task}_{split}")

    # Originality plots (scaffolded — skipped silently if no data)
    for task in DISPLAY_ORDER:
        in_recs = [
            r for r in all_records if r["task"] == task and r["metric_type"] == "in_dist"
        ]
        out_recs = [
            r for r in all_records if r["task"] == task and r["metric_type"] == "out_dist"
        ]
        if not in_recs and not out_recs:
            continue
        fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(20, 8))
        baseline = RANDOM_GUESSING.get(f"{task}_complex")
        task_title = task.replace("_", " ").title()
        fig.suptitle(
            f"Variance Ablation: {task_title} (Originality)", fontweight="bold", fontsize=20
        )
        plot_variance_scaling(
            ax_in,
            in_recs,
            "In-Distribution",
            model_colors,
            active_model_order,
            baseline=baseline,
        )
        plot_variance_scaling(
            ax_out,
            out_recs,
            "Out-of-Distribution",
            model_colors,
            active_model_order,
            baseline=baseline,
        )
        plt.tight_layout()
        save_fig(fig, f"{task}_originality")

    print(f"\nAll {len(saved)} figures saved to: {output_dir}")


if __name__ == "__main__":
    main()

"""
python -m figures.extra.variance_ablation
python -m figures.extra.variance_ablation --model_family qwen3
python -m figures.extra.variance_ablation --models Qwen3-VL-4B-Instruct
"""
