"""Control Hidden Embedding Visualization — layer distance plots for clean vs corrupt conditions."""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from figures._plotting.models import QWEN3_MODELS
from figures._plotting.style import setup_grid_style

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_FAMILY_MAP = {
    "qwen3": QWEN3_MODELS,
    "all": QWEN3_MODELS,
}

# ── Data Loading ─────────────────────────────────────────────────────────────


def discover_json_files(base_path: str, model_family: str):
    """Discover hidden_embedding.json files under control_hidden_emb/."""
    base = Path(base_path)
    family_models = MODEL_FAMILY_MAP.get(model_family, MODEL_FAMILY_MAP["all"])
    entries = []

    for json_path in base.rglob("hidden_embedding.json"):
        parts = json_path.relative_to(base).parts
        # Expected: task/model/experiment/hidden_embedding.json
        if len(parts) < 4:
            continue
        task, model = parts[0], parts[1]
        if model in family_models:
            entries.append((task, model, json_path))

    return entries


def extract_condition_data(results_dict, condition, num_layers, split_name="overall"):
    """Extract per-k layer distance data for a given condition (clean/corrupt)."""
    data_by_k = {}
    for rkey, rval in results_dict.items():
        if not rkey.startswith(f"{condition}_k"):
            continue
        k = int(rkey.split("_k")[1])

        if split_name == "overall":
            section = rval.get("overall", {})
        else:
            section = rval.get("by_split", {}).get(split_name, {})

        if not section:
            continue

        layers, means, stds = [], [], []
        for i in range(num_layers):
            layer_data = section.get(f"layer_{i}")
            if layer_data is None:
                break
            layers.append(i)
            means.append(layer_data["mean"])
            stds.append(layer_data["std"])

        if layers:
            data_by_k[k] = {"layers": layers, "means": means, "stds": stds}

    return data_by_k


def assign_k_colors(k_values):
    """Assign a distinct color to each k value."""
    sorted_ks = sorted(k_values)
    colors = sns.color_palette("husl", len(sorted_ks))
    return {k: c for k, c in zip(sorted_ks, colors)}


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_layer_distances(ax, data_by_k, k_colors, title):
    """Plot mean cosine distance vs layer with all k values overlaid."""
    for k in sorted(data_by_k):
        data = data_by_k[k]
        color = k_colors[k]
        ax.plot(data["layers"], data["means"], linewidth=2, color=color, label=f"k={k}")
        ax.fill_between(
            data["layers"],
            [m - s for m, s in zip(data["means"], data["stds"])],
            [m + s for m, s in zip(data["means"], data["stds"])],
            alpha=0.15, color=color,
        )

    ax.set_xlabel("Layer Index", fontweight="bold", fontsize=16)
    ax.set_ylabel("Cosine Distance", fontweight="bold", fontsize=16)
    ax.set_title(title, fontweight="bold", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=12)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Control Hidden Embedding Visualization")
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
        args.base_path = str(Path(args.results_dir) / "control_hidden_emb/visual_reasoning")

    setup_grid_style(font_size=16)

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        f"output_vis/control_hidden_emb/{args.model_family}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = discover_json_files(args.base_path, args.model_family)
    if not entries:
        print("No hidden_embedding.json files found.")
        return

    tasks = sorted(set(e[0] for e in entries))
    print(f"Found {len(entries)} files across tasks: {tasks}")

    saved = []

    def save_fig(fig, name):
        path = output_dir / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        print(f"Saved: {path}")

    for task, model, json_path in entries:
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: {e}")
            continue

        metadata = data.get("metadata", {})
        num_layers = metadata.get("num_layers", 0)
        results = data.get("results", {})
        if not results or num_layers == 0:
            continue

        # Determine splits
        first_key = next(iter(results))
        first_val = results[first_key]
        splits = ["overall"]
        if "by_split" in first_val:
            splits += sorted(first_val["by_split"].keys())

        # Collect all k values for color assignment
        all_ks = set()
        for rkey in results:
            parts = rkey.split("_k")
            if len(parts) == 2:
                all_ks.add(int(parts[1]))
        k_colors = assign_k_colors(all_ks)

        task_title = task.replace("_", " ").title()

        for split_name in splits:
            split_title = split_name.replace("_", " ").title()

            for condition in ("clean", "corrupt"):
                data_by_k = extract_condition_data(results, condition, num_layers, split_name)
                if not data_by_k:
                    continue

                fig, ax = plt.subplots(figsize=(12, 6))
                plot_layer_distances(
                    ax, data_by_k, k_colors,
                    f"Control Embedding: {task_title} / {split_title} ({condition.title()}, {model})",
                )
                plt.tight_layout()
                save_fig(fig, f"{task}/{model}_{split_name}_{condition}")

    print(f"\nAll {len(saved)} figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
