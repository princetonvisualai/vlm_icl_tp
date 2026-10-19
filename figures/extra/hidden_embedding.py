"""Hidden Embedding Distance Visualization — standalone script.

Plots cosine distances between zero-shot and ICL embeddings across model layers.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from figures._plotting.models import QWEN25_MODELS, QWEN3_MODELS
from figures._plotting.style import setup_grid_style

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_FAMILY_MAP = {
    "qwen3": QWEN3_MODELS,
    "qwen2p5": QWEN25_MODELS,
    "all": QWEN3_MODELS + QWEN25_MODELS,
}

# ── Data Loading ─────────────────────────────────────────────────────────────


def discover_json_files(base_path, model_family, setting=None):
    """Discover hidden_embedding.json files and select the first matching model from the given family."""
    base = Path(base_path)
    family_models = MODEL_FAMILY_MAP[model_family]

    entries = []
    for json_path in base.rglob("hidden_embedding.json"):
        parts = json_path.relative_to(base).parts
        # Expected: task/model/experiment/hidden_embedding.json
        if len(parts) != 4:
            continue
        task, model, experiment, _ = parts
        if model in family_models and (setting is None or experiment == setting):
            entries.append((task, model, experiment, json_path))

    if not entries:
        return []

    # Pick the first model found in family order
    models_found = set(e[1] for e in entries)
    selected_model = None
    for m in family_models:
        if m in models_found:
            selected_model = m
            break

    print(f"Selected model: {selected_model}")
    return [e for e in entries if e[1] == selected_model]


def load_embedding_data(json_path):
    """Load and parse a hidden_embedding.json file."""
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {json_path}: {e}")
        return None


def extract_split_data(results_dict, split_name, num_layers):
    """Extract per-ICL-count layer distance data (means and stds) for a given split."""
    split_data = {}
    for icl_key in sorted(results_dict.keys(), key=lambda k: int(k.split("_")[1])):
        icl_result = results_dict[icl_key]
        # Navigate to the right section
        if split_name == "overall":
            section = icl_result.get("overall", {})
        else:
            section = icl_result.get("by_split", {}).get(split_name, {})

        if not section:
            continue

        layers = []
        means = []
        stds = []
        for i in range(num_layers):
            layer_data = section.get(f"layer_{i}")
            if layer_data is None:
                break
            layers.append(i)
            means.append(layer_data["mean"])
            stds.append(layer_data["std"])

        if layers:
            split_data[icl_key] = {"layers": layers, "means": means, "stds": stds}

    return split_data


def assign_icl_colors(icl_keys):
    """Assign a distinct color to each ICL key, sorted by ICL count."""
    sorted_keys = sorted(icl_keys, key=lambda k: int(k.split("_")[1]))
    colors = sns.color_palette("husl", len(sorted_keys))
    return {k: c for k, c in zip(sorted_keys, colors)}


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_means_overlay(ax, split_data, icl_colors, title):
    """Plot mean cosine distance vs layer with all ICL counts overlaid on one axis."""
    for icl_key in sorted(split_data.keys(), key=lambda k: int(k.split("_")[1])):
        data = split_data[icl_key]
        n = int(icl_key.split("_")[1])
        ax.plot(
            data["layers"],
            data["means"],
            linewidth=2,
            color=icl_colors[icl_key],
            label=f"ICL {n}",
        )

    ax.set_xlabel("Layer Index", fontweight="bold", fontsize=16)
    ax.set_ylabel("Cosine Distance", fontweight="bold", fontsize=16)
    ax.set_title(title, fontweight="bold", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=12)


def plot_subfigure_grid(fig, axes, split_data, icl_colors, task, split_name):
    """Plot a grid of subplots with one panel per ICL count, showing mean and std band."""
    sorted_keys = sorted(split_data.keys(), key=lambda k: int(k.split("_")[1]))
    n_plots = len(sorted_keys)

    # Compute shared y-axis range across all ICL counts
    y_min = float("inf")
    y_max = float("-inf")
    for icl_key in sorted_keys:
        data = split_data[icl_key]
        for m, s in zip(data["means"], data["stds"]):
            y_min = min(y_min, m - s)
            y_max = max(y_max, m + s)
    y_pad = (y_max - y_min) * 0.05
    y_lo = max(0, y_min - y_pad)
    y_hi = y_max + y_pad

    for idx, icl_key in enumerate(sorted_keys):
        row, col = divmod(idx, 3)
        ax = axes[row, col] if axes.ndim > 1 else axes[col]
        data = split_data[icl_key]
        n = int(icl_key.split("_")[1])
        color = icl_colors[icl_key]
        means = data["means"]
        stds = data["stds"]
        layers = data["layers"]

        ax.plot(layers, means, linewidth=2, color=color)
        ax.fill_between(
            layers,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.2,
            color=color,
        )
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(f"ICL {n}", fontweight="bold", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)

    # Hide unused axes
    total_slots = axes.size
    for idx in range(n_plots, total_slots):
        row, col = divmod(idx, 3)
        ax = axes[row, col] if axes.ndim > 1 else axes[col]
        ax.set_visible(False)

    split_title = split_name.replace("_", " ").title()
    fig.suptitle(
        f"Hidden Embedding Distance: {task.replace('_', ' ').title()} / {split_title}",
        fontweight="bold",
        fontsize=18,
    )
    fig.supxlabel("Layer Index", fontweight="bold", fontsize=16)
    fig.supylabel("Cosine Distance", fontweight="bold", fontsize=16)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Hidden Embedding Distance Visualization")
    parser.add_argument(
        "--base_path",
        type=str,
        default=None,
        help="Root directory containing task/model/experiment/hidden_embedding.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for saved figures (default: output_vis/hidden_embedding/visual_reasoning/{model_family})",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        choices=["qwen3", "qwen2p5", "all"],
        default="all",
        help="Filter by model family (default: all)",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default=None,
        help="Filter by experiment setting, e.g. 'sps10_k2' or 'sps50_k2' (default: all settings)",
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
        args.base_path = str(Path(args.results_dir) / "hidden_embedding/visual_reasoning")

    # Style
    setup_grid_style(font_size=16)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(
            f"output_vis/hidden_embedding/visual_reasoning/{args.model_family}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover files
    entries = discover_json_files(args.base_path, args.model_family, args.setting)
    if not entries:
        print("No hidden_embedding.json files found.")
        return

    tasks_found = sorted(set(e[0] for e in entries))
    print(f"Found {len(entries)} files across tasks: {tasks_found}")

    saved = []

    def save_fig(fig, name):
        path = output_dir / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        print(f"Saved: {path}")

    for task, model, experiment, json_path in entries:
        data = load_embedding_data(json_path)
        if data is None:
            continue

        metadata = data["metadata"]
        num_layers = metadata["num_layers"]
        results = data["results"]

        # Build ICL color map
        icl_keys = list(results.keys())
        icl_colors = assign_icl_colors(icl_keys)

        # Determine splits from first ICL key
        first_icl = results[icl_keys[0]]
        splits = ["overall"]
        if "by_split" in first_icl:
            splits += sorted(first_icl["by_split"].keys())

        for split_name in splits:
            split_data = extract_split_data(results, split_name, num_layers)
            if not split_data:
                continue

            split_title = split_name.replace("_", " ").title()

            # Means overlay plot
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_means_overlay(
                ax,
                split_data,
                icl_colors,
                f"Hidden Embedding Distance: {task.replace('_', ' ').title()} ({split_title})",
            )
            plt.tight_layout()
            save_fig(fig, f"{task}/{split_name}_means")

            # Subfigure grid
            n_plots = len(split_data)
            ncols = min(n_plots, 3)
            nrows = math.ceil(n_plots / 3)
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
            if nrows == 1 and ncols == 1:
                import numpy as np

                axes = np.array([[axes]])
            elif axes.ndim == 1:
                axes = axes.reshape(1, -1) if nrows == 1 else axes.reshape(-1, 1)
            plot_subfigure_grid(fig, axes, split_data, icl_colors, task, split_name)
            plt.tight_layout()
            save_fig(fig, f"{task}/{split_name}_grid")

    print(f"\nAll {len(saved)} figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
