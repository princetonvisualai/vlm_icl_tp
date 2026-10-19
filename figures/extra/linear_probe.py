"""Linear Probe Accuracy Visualization — standalone script.

Sample commands:
  uv run python -m figures.extra.linear_probe --model_family qwen3
  uv run python -m figures.extra.linear_probe --model_family all --setting sps1000_k1

Plots per-layer classification accuracy under zero-shot vs ICL conditions.

Plot types:
  - Per-model:   For each model, layer index vs accuracy with different shots overlaid.
  - Cross-model: For each shot count, one subplot per model (shared y-axis).
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from figures._plotting.models import QWEN25_MODELS, QWEN3_MODELS
from figures._plotting.style import setup_grid_style

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_FAMILY_MAP = {
    "qwen3": QWEN3_MODELS,
    "qwen2p5": QWEN25_MODELS,
    "all": QWEN3_MODELS + QWEN25_MODELS,
}

MODEL_SHORT_NAMES = {
    "Qwen3-VL-2B-Instruct": "Qwen3-2B",
    "Qwen3-VL-4B-Instruct": "Qwen3-4B",
    "Qwen3-VL-8B-Instruct": "Qwen3-8B",
    "Qwen3-VL-32B-Instruct": "Qwen3-32B",
    "Qwen3-VL-235B-A22B-Instruct": "Qwen3-235B",
    "Qwen2.5-VL-3B-Instruct": "Qwen2.5-3B",
    "Qwen2.5-VL-7B-Instruct": "Qwen2.5-7B",
    "Qwen2.5-VL-32B-Instruct": "Qwen2.5-32B",
    "Qwen2.5-VL-72B-Instruct": "Qwen2.5-72B",
}

# ── Data Loading ─────────────────────────────────────────────────────────────


def discover_all_json_files(base_path, model_family, setting=None):
    """Discover linear_probe.json files for all models in the given family."""
    base = Path(base_path)
    family_models = MODEL_FAMILY_MAP[model_family]

    entries = []
    for json_path in base.rglob("linear_probe.json"):
        parts = json_path.relative_to(base).parts
        # Expected: task/model/experiment/linear_probe.json
        if len(parts) != 4:
            continue
        task, model, experiment, _ = parts
        if model in family_models and (setting is None or experiment == setting):
            entries.append((task, model, experiment, json_path))

    if not entries:
        return []

    # Sort models by family order
    model_rank = {m: i for i, m in enumerate(family_models)}
    entries.sort(key=lambda e: (e[0], model_rank.get(e[1], 999)))

    models_found = sorted(set(e[1] for e in entries), key=lambda m: model_rank.get(m, 999))
    print(f"Found models: {models_found}")
    return entries


def load_probe_data(json_path):
    """Load and parse a linear_probe.json file."""
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {json_path}: {e}")
        return None


def extract_layer_accuracies(results_dict, num_layers):
    """Extract per-ICL-count layer accuracy data."""
    data = {}
    for icl_key in sorted(results_dict.keys(), key=lambda k: int(k.split("_")[1])):
        per_layer = results_dict[icl_key].get("per_layer", {})
        if not per_layer:
            continue

        layers = []
        accuracies = []
        for i in range(num_layers):
            layer_data = per_layer.get(f"layer_{i}")
            if layer_data is None:
                break
            layers.append(i)
            accuracies.append(layer_data["accuracy"])

        if layers:
            data[icl_key] = {"layers": layers, "accuracies": accuracies}

    return data


def assign_icl_colors(icl_keys):
    """Assign a distinct color to each ICL key, sorted by ICL count."""
    sorted_keys = sorted(icl_keys, key=lambda k: int(k.split("_")[1]))
    colors = sns.color_palette("husl", len(sorted_keys))
    return {k: c for k, c in zip(sorted_keys, colors)}


def assign_model_colors(model_names):
    """Assign a distinct color to each model."""
    colors = sns.color_palette("husl", len(model_names))
    return {m: c for m, c in zip(model_names, colors)}


def short_name(model):
    """Return a short display name for a model."""
    return MODEL_SHORT_NAMES.get(model, model)


def icl_label(icl_key):
    """Convert 'icl_0' to '0-shot', 'icl_2' to '2-shot', etc."""
    n = int(icl_key.split("_")[1])
    return "0-shot" if n == 0 else f"{n}-shot"


# ── Per-model plot ───────────────────────────────────────────────────────────


def plot_per_model(ax, layer_data, icl_colors, title):
    """Plot accuracy vs layer with all ICL counts overlaid for a single model."""
    for icl_key in sorted(layer_data.keys(), key=lambda k: int(k.split("_")[1])):
        d = layer_data[icl_key]
        ax.plot(
            d["layers"], d["accuracies"],
            linewidth=2, color=icl_colors[icl_key], label=icl_label(icl_key),
        )

    ax.set_xlabel("Layer Index", fontweight="bold", fontsize=16)
    ax.set_ylabel("Classification Accuracy", fontweight="bold", fontsize=16)
    ax.set_title(title, fontweight="bold", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=12)


# ── Cross-model plot ─────────────────────────────────────────────────────────


def plot_cross_model(fig, axes, models_layer_data, model_colors, icl_key, task):
    """Plot one subplot per model for a given ICL count, with shared y-axis."""
    models = list(models_layer_data.keys())
    n_models = len(models)

    # Compute shared y-axis range across all models for this ICL count
    y_min = float("inf")
    y_max = float("-inf")
    for model in models:
        accs = models_layer_data[model]["accuracies"]
        y_min = min(y_min, min(accs))
        y_max = max(y_max, max(accs))
    y_pad = (y_max - y_min) * 0.05
    y_lo = max(0, y_min - y_pad)
    y_hi = min(1, y_max + y_pad)

    flat_axes = axes.flatten()
    for idx, model in enumerate(models):
        ax = flat_axes[idx]
        d = models_layer_data[model]
        color = model_colors[model]

        ax.plot(d["layers"], d["accuracies"], linewidth=2, color=color)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(short_name(model), fontweight="bold", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)

    # Hide unused axes
    for idx in range(n_models, flat_axes.size):
        flat_axes[idx].set_visible(False)

    fig.suptitle(
        f"Linear Probe: {task.replace('_', ' ').title()} ({icl_label(icl_key)})",
        fontweight="bold", fontsize=18,
    )
    fig.supxlabel("Layer Index", fontweight="bold", fontsize=16)
    fig.supylabel("Classification Accuracy", fontweight="bold", fontsize=16)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Linear Probe Accuracy Visualization")
    parser.add_argument(
        "--base_path", type=str,
        default=None,
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--model_family", type=str,
        choices=["qwen3", "qwen2p5", "all"], default="all",
    )
    parser.add_argument("--setting", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--results_dir", type=str, default="output",
                        help="Root of the experiment results")
    args = parser.parse_args()
    if args.base_path is None:
        args.base_path = str(Path(args.results_dir) / "linear_probe/visual_reasoning_many")

    setup_grid_style(font_size=16)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"output_vis/linear_probe/visual_reasoning_many/{args.model_family}")
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = discover_all_json_files(args.base_path, args.model_family, args.setting)
    if not entries:
        print("No linear_probe.json files found.")
        return

    tasks_found = sorted(set(e[0] for e in entries))
    models_found = sorted(set(e[1] for e in entries),
                          key=lambda m: MODEL_FAMILY_MAP[args.model_family].index(m)
                          if m in MODEL_FAMILY_MAP[args.model_family] else 999)
    print(f"Found {len(entries)} files across tasks: {tasks_found}, models: {models_found}")

    saved = []

    def save_fig(fig, name):
        path = output_dir / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        print(f"Saved: {path}")

    # ── Load all data, indexed by (task, model) ──────────────────────────────
    # all_data[task][model] = {icl_key: {layers, accuracies}}
    all_data = defaultdict(dict)
    all_num_layers = {}

    for task, model, experiment, json_path in entries:
        data = load_probe_data(json_path)
        if data is None:
            continue

        metadata = data["metadata"]
        num_layers = metadata["num_layers"]
        results = data["results"]

        layer_data = extract_layer_accuracies(results, num_layers)
        if not layer_data:
            continue

        all_data[task][model] = layer_data
        all_num_layers[(task, model)] = num_layers

    # ── Per-model plots: each model gets one overlay figure per task ──────────
    # Gather all ICL keys across all models for consistent colors
    all_icl_keys = set()
    for task_models in all_data.values():
        for layer_data in task_models.values():
            all_icl_keys.update(layer_data.keys())
    icl_colors = assign_icl_colors(all_icl_keys)
    model_colors = assign_model_colors(models_found)

    for task in sorted(all_data.keys()):
        task_models = all_data[task]

        for model in models_found:
            if model not in task_models:
                continue
            layer_data = task_models[model]

            fig, ax = plt.subplots(figsize=(12, 6))
            plot_per_model(
                ax, layer_data, icl_colors,
                f"Linear Probe: {task.replace('_', ' ').title()} — {short_name(model)}",
            )
            plt.tight_layout()
            save_fig(fig, f"{task}/per_model/{short_name(model)}")

        # ── Cross-model plots: one figure per ICL count ──────────────────────
        # Collect which ICL keys are available across all models for this task
        icl_keys_in_task = set()
        for layer_data in task_models.values():
            icl_keys_in_task.update(layer_data.keys())

        for icl_key in sorted(icl_keys_in_task, key=lambda k: int(k.split("_")[1])):
            # Gather models that have this ICL key
            models_with_key = [
                m for m in models_found
                if m in task_models and icl_key in task_models[m]
            ]
            if not models_with_key:
                continue

            models_layer_data = {
                m: task_models[m][icl_key] for m in models_with_key
            }
            n_models = len(models_with_key)
            ncols = min(n_models, 3)
            nrows = math.ceil(n_models / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
            axes = np.atleast_2d(axes)
            if axes.shape[0] == 1 and nrows > 1:
                axes = axes.T

            plot_cross_model(fig, axes, models_layer_data, model_colors, icl_key, task)
            plt.tight_layout()
            save_fig(fig, f"{task}/cross_model/{icl_label(icl_key)}")

    print(f"\nAll {len(saved)} figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
