"""Visualization for formal_lens_ablate_tokens: accuracy comparison, label distribution, and per-sample ablation figures."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from PIL import Image

from figures._plotting.style import (
    setup_style,
    finish_figure,
)


CONDITION_COLORS = {"baseline": "tab:blue", "ablate": "tab:orange", "8shot": "tab:green"}
CONDITIONS = ["baseline", "ablate", "8shot"]


def load_summary(input_dir: Path) -> Optional[Dict[str, Any]]:
    path = input_dir / "summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_results(input_dir: Path) -> Optional[List[Dict[str, Any]]]:
    path = input_dir / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_run_config(input_dir: Path) -> Optional[Dict[str, Any]]:
    path = input_dir / "run_config.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plot_accuracy(summary: Dict, output_dir: Path, dpi: int = 150) -> None:
    accuracy = summary["accuracy"]
    conds = [c for c in CONDITIONS if c in accuracy]
    vals = [accuracy[c] for c in conds]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(conds, vals, color=[CONDITION_COLORS[c] for c in conds], edgecolor="black", width=0.5)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, min(max(vals) + 0.15, 1.05))
    bias = summary.get("expected_bias", "")
    distract = summary.get("distract_cue", "")
    ax.set_title(f"Accuracy: baseline vs ablate vs 8-shot\n(bias={bias}, distractor={distract})")
    ax.grid(axis="y")
    finish_figure(fig, ax)
    fig.savefig(output_dir / "accuracy_comparison.png", dpi=dpi)
    plt.close(fig)


def plot_label_distribution(summary: Dict, output_dir: Path, dpi: int = 150) -> None:
    label_dist = summary["label_distribution"]
    conds = [c for c in CONDITIONS if c in label_dist]
    if not conds:
        return

    # Collect all labels across conditions
    all_labels = sorted({label for dist in label_dist.values() for label in dist})
    if not all_labels:
        return

    expected_bias = summary.get("expected_bias", "")

    n_conds = len(conds)
    fig, axes = plt.subplots(1, n_conds, figsize=(5 * n_conds, 4), squeeze=False)

    for i, cond in enumerate(conds):
        ax = axes[0][i]
        dist = label_dist[cond]
        probs = [dist.get(label, 0) for label in all_labels]
        colors = ["tab:red" if label == expected_bias else "steelblue" for label in all_labels]

        ax.bar(range(len(all_labels)), probs, color=colors, edgecolor="black", width=0.6)
        ax.set_xticks(range(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Frequency")
        ax.set_title(cond)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y")

    fig.suptitle(f"Predicted Label Distribution (red = expected_bias '{expected_bias}')")
    sns.despine(fig=fig)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / "label_distribution.png", dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-sample ablation visualization
# ---------------------------------------------------------------------------

def _load_image_lookup(dataset: str, tasks: Optional[List[str]]) -> Dict[str, Dict]:
    """Load dataset and build image_id -> entry lookup."""
    try:
        from vlm_icl.data import prepare_data
        data = prepare_data(dataset=dataset, tasks=tasks)
        return {entry["image_id"]: entry for entry in data}
    except Exception as e:
        print(f"Warning: Could not load dataset for image lookup: {e}")
        return {}


def _resolve_query_image(entry_data: Dict, image_lookup: Dict) -> Optional[Image.Image]:
    """Resolve query image from dataset lookup."""
    image_id = entry_data.get("image_id")
    if not image_id or image_id not in image_lookup:
        return None
    try:
        from vlm_icl.images import resolve_image
        img = resolve_image(image_lookup[image_id]["image"], prefer_pil=True)
        if isinstance(img, str):
            img = Image.open(img)
        return img.convert("RGB")
    except Exception:
        return None


def plot_sample_ablation_figure(
    record: Dict[str, Any],
    answer_choices: List[str],
    query_image: Optional[Image.Image],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Create a composite figure: image with ablated positions (top) + compact answer prob bar (bottom)."""
    grid_thw = record.get("grid_thw")
    grid_positions = record.get("distract_grid_positions", [])
    baseline_cond = record["conditions"].get("baseline", {})
    ablate_cond = record["conditions"].get("ablate", {})
    eightshot_cond = record["conditions"].get("8shot", {})
    baseline_probs = baseline_cond.get("answer_probs")
    ablate_probs = ablate_cond.get("answer_probs")
    eightshot_probs = eightshot_cond.get("answer_probs")

    has_probs = baseline_probs is not None and ablate_probs is not None

    fig, (ax_img, ax_bar) = plt.subplots(
        2, 1, figsize=(7, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.25},
    )

    # --- Top panel: query image with ablated positions highlighted ---
    if query_image is not None and grid_thw is not None:
        _t, h, w = grid_thw
        ax_img.imshow(np.array(query_image), extent=[0, w, h, 0])

        # Draw ablated position highlights
        ablated_set = {(r, c) for r, c in grid_positions}
        for row, col in ablated_set:
            rect = mpatches.FancyBboxPatch(
                (col + 0.02, row + 0.02), 0.96, 0.96,
                boxstyle="round,pad=0.02",
                facecolor=(1.0, 0.2, 0.1, 0.45),
                edgecolor=(1.0, 0.1, 0.0, 0.8),
                linewidth=0.6,
            )
            ax_img.add_patch(rect)

        # Draw faint grid for context
        for row in range(h):
            for col in range(w):
                if (row, col) not in ablated_set:
                    rect = mpatches.FancyBboxPatch(
                        (col + 0.02, row + 0.02), 0.96, 0.96,
                        boxstyle="round,pad=0.02",
                        facecolor=(0.8, 0.8, 0.8, 0.08),
                        edgecolor=(0.5, 0.5, 0.5, 0.15),
                        linewidth=0.2,
                    )
                    ax_img.add_patch(rect)

        ax_img.set_xlim(0, w)
        ax_img.set_ylim(h, 0)
        ax_img.set_aspect("equal")

        # Legend
        legend_patch = mpatches.Patch(
            facecolor=(1.0, 0.2, 0.1, 0.45), edgecolor=(1.0, 0.1, 0.0, 0.8),
            label=f"Abated ({len(grid_positions)}/{h * w} tokens)",
        )
        ax_img.legend(handles=[legend_patch], loc="lower left", fontsize=8, framealpha=0.7)
    elif query_image is not None:
        ax_img.imshow(np.array(query_image))
    else:
        ax_img.text(0.5, 0.5, "Image not available", ha="center", va="center",
                    transform=ax_img.transAxes, fontsize=12, color="gray")

    ax_img.set_title("Query Image (ablated tokens highlighted)", fontsize=10)
    ax_img.axis("off")

    # --- Bottom panel: compact grouped bar chart of answer probs ---
    if has_probs:
        all_prob_keys = list(baseline_probs.keys()) + list(ablate_probs.keys())
        if eightshot_probs:
            all_prob_keys += list(eightshot_probs.keys())
        choices = answer_choices if answer_choices else sorted(set(all_prob_keys))
        x = np.arange(len(choices))
        has_8shot = eightshot_probs is not None
        n_bars = 3 if has_8shot else 2
        bar_width = 0.8 / n_bars

        baseline_vals = [baseline_probs.get(c, 0) for c in choices]
        ablate_vals = [ablate_probs.get(c, 0) for c in choices]
        eightshot_vals = [eightshot_probs.get(c, 0) for c in choices] if has_8shot else []

        if has_8shot:
            offsets = [-bar_width, 0, bar_width]
        else:
            offsets = [-bar_width / 2, bar_width / 2]

        bars_b = ax_bar.bar(x + offsets[0], baseline_vals, bar_width,
                            color=CONDITION_COLORS["baseline"], edgecolor="black",
                            linewidth=0.5, label="baseline")
        bars_a = ax_bar.bar(x + offsets[1], ablate_vals, bar_width,
                            color=CONDITION_COLORS["ablate"], edgecolor="black",
                            linewidth=0.5, label="ablate")
        if has_8shot:
            bars_8 = ax_bar.bar(x + offsets[2], eightshot_vals, bar_width,
                                color=CONDITION_COLORS["8shot"], edgecolor="black",
                                linewidth=0.5, label="8shot")

        # Annotate bar values
        all_bars = [bars_b, bars_a]
        if has_8shot:
            all_bars.append(bars_8)
        for bars in all_bars:
            for bar in bars:
                h_val = bar.get_height()
                if h_val > 0.01:
                    ax_bar.text(bar.get_x() + bar.get_width() / 2, h_val + 0.005,
                                f"{h_val:.2f}", ha="center", va="bottom", fontsize=6)

        # Highlight ground truth on x-axis
        gt = record.get("ground_truth", "")
        tick_labels = []
        for c in choices:
            if c == gt:
                tick_labels.append(f"*{c}*")
            else:
                tick_labels.append(c)

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

        # Bold the ground truth tick label
        for tick_label in ax_bar.get_xticklabels():
            if tick_label.get_text().startswith("*"):
                tick_label.set_fontweight("bold")
                tick_label.set_color("darkgreen")
                tick_label.set_text(tick_label.get_text().strip("*"))

        ax_bar.set_ylabel("Prob", fontsize=8)
        ax_bar.tick_params(axis="y", labelsize=7)
        all_vals = baseline_vals + ablate_vals + eightshot_vals
        max_val = max(all_vals) if all_vals else 0
        ax_bar.set_ylim(0, min(max_val + 0.15, 1.05))
        ax_bar.legend(fontsize=7, loc="upper right")
        ax_bar.grid(axis="y")
    else:
        # Fallback: show text comparison
        baseline_ans = baseline_cond.get("answer", "N/A")
        ablate_ans = ablate_cond.get("answer", "N/A")
        eightshot_ans = eightshot_cond.get("answer", "N/A")
        baseline_correct = baseline_cond.get("correct", False)
        ablate_correct = ablate_cond.get("correct", False)
        eightshot_correct = eightshot_cond.get("correct", False)
        text = (
            f"Baseline: {baseline_ans} ({'correct' if baseline_correct else 'wrong'})\n"
            f"Ablate:   {ablate_ans} ({'correct' if ablate_correct else 'wrong'})\n"
            f"8-shot:   {eightshot_ans} ({'correct' if eightshot_correct else 'wrong'})\n"
            f"GT:       {record.get('ground_truth', 'N/A')}"
        )
        ax_bar.text(0.5, 0.5, text, ha="center", va="center",
                    transform=ax_bar.transAxes, fontsize=10, family="monospace")
        ax_bar.axis("off")

    # Suptitle
    image_id = record.get("image_id", "?")
    gt = record.get("ground_truth", "?")
    baseline_ans = baseline_cond.get("answer", "?")
    ablate_ans = ablate_cond.get("answer", "?")
    eightshot_ans = eightshot_cond.get("answer", "?")
    baseline_mark = "Y" if baseline_cond.get("correct") else "N"
    ablate_mark = "Y" if ablate_cond.get("correct") else "N"
    eightshot_mark = "Y" if eightshot_cond.get("correct") else "N"
    fig.suptitle(
        f"{image_id}  |  GT={gt}  |  baseline={baseline_ans} ({baseline_mark})  |  ablate={ablate_ans} ({ablate_mark})  |  8shot={eightshot_ans} ({eightshot_mark})",
        fontsize=8, y=0.99,
    )

    sns.despine(ax=ax_bar)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_sample_figures(
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    dpi: int,
    n_samples: int = 10,
    dataset: str = "visual_reasoning",
    tasks: Optional[List[str]] = None,
) -> None:
    """Select samples and generate per-sample ablation figures."""
    # Resolve tasks from run_config if not provided
    if tasks is None:
        run_config = load_run_config(input_dir)
        if run_config and run_config.get("tasks"):
            tasks = run_config["tasks"]

    # Load dataset for image lookup
    image_lookup = _load_image_lookup(dataset, tasks)

    answer_choices = summary.get("answer_choices", [])

    # Filter to entries with distract tokens and answer_probs
    valid = [
        r for r in results
        if r.get("distract_grid_positions")
        and r["conditions"].get("baseline", {}).get("answer_probs") is not None
        and r["conditions"].get("ablate", {}).get("answer_probs") is not None
    ]

    # Group by (ground_truth, distract_cue) for diversity
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in valid:
        key = (r.get("ground_truth", ""), r.get("distract_cue", ""))
        groups[key].append(r)

    # Within each group: flipped-correctness first, then by most distract tokens
    for key in groups:
        flipped = [
            r for r in groups[key]
            if r["conditions"]["baseline"]["correct"] != r["conditions"]["ablate"]["correct"]
        ]
        rest = [r for r in groups[key] if r not in flipped]
        rest.sort(key=lambda r: r.get("n_distract_tokens", 0), reverse=True)
        groups[key] = flipped + rest

    # Round-robin across groups
    selected: List[Dict] = []
    group_iters = {k: iter(v) for k, v in sorted(groups.items())}
    while len(selected) < n_samples and group_iters:
        exhausted = []
        for key, it in group_iters.items():
            if len(selected) >= n_samples:
                break
            nxt = next(it, None)
            if nxt is not None:
                selected.append(nxt)
            else:
                exhausted.append(key)
        for key in exhausted:
            del group_iters[key]

    if not selected:
        # Fallback: use any results with grid positions (even without probs)
        fallback = [r for r in results if r.get("distract_grid_positions")]
        fb_groups: Dict[tuple, List[Dict]] = defaultdict(list)
        for r in fallback:
            fb_groups[(r.get("ground_truth", ""), r.get("distract_cue", ""))].append(r)
        fb_iters = {k: iter(v) for k, v in sorted(fb_groups.items())}
        while len(selected) < n_samples and fb_iters:
            exhausted = []
            for key, it in fb_iters.items():
                if len(selected) >= n_samples:
                    break
                nxt = next(it, None)
                if nxt is not None:
                    selected.append(nxt)
                else:
                    exhausted.append(key)
            for key in exhausted:
                del fb_iters[key]

    if not selected:
        print("  No samples with distract_grid_positions found, skipping per-sample plots")
        return

    samples_dir = output_dir / "samples"
    for i, record in enumerate(selected):
        image_id = record.get("image_id", f"unknown_{i}")
        gt = record.get("ground_truth", "unk")
        dc = record.get("distract_cue", "unk")
        query_image = _resolve_query_image(record, image_lookup)
        out_path = samples_dir / f"sample_{i:02d}_gt{gt}_dc{dc}_{image_id}.png"
        plot_sample_ablation_figure(record, answer_choices, query_image, out_path, dpi)

    print(f"  Saved {len(selected)} sample figures to: {samples_dir}")


def process_single(input_dir: Path, output_dir: Path, dpi: int,
                   n_samples: int = 10, dataset: str = "visual_reasoning",
                   tasks: Optional[List[str]] = None) -> None:
    summary = load_summary(input_dir)
    if summary is None:
        print(f"No summary.json in {input_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy(summary, output_dir, dpi)
    plot_label_distribution(summary, output_dir, dpi)

    # Per-sample ablation visualizations
    results = load_results(input_dir)
    if results:
        plot_sample_figures(results, summary, input_dir, output_dir, dpi,
                           n_samples=n_samples, dataset=dataset, tasks=tasks)

    print(f"Saved plots to: {output_dir}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize formal_lens_ablate_tokens results")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Path to a single results dir with summary.json")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Root dir to auto-discover (default: <results_dir>/formal_lens_ablate_tokens/<dataset>)")
    parser.add_argument("--results_dir", type=str, default="output",
                        help="Root of the experiment results")
    parser.add_argument("--output_dir", type=str, default="output_vis/formal_lens_ablate_tokens")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of per-sample ablation figures to generate")
    parser.add_argument("--dataset", type=str, default="visual_reasoning",
                        help="Dataset name for image lookup")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Task filter for dataset loading")
    args = parser.parse_args(argv)
    if not args.input_dir and not args.base_dir:
        args.base_dir = str(Path(args.results_dir) / "formal_lens_ablate_tokens" / args.dataset)
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    setup_style()

    if args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir) / input_dir.parent.name / input_dir.name
        process_single(input_dir, output_dir, args.dpi,
                       n_samples=args.n_samples, dataset=args.dataset, tasks=args.tasks)

    elif args.base_dir:
        base = Path(args.base_dir)
        for task_dir in sorted(base.iterdir()):
            if not task_dir.is_dir():
                continue
            for model_dir in sorted(task_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                for leaf_dir in sorted(model_dir.iterdir()):
                    if not leaf_dir.is_dir():
                        continue
                    if (leaf_dir / "summary.json").exists():
                        out = Path(args.output_dir) / task_dir.name / model_dir.name / leaf_dir.name
                        process_single(leaf_dir, out, args.dpi,
                                       n_samples=args.n_samples, dataset=args.dataset,
                                       tasks=args.tasks or [task_dir.name])
    else:
        print("Provide either --input_dir or --base_dir")


if __name__ == "__main__":
    main()

# --- Sample running commands ---
# Auto-discover all tasks/models:
#   uv run python -m figures.fig_lens_ablate_tokens --base_dir output/formal_lens_ablate_tokens/visual_reasoning_lens
#
# Single task/model/leaf:
#   uv run python -m figures.fig_lens_ablate_tokens --input_dir output/formal_lens_ablate_tokens/visual_reasoning/shape_identification/Qwen3-VL-4B-Instruct/npl40_octagon
#
# With per-sample ablation figures (more samples, higher DPI):
#   uv run python -m figures.fig_lens_ablate_tokens --base_dir output/formal_lens_ablate_tokens/visual_reasoning --n_samples 20 --dpi 300
#
# Specify dataset and tasks for image lookup:
#   uv run python -m figures.fig_lens_ablate_tokens --input_dir output/formal_lens_ablate_tokens/visual_reasoning/counting/Qwen3-VL-4B-Instruct/npl10_3 --dataset visual_reasoning --tasks counting
