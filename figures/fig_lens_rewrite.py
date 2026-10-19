"""Visualization for formal_lens_rewrite experiment: survival, similarity, drift, and grid overlay plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from tqdm import tqdm

from figures._plotting.style import (
    setup_style,
    finish_figure,
)


# ---------------------------------------------------------------------------
# Rank helpers (backward-compatible with old results lacking rank fields)
# ---------------------------------------------------------------------------

def _rank_of_keyword(neighbors, keyword, top_k):
    """1-indexed rank of first neighbor matching *keyword*; top_k+1 if not found."""
    kw = keyword.lower()
    for i, n in enumerate(neighbors):
        if kw in n["token_str"].lower():
            return i + 1
    return top_k + 1


def _std(vals):
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def _compute_rank_from_results(results, distract_cue, top_k=5):
    """Extract or compute per-sample rank metrics from results."""
    if not distract_cue:
        return None
    per_sample = []
    for r in results:
        pos_ranks = []
        for p in r.get("per_position", []):
            if "rank_distract_baseline" in p:
                rb, ri = p["rank_distract_baseline"], p["rank_distract_icl"]
            else:
                rb_layers, ri_layers = [], []
                for neighbors in p.get("decoded_baseline", {}).values():
                    rb_layers.append(_rank_of_keyword(neighbors, distract_cue, top_k))
                for neighbors in p.get("decoded_icl", {}).values():
                    ri_layers.append(_rank_of_keyword(neighbors, distract_cue, top_k))
                rb = sum(rb_layers) / len(rb_layers) if rb_layers else top_k + 1
                ri = sum(ri_layers) / len(ri_layers) if ri_layers else top_k + 1
            pos_ranks.append((rb, ri))
        if pos_ranks:
            per_sample.append({
                "image_id": r["image_id"],
                "mean_rank_baseline": sum(b for b, _ in pos_ranks) / len(pos_ranks),
                "mean_rank_icl": sum(i for _, i in pos_ranks) / len(pos_ranks),
                "mean_rank_drop": sum(i - b for b, i in pos_ranks) / len(pos_ranks),
            })
    return per_sample if per_sample else None


def load_data(input_dir: Path):
    results_path = input_dir / "results.json"
    summary_path = input_dir / "summary.json"
    if not results_path.exists():
        return None, None
    with open(results_path) as f:
        results = json.load(f)
    summary = None
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    return results, summary


def plot_survival_bar(summary: Dict, output_dir: Path, dpi: int = 150) -> None:
    sr = summary["mean_survival_rate"]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["Baseline\n(always 1.0)", "8-shot"],
        [1.0, sr["mean"]],
        color=["tab:gray", "tab:blue"],
        width=0.5,
    )
    ax.errorbar(1, sr["mean"], yerr=sr["std"], color="black", capsize=5, fmt="none")
    ax.set_ylabel("Distractor Survival Rate")
    ax.set_title("Distractor Token Survival Under ICL")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y")
    finish_figure(fig, ax)
    fig.savefig(output_dir / "survival_bar.png", dpi=dpi)
    plt.close(fig)


def plot_similarity_scatter(results: List[Dict], output_dir: Path, dpi: int = 150) -> None:
    x = [r["aggregate"]["mean_distract_sim_delta"] for r in results]
    y = [r["aggregate"]["mean_gt_sim_delta"] for r in results]
    c = [r["aggregate"]["survival_rate"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(x, y, c=c, cmap="RdYlGn_r", vmin=0, vmax=1, s=50, edgecolors="k", linewidths=0.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Distractor Sim Delta (baseline - ICL)\n(positive = pushed away)")
    ax.set_ylabel("GT Sim Delta (ICL - baseline)\n(positive = pushed toward)")
    ax.set_title("Per-Sample Similarity Shift")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Survival Rate")
    ax.grid(True)
    finish_figure(fig, ax)
    fig.savefig(output_dir / "similarity_scatter.png", dpi=dpi)
    plt.close(fig)


def plot_drift_bar(summary: Dict, output_dir: Path, dpi: int = 150) -> None:
    d_dist = summary["mean_drift_distractor"]
    d_ctrl = summary["mean_drift_control"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["Distractor", "Control"],
        [d_dist["mean"], d_ctrl["mean"]],
        color=["tab:red", "tab:green"],
        width=0.5,
    )
    ax.errorbar(0, d_dist["mean"], yerr=d_dist["std"], color="black", capsize=5, fmt="none")
    ax.errorbar(1, d_ctrl["mean"], yerr=d_ctrl["std"], color="black", capsize=5, fmt="none")
    ax.set_ylabel("Representation Drift (cosine distance)")
    ax.set_title("Drift: Distractor vs Control Positions")
    ax.grid(axis="y")
    finish_figure(fig, ax)
    fig.savefig(output_dir / "drift_bar.png", dpi=dpi)
    plt.close(fig)


def plot_rank_bar(summary: Dict, results: List[Dict], output_dir: Path,
                  distract_cue: str = None, top_k: int = 5, dpi: int = 150) -> None:
    """Paired bar chart: distractor rank at baseline vs 8-shot."""
    if "mean_rank_distract_baseline" in summary:
        rb = summary["mean_rank_distract_baseline"]
        ri = summary["mean_rank_distract_icl"]
    else:
        rank_data = _compute_rank_from_results(results, distract_cue, top_k)
        if not rank_data:
            return
        rb_vals = [d["mean_rank_baseline"] for d in rank_data]
        ri_vals = [d["mean_rank_icl"] for d in rank_data]
        rb = {"mean": sum(rb_vals) / len(rb_vals), "std": _std(rb_vals)}
        ri = {"mean": sum(ri_vals) / len(ri_vals), "std": _std(ri_vals)}

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        ["Baseline", "8-shot"],
        [rb["mean"], ri["mean"]],
        color=["tab:blue", "tab:orange"],
        width=0.5,
    )
    ax.errorbar(0, rb["mean"], yerr=rb["std"], color="black", capsize=5, fmt="none")
    ax.errorbar(1, ri["mean"], yerr=ri["std"], color="black", capsize=5, fmt="none")
    ax.set_ylabel("Mean Distractor Rank (1=top, higher=pushed away)")
    ax.set_title("Distractor Rank: Baseline vs 8-shot ICL")
    ax.set_ylim(0, top_k + 1.5)
    ax.grid(axis="y")
    finish_figure(fig, ax)
    fig.savefig(output_dir / "rank_bar.png", dpi=dpi)
    plt.close(fig)


def plot_rank_drop_histogram(results: List[Dict], output_dir: Path,
                             distract_cue: str = None, top_k: int = 5,
                             dpi: int = 150) -> None:
    """Histogram of per-position rank drop (ICL rank - baseline rank)."""
    drops = []
    for r in results:
        for p in r.get("per_position", []):
            if "rank_distract_baseline" in p:
                drops.append(p["rank_distract_icl"] - p["rank_distract_baseline"])
            elif distract_cue:
                rb_layers, ri_layers = [], []
                for neighbors in p.get("decoded_baseline", {}).values():
                    rb_layers.append(_rank_of_keyword(neighbors, distract_cue, top_k))
                for neighbors in p.get("decoded_icl", {}).values():
                    ri_layers.append(_rank_of_keyword(neighbors, distract_cue, top_k))
                if rb_layers and ri_layers:
                    rb = sum(rb_layers) / len(rb_layers)
                    ri = sum(ri_layers) / len(ri_layers)
                    drops.append(ri - rb)

    if not drops:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(min(drops) - 0.5, max(drops) + 1.5, 1.0)
    ax.hist(drops, bins=bins, color="tab:purple", edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="No change")
    ax.axvline(np.mean(drops), color="green", linestyle="-", linewidth=1.5,
               label=f"Mean = {np.mean(drops):.2f}")
    ax.set_xlabel("Rank Drop (ICL rank - baseline rank)\nPositive = distractor pushed down")
    ax.set_ylabel("Count (positions)")
    ax.set_title("Distribution of Distractor Rank Drop")
    ax.legend()
    ax.grid(axis="y")
    finish_figure(fig, ax)
    fig.savefig(output_dir / "rank_drop_histogram.png", dpi=dpi)
    plt.close(fig)


def plot_grid_overlay(results: List[Dict], output_dir: Path,
                      image_lookup: Optional[Dict] = None,
                      max_samples: int = 10, dpi: int = 150) -> None:
    """Plot spatial grid showing distractor positions colored by survival/new status,
    with the original image preserved in the background."""
    grid_dir = output_dir / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    # Sort by number of distractors descending (most interesting first)
    sorted_results = sorted(results, key=lambda r: r["n_distract_tokens_baseline"], reverse=True)

    for r in tqdm(sorted_results[:max_samples], desc="Rendering grids"):
        grid_thw = r.get("grid_thw")
        if not grid_thw or len(grid_thw) < 3:
            continue

        _, h, w = grid_thw

        # Mark baseline distractor positions
        baseline_set = set(tuple(p) for p in r.get("distract_grid_positions_baseline", []))
        icl_set = set(tuple(p) for p in r.get("distract_grid_positions_icl", []))
        new_icl = icl_set - baseline_set

        # Color survived (red) vs rewritten (green) baseline distractors
        survived_positions = set()
        for pp in r.get("per_position", []):
            if pp["survival"]:
                survived_positions.add(tuple(pp["grid_pos"]))

        cell_px = max(60, 400 // max(h, w))
        fig_w = w * cell_px / dpi * 2.5
        fig_h = h * cell_px / dpi * 2.5

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # Show original image as background
        query_image = _resolve_query_image(r, image_lookup) if image_lookup else None

        if query_image is not None:
            img_resized = query_image.resize((w * cell_px, h * cell_px), Image.BILINEAR)
            ax.imshow(np.array(img_resized), extent=[0, w, h, 0])
        else:
            ax.set_facecolor("#f0f0f0")
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)

        # Overlay colored cells for distractor positions
        color_map = {
            "survived": (0.9, 0.2, 0.2),
            "rewritten": (0.2, 0.8, 0.2),
            "new": (0.2, 0.4, 0.9),
        }

        for row in range(h):
            for col in range(w):
                pos = (row, col)
                if pos in baseline_set:
                    if pos in survived_positions:
                        rgb = color_map["survived"]
                        alpha = 0.55
                    else:
                        rgb = color_map["rewritten"]
                        alpha = 0.55
                elif pos in new_icl:
                    rgb = color_map["new"]
                    alpha = 0.55
                else:
                    # Non-distractor: very faint overlay so image shows through
                    rect = mpatches.FancyBboxPatch(
                        (col + 0.02, row + 0.02), 0.96, 0.96,
                        boxstyle="round,pad=0.02",
                        facecolor=(0.8, 0.8, 0.8, 0.08),
                        edgecolor=(0.5, 0.5, 0.5, 0.05),
                        linewidth=0.2,
                    )
                    ax.add_patch(rect)
                    continue

                rect = mpatches.FancyBboxPatch(
                    (col + 0.02, row + 0.02), 0.96, 0.96,
                    boxstyle="round,pad=0.02",
                    facecolor=(*rgb, alpha),
                    edgecolor=(*rgb, min(alpha + 0.2, 1.0)),
                    linewidth=0.5,
                )
                ax.add_patch(rect)

        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal")
        ax.set_title(f"{r['image_id']} | {r['question_type']}\n"
                     f"Baseline: {r['n_distract_tokens_baseline']} | "
                     f"ICL: {r['n_distract_tokens_icl']} | "
                     f"Survival: {r['aggregate']['survival_rate']:.2f}",
                     fontsize=9)
        ax.axis("off")

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=(*color_map["survived"], 0.55), label="Survived"),
            mpatches.Patch(facecolor=(*color_map["rewritten"], 0.55), label="Rewritten"),
            mpatches.Patch(facecolor=(*color_map["new"], 0.55), label="New (ICL)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
        fig.tight_layout()
        fig.savefig(grid_dir / f"{r['image_id']}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def process_input_dir(input_dir: Path, output_dir: Path, dpi: int,
                      dataset: str = "visual_reasoning",
                      tasks: Optional[List[str]] = None) -> None:
    results, summary = load_data(input_dir)
    if results is None:
        print(f"No results.json in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load run_config for rank computation and task resolution
    run_config = _load_run_config(input_dir)
    distract_cue = run_config.get("distract_cue") if run_config else None
    top_k = run_config.get("top_k", 5) if run_config else 5

    if summary:
        plot_survival_bar(summary, output_dir, dpi)
        plot_drift_bar(summary, output_dir, dpi)
        plot_rank_bar(summary, results or [], output_dir, distract_cue, top_k, dpi)

    if results:
        plot_similarity_scatter(results, output_dir, dpi)
        plot_rank_drop_histogram(results, output_dir, distract_cue, top_k, dpi)

        # Resolve tasks from run_config if not provided
        effective_tasks = tasks
        if effective_tasks is None:
            if run_config and run_config.get("tasks"):
                effective_tasks = run_config["tasks"]

        image_lookup = _load_image_lookup(dataset, effective_tasks)
        plot_grid_overlay(results, output_dir, image_lookup=image_lookup, dpi=dpi)

    print(f"Saved plots to: {output_dir}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize formal_lens_rewrite experiment results")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Path to a single output dir with results.json")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Root dir to auto-discover all tasks (default: <results_dir>/formal_lens_rewrite/<dataset>)")
    parser.add_argument("--results_dir", type=str, default="output",
                        help="Root of the experiment results")
    parser.add_argument("--output_dir", type=str, default="output_vis/formal_lens_rewrite")
    parser.add_argument("--dataset", type=str, default="visual_reasoning",
                        help="Dataset name for loading original images (default: visual_reasoning)")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Task filter for dataset loading")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)
    if not args.input_dir and not args.base_dir:
        args.base_dir = str(Path(args.results_dir) / "formal_lens_rewrite" / args.dataset)
    return args


def _load_image_lookup(dataset: str, tasks: Optional[List[str]] = None) -> Dict:
    """Load dataset and build image_id -> entry lookup for background images."""
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


def _load_run_config(input_dir: Path) -> Optional[Dict]:
    path = input_dir / "run_config.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    setup_style()

    if args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir) / input_dir.parent.name / input_dir.name
        process_input_dir(input_dir, output_dir, args.dpi,
                          dataset=args.dataset, tasks=args.tasks)

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
                    if (leaf_dir / "results.json").exists():
                        out = Path(args.output_dir) / task_dir.name / model_dir.name / leaf_dir.name
                        process_input_dir(leaf_dir, out, args.dpi,
                                          dataset=args.dataset,
                                          tasks=args.tasks or [task_dir.name])
    else:
        print("Provide either --input_dir or --base_dir")


if __name__ == "__main__":
    main()

# --- Sample running commands ---
# Auto-discover all tasks/models:
#   uv run python -m figures.fig_lens_rewrite --base_dir output/formal_lens_rewrite/visual_reasoning
#
# Single output dir:
#   uv run python -m figures.fig_lens_rewrite --input_dir output/formal_lens_rewrite/visual_reasoning/counting/Qwen3-VL-4B-Instruct/npl10_octagon
