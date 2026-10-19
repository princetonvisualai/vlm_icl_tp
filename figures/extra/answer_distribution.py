"""Answer Distribution Visualization — standalone script.

Generates per-answer-label distribution plots from answer_dist output directories, saving plots to output_vis/.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from figures._plotting.style import setup_grid_style

# ── Loading + plotting helpers (from answer_distribution.ipynb) ──────────────


def _read_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _is_partition_dir(d: Path) -> bool:
    """Check if a directory contains overall.json and label_order.json."""
    return (d / "overall.json").exists() and (d / "label_order.json").exists()


def discover_partitions(answer_dist_dir: Path) -> List[Tuple[str, str, Path]]:
    """Discover (task, split, partition_dir) tuples within an answer_distributions directory, handling both multi-task and single-task layouts."""
    parts: List[Tuple[str, str, Path]] = []

    # Try two-level first: child=task, grandchild=split
    for child in sorted(p for p in answer_dist_dir.iterdir() if p.is_dir()):
        for grandchild in sorted(p for p in child.iterdir() if p.is_dir()):
            if _is_partition_dir(grandchild):
                parts.append((child.name, grandchild.name, grandchild))
    if parts:
        return parts

    # Fall back to one-level: child=split, task read from overall.json
    for child in sorted(p for p in answer_dist_dir.iterdir() if p.is_dir()):
        if _is_partition_dir(child):
            overall = _read_json(child / "overall.json")
            task = overall.get("partition", {}).get("task", "unknown")
            parts.append((task, child.name, child))
    return parts


def collect_all_partitions(
    base_dir: Path,
) -> List[Tuple[str, str, str, Path]]:
    """Find all partition dirs containing overall.json and label_order.json, returning (rel_context, task, split, partition_dir) tuples."""
    # Try base_dir itself via discover_partitions (handles direct + two-level)
    direct = discover_partitions(base_dir)
    if direct:
        return [(".", task, split, pdir) for task, split, pdir in direct]

    # Recursive: find all dirs containing overall.json + label_order.json
    results: List[Tuple[str, str, str, Path]] = []
    for overall_path in sorted(base_dir.rglob("overall.json")):
        pdir = overall_path.parent
        if not _is_partition_dir(pdir):
            continue
        rel = str(pdir.relative_to(base_dir))
        meta = _read_json(overall_path)
        task = meta.get("partition", {}).get("task", pdir.parent.name)
        split = meta.get("partition", {}).get("split", pdir.name)
        results.append((rel, task, split, pdir))
    return results


def load_partition(partition_dir: Path) -> Dict[str, Any]:
    labels = _read_json(partition_dir / "label_order.json")
    overall = _read_json(partition_dir / "overall.json")
    by_gt_path = partition_dir / "by_ground_truth.json"
    by_gt = _read_json(by_gt_path) if by_gt_path.exists() else {}
    return {"labels": labels, "overall": overall, "by_ground_truth": by_gt}


def gaussian_kernel1d(sigma: float, *, radius: Optional[int] = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    if radius is None:
        radius = int(math.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k


def smooth_distribution(probs: np.ndarray, *, sigma: float) -> np.ndarray:
    """Apply discrete Gaussian smoothing over label indices, renormalized to sum to 1."""
    probs = np.asarray(probs, dtype=np.float64)
    if sigma <= 0 or probs.size == 0:
        return probs
    k = gaussian_kernel1d(sigma)
    pad = k.size // 2
    padded = np.pad(probs, (pad, pad), mode="reflect")
    sm = np.convolve(padded, k, mode="valid")
    sm = np.clip(sm, 0.0, None)
    s = sm.sum()
    if s > 0:
        sm /= s
    return sm


def plot_bar_with_kernel(
    ax: plt.Axes,
    labels: List[str],
    probs: np.ndarray,
    *,
    title: str,
    sigma: float,
    show_ylabel: bool = True,
    highlight_index: Optional[int] = None,
) -> None:
    x = np.arange(len(labels))
    probs = np.asarray(probs, dtype=np.float64)

    # Create color array: green for highlighted bar, default blue for others
    if highlight_index is not None and 0 <= highlight_index < len(labels):
        colors = ["tab:blue"] * len(labels)
        colors[highlight_index] = "green"
        ax.bar(x, probs, alpha=0.85, color=colors)
    else:
        ax.bar(x, probs, alpha=0.85)

    if sigma > 0:
        sm = smooth_distribution(probs, sigma=sigma)
        ax.plot(x, sm, color="black", linewidth=2, label=f"kernel σ={sigma}")

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    if show_ylabel:
        ax.set_ylabel("Probability")
    ax.set_xlabel("Answer label")


def ordered_gt_keys(labels: List[str], by_gt: Dict[str, Any]) -> List[str]:
    present = set(by_gt.keys())
    ordered = [l for l in labels if l in present]
    ordered += sorted([k for k in by_gt.keys() if k not in set(ordered)])
    return ordered


# ── By-ground-truth subplot figure ──────────────────────────────────────────


def plot_by_ground_truth(
    partition_dir: Path, *, sigma: float, subplot_cols: int
) -> Optional[plt.Figure]:
    """Create a by-ground-truth subplot figure for one partition, or return None if no data exists."""
    payload = load_partition(partition_dir)
    labels = payload["labels"]
    overall = payload["overall"]
    by_gt = payload["by_ground_truth"]

    if not by_gt:
        return None

    task = overall.get("partition", {}).get("task", partition_dir.parent.name)
    split = overall.get("partition", {}).get("split", partition_dir.name)

    gt_keys = ordered_gt_keys(labels, by_gt)
    n = len(gt_keys)
    cols = max(1, int(subplot_cols))
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.8 * rows), squeeze=False)
    for i, gt in enumerate(gt_keys):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        obj = by_gt[gt]
        dist = np.asarray(obj["mean_distribution"], dtype=np.float64)
        count = obj.get("count")

        # Find the index of the ground truth label to highlight it
        gt_index = labels.index(gt) if gt in labels else None

        plot_bar_with_kernel(
            ax,
            labels,
            dist,
            title=f"GT={gt} (n={count})",
            sigma=sigma,
            show_ylabel=(c == 0),
            highlight_index=gt_index,
        )
        if sigma > 0:
            ax.legend(loc="upper right")

    # Turn off unused axes
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    fig.suptitle(f"Distributions by ground-truth — {task}/{split}", y=1.02)
    fig.tight_layout()
    return fig


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Answer Distribution Visualization")
    parser.add_argument(
        "--answer_dist_dir",
        type=str,
        default=None,
        help="Root directory containing answer distribution outputs "
        "(default: <results_dir>/answer_dist/visual_reasoning)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for saved figures (default: output_vis/answer_dist/visual_reasoning)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Filter to specific tasks (default: all)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Filter to specific splits (default: all)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Kernel smoothing sigma (default: 1.0, 0 to disable)",
    )
    parser.add_argument(
        "--subplot_cols",
        type=int,
        default=3,
        help="Number of columns in subplot grid (default: 3)",
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
    if args.answer_dist_dir is None:
        args.answer_dist_dir = str(Path(args.results_dir) / "answer_dist/visual_reasoning")

    # Style
    setup_grid_style(font_size=12)

    # Validate input
    base_dir = Path(args.answer_dist_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Not found: {base_dir}")

    # Discover partitions (direct or recursive)
    all_parts = collect_all_partitions(base_dir)

    # Apply filters
    if args.tasks:
        task_set = set(args.tasks)
        all_parts = [p for p in all_parts if p[1] in task_set]
    if args.splits:
        split_set = set(args.splits)
        all_parts = [p for p in all_parts if p[2] in split_set]

    if not all_parts:
        raise ValueError(f"No partitions found under: {base_dir}")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("output_vis/answer_dist/visual_reasoning")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary
    print(f"Found {len(all_parts)} partitions:")
    for rel, task, split, pdir in all_parts:
        print(f"  [{rel}] {task}/{split}")

    # Generate figures
    saved = []
    for rel, task, split, pdir in all_parts:
        fig = plot_by_ground_truth(pdir, sigma=args.sigma, subplot_cols=args.subplot_cols)
        if fig is None:
            print(f"Skipping [{rel}] {task}/{split}: no by_ground_truth data")
            continue

        # Mirror relative path structure in output
        rel_to_base = pdir.relative_to(base_dir)
        parent_rel = rel_to_base.parent
        sub_dir = output_dir / parent_rel
        sub_dir.mkdir(parents=True, exist_ok=True)
        path = sub_dir / f"{task}_{split}_by_gt.png"

        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        print(f"Saved: {path}")

    print(f"\nAll {len(saved)} figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
