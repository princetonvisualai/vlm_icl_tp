"""Control output distribution visualization — P(y_correct), entropy, KL divergence, per-GT shift."""

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


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _read_json(p: Path) -> Any:
    with open(p) as f:
        return json.load(f)


def discover_tasks(base_dir: Path) -> List[str]:
    """Return sorted list of task names found under base_dir (first-level dirs containing condition subdirs)."""
    tasks = []
    for child in sorted(base_dir.iterdir()):
        if child.is_dir() and (child / "clean").is_dir():
            tasks.append(child.name)
    return tasks


def load_condition_data(
    task_dir: Path, condition: str, k_values: List[int], splits: Optional[List[str]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Load overall.json per k per split for a given condition, returning {k: {split: overall_dict}}."""
    result = {}
    cond_dir = task_dir / condition
    if not cond_dir.exists():
        return result
    for k_dir in sorted(cond_dir.iterdir()):
        if not k_dir.is_dir() or not k_dir.name.startswith("k"):
            continue
        k = int(k_dir.name[1:])
        if k_values and k not in k_values:
            continue
        result[k] = {}
        for split_dir in sorted(k_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            if splits and split_dir.name not in splits:
                continue
            overall_path = split_dir / "overall.json"
            if overall_path.exists():
                result[k][split_dir.name] = _read_json(overall_path)
    return result


def _split_metric(data: Dict[int, Dict[str, Any]], metric: str, k_list: List[int], split: str) -> List[Optional[float]]:
    """Extract a metric for a single split for each k."""
    vals = []
    for k in k_list:
        if k not in data or split not in data[k]:
            vals.append(None)
            continue
        v = data[k][split].get(metric)
        vals.append(float(v) if v is not None else None)
    return vals


def _discover_splits(task_dirs: Dict[str, Path], condition: str) -> List[str]:
    """Discover available split names across tasks for a condition."""
    splits = set()
    for tdir in task_dirs.values():
        k0_dir = tdir / condition / "k0"
        if k0_dir.exists():
            for s in k0_dir.iterdir():
                if s.is_dir():
                    splits.add(s.name)
    return sorted(splits)


# ---------------------------------------------------------------------------
# Plot 1 & 2: Scaling curves (P(y_correct) or entropy)
# ---------------------------------------------------------------------------

def plot_scaling(
    task_dirs: Dict[str, Path],
    metric: str,
    ylabel: str,
    k_values: List[int],
    splits: Optional[List[str]],
    subplot_cols: int,
    *,
    model_label: str = "",
    std_metric: Optional[str] = None,
) -> List[Tuple[str, plt.Figure]]:
    """One figure per split. Returns list of (split_name, fig).

    If *std_metric* is provided, the corresponding values are loaded and
    plotted as a shaded ±1-std band around the main line.
    """
    tasks = sorted(task_dirs)
    n = len(tasks)
    if n == 0:
        return []

    # Discover all splits from data
    all_splits = set()
    for condition in ("clean", "corrupt"):
        all_splits |= set(_discover_splits(task_dirs, condition))
    if splits:
        all_splits &= set(splits)
    all_splits = sorted(all_splits)

    figs = []
    for split in all_splits:
        cols = min(n, subplot_cols)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows), squeeze=False)

        for idx, task in enumerate(tasks):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            for condition, color, ls in [("clean", "tab:blue", "-"), ("corrupt", "tab:red", "--")]:
                data = load_condition_data(task_dirs[task], condition, k_values, [split])
                ys = _split_metric(data, metric, k_values, split)
                xs = [k for k, y in zip(k_values, ys) if y is not None]
                ys_clean = [y for y in ys if y is not None]
                if xs:
                    ax.plot(xs, ys_clean, color=color, ls=ls, marker="o", markersize=5, lw=2, label=condition)
                    if std_metric:
                        stds = _split_metric(data, std_metric, k_values, split)
                        stds_clean = [s for s, y in zip(stds, ys) if y is not None]
                        if stds_clean and all(s is not None for s in stds_clean):
                            lo = [m - s for m, s in zip(ys_clean, stds_clean)]
                            hi = [m + s for m, s in zip(ys_clean, stds_clean)]
                            ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
            ax.set_title(task)
            ax.set_xlabel("k (demos)")
            if c == 0:
                ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)

        for j in range(n, rows * cols):
            r, c = divmod(j, cols)
            axes[r][c].axis("off")

        title = f"{ylabel} scaling — {split}"
        if model_label:
            title += f" — {model_label}"
        fig.suptitle(title, y=1.02)
        fig.tight_layout()
        figs.append((split, fig))

    return figs


# ---------------------------------------------------------------------------
# Plot 3: Per-GT distribution shift at selected k values
# ---------------------------------------------------------------------------

def plot_dist_shift_by_gt(
    task_dir: Path, condition: str, k_show: List[int],
    splits: Optional[List[str]], subplot_cols: int,
    *, task_name: Optional[str] = None,
) -> List[Tuple[str, plt.Figure]]:
    """One figure per (split). Rows = GT labels, cols = k values."""
    figs = []
    cond_dir = task_dir / condition
    if not cond_dir.exists():
        return figs

    # Discover splits from k0
    k0_dir = cond_dir / "k0"
    if not k0_dir.exists():
        return figs

    for split_dir in sorted(k0_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        if splits and split not in splits:
            continue

        label_path = split_dir / "label_order.json"
        if not label_path.exists():
            continue
        labels = _read_json(label_path)

        # Collect by_ground_truth for each k
        by_gt_per_k: Dict[int, Dict[str, Any]] = {}
        for k in [0] + k_show:
            bg_path = cond_dir / f"k{k}" / split / "by_ground_truth.json"
            if bg_path.exists():
                by_gt_per_k[k] = _read_json(bg_path)

        if not by_gt_per_k:
            continue

        gt_keys = sorted(set().union(*(d.keys() for d in by_gt_per_k.values())))
        k_list = sorted(by_gt_per_k.keys())
        n_gt = len(gt_keys)
        n_k = len(k_list)
        if n_gt == 0 or n_k == 0:
            continue

        fig, axes = plt.subplots(n_gt, n_k, figsize=(3.5 * n_k, 2.8 * n_gt), squeeze=False)
        for gi, gt in enumerate(gt_keys):
            for ki, k in enumerate(k_list):
                ax = axes[gi][ki]
                bg = by_gt_per_k.get(k, {}).get(gt)
                if bg and "mean_distribution" in bg:
                    dist = np.array(bg["mean_distribution"])
                    x = np.arange(len(labels))
                    colors = ["green" if labels[i] == gt else "tab:blue" for i in range(len(labels))]
                    ax.bar(x, dist, color=colors, alpha=0.85)
                    ax.set_ylim(0, 1)
                    ax.set_xticks(x)
                    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                if ki == 0:
                    ax.set_ylabel(f"GT={gt}", fontsize=8)
                if gi == 0:
                    ax.set_title(f"k={k}", fontsize=9)

        resolved_name = task_name or task_dir.name
        fig.suptitle(f"{resolved_name}/{split} — {condition}", y=1.02)
        fig.tight_layout()
        figs.append((f"{resolved_name}_{split}_{condition}", fig))

    return figs


# ---------------------------------------------------------------------------
# Plot 4: KL divergence from zero-shot
# ---------------------------------------------------------------------------

def plot_kl_from_zeroshot(
    task_dirs: Dict[str, Path],
    k_values: List[int],
    splits: Optional[List[str]],
    subplot_cols: int,
) -> List[Tuple[str, plt.Figure]]:
    """One figure per split. Returns list of (split_name, fig)."""
    tasks = sorted(task_dirs)
    n = len(tasks)
    if n == 0:
        return []

    # Discover all splits from data
    all_splits = set()
    for condition in ("clean", "corrupt"):
        all_splits |= set(_discover_splits(task_dirs, condition))
    if splits:
        all_splits &= set(splits)
    all_splits = sorted(all_splits)

    figs = []
    for split in all_splits:
        cols = min(n, subplot_cols)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows), squeeze=False)

        for idx, task in enumerate(tasks):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            for condition, color, ls in [("clean", "tab:blue", "-"), ("corrupt", "tab:red", "--")]:
                # Load k=0 distribution for this split
                data_0 = load_condition_data(task_dirs[task], condition, [0], [split])
                if 0 not in data_0 or split not in data_0[0]:
                    continue
                v0 = data_0[0][split]
                if "macro_mean_distribution" not in v0:
                    continue
                p0 = np.array(v0["macro_mean_distribution"])

                data_k = load_condition_data(task_dirs[task], condition, k_values, [split])
                xs, ys = [], []
                for k in k_values:
                    if k not in data_k or split not in data_k[k]:
                        continue
                    vk = data_k[k][split]
                    if "macro_mean_distribution" not in vk:
                        continue
                    pk = np.array(vk["macro_mean_distribution"])
                    if pk.shape != p0.shape:
                        continue
                    kl = float(np.sum(pk * np.log((pk + 1e-12) / (p0 + 1e-12))))
                    xs.append(k)
                    ys.append(kl)
                if xs:
                    ax.plot(xs, ys, color=color, ls=ls, marker="o", markersize=5, lw=2, label=condition)

            ax.set_title(task)
            ax.set_xlabel("k (demos)")
            if c == 0:
                ax.set_ylabel("KL(p_k || p_0)")
            ax.legend(fontsize=8)

        for j in range(n, rows * cols):
            r, c = divmod(j, cols)
            axes[r][c].axis("off")

        fig.suptitle(f"KL divergence from zero-shot — {split}", y=1.02)
        fig.tight_layout()
        figs.append((split, fig))

    return figs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Control Output Distribution Visualization")
    parser.add_argument("--control_dir", type=str, default=None,
                        help="Single model output dir (default: <results_dir>/control_output_dist/visual_reasoning)")
    parser.add_argument("--control_dirs", nargs="+", default=None,
                        help="Multiple model dirs for overlay comparison")
    parser.add_argument("--model_labels", nargs="+", default=None,
                        help="Legend labels for multi-model comparison")
    parser.add_argument("--output_dir", type=str, default="output_vis/control_output_dist")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--k_values", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16, 32])
    parser.add_argument("--k_show_dist", nargs="+", type=int, default=[0, 2, 4, 8, 16, 32],
                        help="k values to show in per-GT distribution shift plot")
    parser.add_argument("--subplot_cols", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--results_dir", type=str, default="output",
                        help="Root of the experiment results")
    args = parser.parse_args()

    setup_grid_style(figsize=None, font_size=11)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve input directories
    if args.control_dirs:
        base_dirs = [Path(d) for d in args.control_dirs]
    elif args.control_dir:
        base_dirs = [Path(args.control_dir)]
    else:
        base_dirs = [Path(args.results_dir) / "control_output_dist" / "visual_reasoning"]

    k_nonzero = [k for k in args.k_values if k > 0]
    all_k = args.k_values

    for base_dir in base_dirs:
        if not base_dir.exists():
            print(f"Skipping {base_dir}: not found")
            continue

        # Discover model dirs under base_dir (or treat base_dir as containing tasks directly)
        # Layout: base_dir/{task}/{model}/leaf/ or base_dir/{model}/leaf/{condition}/...
        # We look for dirs containing "clean" subdir at any depth
        model_task_dirs: Dict[str, Dict[str, Path]] = {}  # model -> {task -> task_dir}

        # Walk to find condition dirs
        for clean_dir in sorted(base_dir.rglob("clean")):
            parent = clean_dir.parent  # the leaf dir containing clean/corrupt
            # Infer task = grandparent or great-grandparent
            rel = parent.relative_to(base_dir)
            parts = rel.parts
            if len(parts) < 2:
                continue
            task = parts[0]
            model = parts[1] if len(parts) > 1 else "unknown"
            leaf = "/".join(parts[2:]) if len(parts) > 2 else ""

            if args.tasks and task not in args.tasks:
                continue

            key = f"{model}/{leaf}" if leaf else model
            model_task_dirs.setdefault(key, {})[task] = parent

        if not model_task_dirs:
            print(f"No data found under {base_dir}")
            continue

        for model_key, task_dirs in model_task_dirs.items():
            model_out = out / model_key
            model_out.mkdir(parents=True, exist_ok=True)
            print(f"Model: {model_key}, tasks: {sorted(task_dirs)}")

            # Plot 1: P(y_correct) scaling (per split)
            for split, fig in plot_scaling(task_dirs, "mean_p_correct", "P(y_correct)", all_k, args.splits, args.subplot_cols):
                path = model_out / f"p_correct_scaling_{split}.png"
                fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved: {path}")

            # Plot 2: Entropy scaling (per split)
            for split, fig in plot_scaling(task_dirs, "mean_entropy", "Entropy", all_k, args.splits, args.subplot_cols, std_metric="std_entropy"):
                path = model_out / f"entropy_scaling_{split}.png"
                fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved: {path}")

            # Plot 3: Per-GT distribution shift
            for task, tdir in sorted(task_dirs.items()):
                for condition in ("clean", "corrupt"):
                    for name, fig in plot_dist_shift_by_gt(tdir, condition, args.k_show_dist, args.splits, args.subplot_cols, task_name=task):
                        path = model_out / f"dist_shift_{name}.png"
                        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
                        plt.close(fig)
                        print(f"  Saved: {path}")

            # Plot 4: KL divergence (per split)
            for split, fig in plot_kl_from_zeroshot(task_dirs, k_nonzero, args.splits, args.subplot_cols):
                path = model_out / f"kl_from_zeroshot_{split}.png"
                fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved: {path}")

    print("Done.")


if __name__ == "__main__":
    main()
