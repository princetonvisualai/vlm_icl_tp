"""Visualization for formal_icl_scale: accuracy vs ICL shots aggregated across all tasks.

Generates 4 groups of plots (overall, simple, noprior, complex), each with:
- Cross-model comparison (all models on one figure)
- Per-family comparison (one figure per model family)

Also generates a single 2x2 subplot figure for qwen3vl showing all 4 splits.

Aggregation is weighted by sample count: sum(correct) / sum(samples) across tasks,
NOT simple average of per-task accuracies.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figures._plotting.io import (
    discover_model_summaries,
    extract_split_from_leaf,
    aggregate_across_tasks,
    aggregate_across_tasks_with_variance,
    to_summaries_format,
    to_summaries_format_with_variance,
)
from figures._plotting.plots import (
    plot_accuracy_cross_model,
    plot_accuracy_per_family,
)
from figures._plotting.style import (
    setup_style,
    finish_figure,
)
from figures._plotting.models import (
    get_model_color,
    get_model_family,
)
from figures._plotting.models import (
    FAMILY_ORDER,
)
import matplotlib.lines as mlines


def _add_family_grid_legend(
    fig,
    handles_map: dict[str, Any],
    *,
    fontsize: str = "medium",
    bottom: float = 0.14,
    bbox_to_anchor: Optional[tuple[float, float]] = None,
) -> None:
    """Figure-level legend with one column per model family, rows = sizes.

    Mirrors ``add_family_legend`` but accepts a pre-collected ``{label: handle}``
    mapping so it works across multi-axes figures.
    """
    if not handles_map:
        return

    def _size_key(name: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)[Bb]", name)
        return float(m.group(1)) if m else float("inf")

    family_models: dict[str, list[str]] = {}
    for name in handles_map:
        family_models.setdefault(get_model_family(name), []).append(name)
    for fam in family_models:
        family_models[fam].sort(key=_size_key)

    family_order = [f for f in FAMILY_ORDER if f in family_models]
    for f in family_models:
        if f not in family_order:
            family_order.append(f)

    max_rows = max(len(v) for v in family_models.values())
    ncol = len(family_order)

    dummy = mlines.Line2D([], [], color="none", marker="None", linestyle="None")
    grid_handles: list = []
    grid_labels: list[str] = []
    for fam in family_order:
        models = family_models[fam]
        for row in range(max_rows):
            if row < len(models):
                grid_handles.append(handles_map[models[row]])
                grid_labels.append(models[row])
            else:
                grid_handles.append(dummy)
                grid_labels.append(" ")

    legend_kwargs: dict[str, Any] = dict(
        loc="lower center",
        ncol=ncol,
        fontsize=fontsize,
        frameon=True,
        edgecolor="0.8",
        facecolor="white",
        framealpha=0.95,
        columnspacing=1.4,
        handletextpad=0.5,
        labelspacing=0.4,
    )
    if bbox_to_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = bbox_to_anchor
    fig.legend(grid_handles, grid_labels, **legend_kwargs)
    fig.subplots_adjust(bottom=bottom + 0.035 * max_rows)


def _qwen3vl_filter(model_name: str, family: str) -> bool:
    """Keep only 4B and 32B for Qwen3-VL in cross-model plots."""
    if family == "qwen3vl":
        return any(size in model_name.lower() for size in {"4b", "32b"})
    return True


def plot_qwen3vl_combined(
    aggregated: dict[str, dict[str, dict[int, tuple[int, int]]]],
    output_dir: Path,
    *,
    dpi: int = 150,
) -> None:
    """Create a single 1x4 figure with all 4 splits for qwen3vl models only.

    Args:
        aggregated: {split: {model: {n_shots: (correct, samples)}}}
        output_dir: Output directory for the plot
        dpi: DPI for the saved figure
    """
    groups = ["overall", "complex", "simple", "noprior"]
    titles = ["Overall", "Complex", "Simple", "No-Prior"]

    # Filter to qwen3vl models only
    qwen3vl_data: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for group in groups:
        if group not in aggregated or not aggregated[group]:
            continue
        qwen3vl_models = {
            name: data for name, data in aggregated[group].items()
            if get_model_family(name) == "qwen3vl"
        }
        if qwen3vl_models:
            qwen3vl_data[group] = to_summaries_format(qwen3vl_models)

    if not qwen3vl_data:
        print("No qwen3vl data found for combined plot")
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    def _size_key(model_name: str) -> float:
        """Extract size in B for sorting (smaller first)."""
        m = re.search(r"(\d+(?:\.\d+)?)[Bb]", model_name)
        return float(m.group(1)) if m else float("inf")

    for idx, (group, title) in enumerate(zip(groups, titles)):
        ax = axes[idx]

        if group not in qwen3vl_data:
            ax.set_visible(False)
            continue

        model_summaries = qwen3vl_data[group]

        for model_name in sorted(model_summaries.keys(), key=_size_key):
            summaries = model_summaries[model_name]
            shots = [s["n_shots"] for s in summaries]
            overall = [s["accuracy"]["overall"] for s in summaries]
            display_name = model_name.replace("-Instruct", "")
            ax.plot(shots, overall, "o-", label=display_name, color=get_model_color(model_name))

        ax.set_xlabel("Number of ICL shots", fontsize=18)
        if idx == 0:
            ax.set_ylabel("Accuracy", fontsize=22)
        ax.set_title(title, fontsize=24, fontweight="bold")
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=18)

        all_shots = sorted({
            s["n_shots"]
            for sums in model_summaries.values()
            for s in sums
        })
        if all_shots:
            ax.set_xticks(all_shots)
        ax.grid(True)

    # Single horizontal legend spanning the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center",
            ncol=len(handles),
            fontsize=22,
            frameon=True,
            edgecolor="0.8",
        )

    finish_figure(fig, axes)
    fig.tight_layout(rect=[0, 0.15, 1, 1.0])

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "qwen3vl_combined.png", dpi=dpi)
    plt.close(fig)
    print(f"Saved qwen3vl combined plot to: {output_dir / 'qwen3vl_combined.png'}")


def plot_qwen3vl_complex_per_task(
    model_summaries: dict[str, dict[str, dict[str, tuple[Path, list[dict[str, Any]]]]]],
    output_dir: Path,
    *,
    dpi: int = 150,
) -> None:
    """2x3 figure: Qwen3-VL accuracy vs ICL shots on the *complex* split.

    Row 1: Overall, Color, Shape
    Row 2: Counting, Direction, Relation

    Overall aggregates across the 5 visual_reasoning tasks (complex split only).
    Per-task panels read each task's complex leaf directly.
    """
    panels: list[tuple[str, Optional[str]]] = [
        ("Overall", None),
        ("Color", "color_identification"),
        ("Shape", "shape_identification"),
        ("Counting", "counting"),
        ("Direction", "direction"),
        ("Relation", "relation"),
    ]

    per_panel: dict[str, dict[str, dict[int, tuple[int, int]]]] = {
        title: {} for title, _ in panels
    }

    for task_name, leaf_dict in model_summaries.items():
        for leaf, model_dict in leaf_dict.items():
            if extract_split_from_leaf(leaf) != "complex":
                continue
            for model_name, (_, summaries) in model_dict.items():
                if get_model_family(model_name) != "qwen3vl":
                    continue
                for s in summaries:
                    n_shots = s["n_shots"]
                    n_samples = s["n_samples"]
                    acc = s["accuracy"]["overall"]
                    correct = round(acc * n_samples)

                    targets: list[str] = ["Overall"]
                    for title, task in panels:
                        if task == task_name:
                            targets.append(title)

                    for title in targets:
                        bucket = per_panel[title].setdefault(model_name, {})
                        old_c, old_s = bucket.get(n_shots, (0, 0))
                        bucket[n_shots] = (old_c + correct, old_s + n_samples)

    if not any(per_panel[title] for title, _ in panels):
        print("No qwen3vl data found for complex per-task plot")
        return

    def _size_key(name: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)[Bb]", name)
        return float(m.group(1)) if m else float("inf")

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes_flat = axes.flatten()

    for idx, (title, _) in enumerate(panels):
        ax = axes_flat[idx]
        per_model = per_panel[title]
        if not per_model:
            ax.set_visible(False)
            continue

        summaries_fmt = to_summaries_format(per_model)
        for name in sorted(summaries_fmt.keys(), key=_size_key):
            sums = summaries_fmt[name]
            shots = [s["n_shots"] for s in sums]
            acc = [s["accuracy"]["overall"] for s in sums]
            display_name = name.replace("-Instruct", "")
            ax.plot(shots, acc, "o-", label=display_name, color=get_model_color(name))

        ax.set_title(title, fontweight="bold", fontsize=24)
        if idx >= 3:
            ax.set_xlabel("Number of ICL shots", fontsize=22)
        if idx % 3 == 0:
            ax.set_ylabel("Accuracy", fontsize=22)
        ax.tick_params(axis="both", labelsize=18)

        all_shots = sorted({s["n_shots"] for sums in summaries_fmt.values() for s in sums})
        if all_shots:
            ax.set_xticks(all_shots)
        ax.grid(True, alpha=0.3)

    handles_map: dict[str, Any] = {}
    for ax in axes_flat:
        if not ax.get_visible():
            continue
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles_map.setdefault(l, h)

    if handles_map:
        labels = sorted(handles_map.keys(), key=_size_key)
        handles = [handles_map[l] for l in labels]
        fig.legend(
            handles, labels,
            loc="lower center",
            ncol=len(handles),
            fontsize=20,
            frameon=True,
            edgecolor="0.8",
        )

    fig.tight_layout(rect=[0, 0.08, 1, 1.0])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "qwen3vl_complex.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved qwen3vl complex per-task plot to: {out_path}")


def write_qwen3vl_combined_stats(
    aggregated: dict[str, dict[str, dict[int, tuple[int, int]]]],
    output_dir: Path,
) -> None:
    """Write the per-(model, split, n_shots) stats shown in qwen3vl_combined.png.

    Mirrors the data plotted by ``plot_qwen3vl_combined``: for each Qwen3-VL
    model (sorted by size) emits a table with one row per n_shots and columns
    for Overall / Complex / Simple / No-Prior accuracy + correct/total counts.
    """
    groups = ["overall", "complex", "simple", "noprior"]
    titles = ["Overall", "Complex", "Simple", "No-Prior"]

    qwen3vl_models: set[str] = set()
    for group in groups:
        for name in aggregated.get(group, {}):
            if get_model_family(name) == "qwen3vl":
                qwen3vl_models.add(name)
    if not qwen3vl_models:
        print("No qwen3vl data for combined stats")
        return

    def _size_key(name: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)[Bb]", name)
        return float(m.group(1)) if m else float("inf")

    lines: list[str] = []
    width = 138
    for model_name in sorted(qwen3vl_models, key=_size_key):
        display = model_name.replace("-Instruct", "")
        m = re.search(r"(\d+(?:\.\d+)?)[Bb]", model_name)
        size_label = f"{m.group(1)}B" if m else "?"
        header = f"Qwen3-VL {size_label}  ({display})"

        all_shots = sorted({
            n
            for group in groups
            for n in aggregated.get(group, {}).get(model_name, {}).keys()
        })

        col_header = (
            f"{'n_shots':>8} "
            f"{'Overall_acc':>14} {'Overall_c/t':>16} "
            f"{'Complex_acc':>14} {'Complex_c/t':>16} "
            f"{'Simple_acc':>14} {'Simple_c/t':>16} "
            f"{'NoPrior_acc':>14} {'NoPrior_c/t':>16}"
        )
        lines.append(header)
        lines.append("-" * width)
        lines.append(col_header)
        lines.append("-" * width)

        for n in all_shots:
            row = [f"{n:>8}"]
            for group in groups:
                cs = aggregated.get(group, {}).get(model_name, {}).get(n)
                if not cs or cs[1] == 0:
                    row.append(f"{'-':>14}")
                    row.append(f"{'-':>16}")
                else:
                    correct, total = cs
                    acc = correct / total
                    row.append(f"{acc:>14.4f}")
                    row.append(f"{f'{correct}/{total}':>16}")
            lines.append(" ".join(row))
        lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "qwen3vl_combined.txt"
    out_path.write_text("\n".join(lines))
    print(f"Saved qwen3vl combined stats to: {out_path}")


def plot_overall_combined(
    aggregated: dict[str, dict[str, dict[int, tuple[int, int]]]],
    output_dir: Path,
    *,
    dpi: int = 150,
) -> None:
    """5x4 grid: rows = [all families, qwen25vl, qwen35vl, internvl35, gemma4],
    cols = [Overall, Complex, Simple, No-Prior]. Qwen3-VL restricted to 4B/32B in row 1.
    """
    groups = ["overall", "complex", "simple", "noprior"]
    titles = ["Overall", "Complex", "Simple", "No-Prior"]
    rows: list[tuple[str, Any]] = [
        ("All Models", lambda n: _qwen3vl_filter(n, get_model_family(n))),
        ("Qwen3-VL",  lambda n: get_model_family(n) == "qwen3vl" and _qwen3vl_filter(n, "qwen3vl")),
        ("Qwen2.5-VL", lambda n: get_model_family(n) == "qwen25vl"),
        ("Qwen3.5", lambda n: get_model_family(n) == "qwen35vl"),
        ("InternVL3.5", lambda n: get_model_family(n) == "internvl35"),
        ("Gemma4",     lambda n: get_model_family(n) == "gemma4"),
    ]

    def _size_key(name: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)[Bb]", name)
        return float(m.group(1)) if m else float("inf")

    fig, axes = plt.subplots(len(rows), len(groups), figsize=(5 * len(groups), 3.5 * len(rows)))

    for r, (row_label, model_filter) in enumerate(rows):
        for c, (group, title) in enumerate(zip(groups, titles)):
            ax = axes[r, c]
            data = aggregated.get(group, {})
            filtered = {n: d for n, d in data.items() if model_filter(n)}
            if not filtered:
                ax.set_visible(False)
                continue
            summaries_fmt = to_summaries_format(filtered)
            for name in sorted(summaries_fmt.keys(), key=_size_key):
                sums = summaries_fmt[name]
                shots = [s["n_shots"] for s in sums]
                acc = [s["accuracy"]["overall"] for s in sums]
                ax.plot(shots, acc, "o-", label=name, color=get_model_color(name), markersize=4)
            if r == 0:
                ax.set_title(title, fontweight="bold")
            if r == len(rows) - 1:
                ax.set_xlabel("Number of ICL shots")
            if c == 0:
                ax.set_ylabel(f"{row_label}\nAccuracy")
            all_shots = sorted({s["n_shots"] for sums in summaries_fmt.values() for s in sums})
            if all_shots:
                ax.set_xticks(all_shots)
            ax.grid(True, alpha=0.3)

    handles_map: dict[str, Any] = {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles_map.setdefault(l, h)

    fig.suptitle("ICL Scaling Across Splits and Model Families", fontweight="bold", fontsize=28)
    fig.tight_layout(rect=[0, 0.14, 1, 0.95])
    _add_family_grid_legend(fig, handles_map, fontsize=18, bottom=0.05)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "overall_combined.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overall combined plot to: {output_dir / 'overall_combined.png'}")


def plot_progress_qwen(
    aggregated: dict[str, dict[str, dict[int, tuple[int, int]]]],
    output_dir: Path,
    *,
    split: str = "overall",
    dpi: int = 150,
) -> None:
    """1x4 grid of accuracy vs shots across Qwen groupings for a given split."""
    data = aggregated.get(split, {})
    if not data:
        print(f"No {split} data for progress_qwen plot")
        return

    def _size_key(name: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)[Bb]", name)
        return float(m.group(1)) if m else float("inf")

    qwen_families = {"qwen25vl", "qwen3vl", "qwen35vl"}
    col_filters: list[tuple[str, Callable[[str], bool]]] = [
        ("All Qwen",   lambda n: get_model_family(n) in qwen_families and _qwen3vl_filter(n, get_model_family(n))),
        ("Qwen2.5-VL", lambda n: get_model_family(n) == "qwen25vl"),
        ("Qwen3-VL",   lambda n: get_model_family(n) == "qwen3vl" and _qwen3vl_filter(n, "qwen3vl")),
        ("Qwen3.5", lambda n: get_model_family(n) == "qwen35vl"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24, 4.5))
    for ax, (title, keep) in zip(axes, col_filters):
        filtered = {n: d for n, d in data.items() if keep(n)}
        if not filtered:
            ax.set_visible(False)
            continue
        sums_fmt = to_summaries_format(filtered)
        for name in sorted(sums_fmt.keys(), key=_size_key):
            sums = sums_fmt[name]
            shots = [s["n_shots"] for s in sums]
            acc = [s["accuracy"]["overall"] for s in sums]
            ax.plot(shots, acc, "o-", label=name, color=get_model_color(name))
        ax.set_title(title, fontweight="bold", fontsize=24)
        ax.set_xlabel("Number of ICL shots", fontsize=22)
        ax.tick_params(axis="both", labelsize=18)
        all_shots = sorted({s["n_shots"] for sums in sums_fmt.values() for s in sums})
        if all_shots:
            ax.set_xticks(all_shots)
        ax.grid(True, alpha=0.3)

    ylabel = "Overall Accuracy" if split == "overall" else f"{split.capitalize()} Accuracy"
    axes[0].set_ylabel(ylabel, fontsize=22)

    handles_map: dict[str, Any] = {}
    for ax in axes:
        if not ax.get_visible():
            continue
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles_map.setdefault(l, h)

    fig.tight_layout(rect=[0, 0.35, 1, 1.0])
    _add_family_grid_legend(fig, handles_map, fontsize=20, bottom=0.18, bbox_to_anchor=(0.5, -0.18))
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "progress_qwen.png" if split == "overall" else f"progress_qwen_{split}.png"
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved progress_qwen ({split}) plot to: {out_path}")


def write_overall_table(
    aggregated: dict[str, dict[str, dict[int, tuple[int, int]]]],
    output_dir: Path,
    *,
    shots: tuple[int, int, int] = (0, 8, 16),
) -> None:
    """Write per-model overall accuracies at selected shot counts + delta."""
    import csv

    overall = aggregated.get("overall", {})
    if not overall:
        print("No overall data for overall_table")
        return

    def _size_key(name: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)[Bb]", name)
        return float(m.group(1)) if m else float("inf")

    family_sort = {f: i for i, f in enumerate(
        ["qwen3vl", "qwen25vl", "qwen35vl", "internvl35", "gemma4"]
    )}

    def _row_key(name: str) -> tuple[int, float, str]:
        return (family_sort.get(get_model_family(name), 99), _size_key(name), name)

    models = [n for n in overall if _qwen3vl_filter(n, get_model_family(n))]
    models.sort(key=_row_key)

    s0, s8, s16 = shots
    header = ["Model", f"{s0}-shot", f"{s8}-shot", f"{s16}-shot", f"Δ({s16}-{s0})"]

    def _acc(name: str, n: int) -> float | None:
        cs = overall[name].get(n)
        if not cs or cs[1] == 0:
            return None
        return cs[0] / cs[1]

    def _fmt(v: float | None) -> str:
        return "-" if v is None else f"{v:.3f}"

    rows: list[list[str]] = []
    for name in models:
        a0, a8, a16 = _acc(name, s0), _acc(name, s8), _acc(name, s16)
        delta = (a16 - a0) if (a0 is not None and a16 is not None) else None
        rows.append([name, _fmt(a0), _fmt(a8), _fmt(a16), _fmt(delta)])

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "overall_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    md = ["| " + " | ".join(header) + " |",
          "|" + "|".join("---" for _ in header) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (output_dir / "overall_table.md").write_text("\n".join(md) + "\n")
    print(f"Saved overall table to: {output_dir / 'overall_table.csv'}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize ICL scaling aggregated across all tasks"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Root dir containing task/model/config/shots structure "
        "(default: <results_dir>/formal_icl_scale/visual_reasoning)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="output",
        help="Root of the experiment results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_vis/formal_icl_scale",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)
    if args.base_dir is None:
        args.base_dir = str(Path(args.results_dir) / "formal_icl_scale" / "visual_reasoning")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    setup_style()

    plt.rcParams.update({
        "axes.titlesize": 22,
        "axes.titleweight": "bold",
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "figure.titlesize": 26,
        "figure.titleweight": "bold",
        "lines.markersize": 8,
    })

    # 1. Discover all data
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Base directory not found: {base_dir}")
        return

    model_summaries = discover_model_summaries(base_dir)
    if not model_summaries:
        print(f"No data found in {base_dir}")
        return

    # 2. Aggregate across tasks
    splits = ["simple", "noprior", "complex"]
    aggregated = aggregate_across_tasks(model_summaries, splits)
    aggregated_var = aggregate_across_tasks_with_variance(model_summaries, splits)

    # 3. Generate plots for each group
    output_base = Path(args.output_dir)
    groups = ["overall", "simple", "noprior", "complex"]

    for group in groups:
        if group not in aggregated or not aggregated[group]:
            print(f"Skipping {group}: no data")
            continue

        group_dir = output_base / group
        model_summaries_fmt = to_summaries_format_with_variance(
            aggregated_var.get(group, {})
        )

        # Cross-model plot (filter Qwen3-VL to 4B and 32B only)
        plot_accuracy_cross_model(
            model_summaries_fmt,
            group_dir,
            split=group.capitalize(),
            model_filter=_qwen3vl_filter,
            legend_fontsize=16,
            dpi=args.dpi,
        )

        # Per-family plots
        plot_accuracy_per_family(
            model_summaries_fmt,
            group_dir,
            split=group.capitalize(),
            use_family_legend=False,
            legend_fontsize=16,
            dpi=args.dpi,
        )

        print(f"Saved {group} plots to: {group_dir}")

    # 4. Generate combined 2x2 subplot for qwen3vl
    plot_qwen3vl_combined(
        aggregated,
        output_base / "qwen3vl",
        dpi=args.dpi,
    )
    write_qwen3vl_combined_stats(aggregated, output_base / "qwen3vl")

    # 4b. Per-task complex-split plot for qwen3vl
    plot_qwen3vl_complex_per_task(
        model_summaries,
        output_base / "qwen3vl",
        dpi=args.dpi,
    )

    # 5. Generate combined 5x4 grid (all families + per-family rows)
    plot_overall_combined(
        aggregated,
        output_base / "overall",
        dpi=args.dpi,
    )

    # 6. Qwen progress 1x4 plot
    plot_progress_qwen(
        aggregated,
        output_base / "qwen3vl",
        split="overall",
        dpi=args.dpi,
    )
    plot_progress_qwen(
        aggregated,
        output_base / "qwen3vl",
        split="complex",
        dpi=args.dpi,
    )

    # 7. Overall accuracy table
    write_overall_table(aggregated, output_base / "overall")


if __name__ == "__main__":
    main()


# --- Sample running commands ---
# Default (aggregate all tasks):
#   uv run python -m figures.fig_icl_scale
#
# Custom paths:
#   uv run python -m figures.fig_icl_scale \
#       --base_dir output/formal_icl_scale/visual_reasoning \
#       --output_dir output_vis/formal_icl_scale
#
# Higher DPI for publication:
#   uv run python -m figures.fig_icl_scale --dpi 300
