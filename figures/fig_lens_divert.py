"""Visualization for formal_lens_divert: per-group mean ± std of ICL − 0-shot delta."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figures._plotting.style import (
    setup_style,
    finish_figure,
)


BUCKETS = ("prior_cues", "grounding_cues", "others")


def load_records(input_dir: Path) -> List[Dict]:
    """Read `results.jsonl` line-by-line into a list of dicts."""
    path = input_dir / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing results.jsonl in {input_dir}")
    records: List[Dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_group_stats(records: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Aggregate per-group deltas into mean / variance / std / n (population, ddof=0)."""
    stats: Dict[str, Dict[str, float]] = {}
    for g in BUCKETS:
        vals = np.array(
            [r["delta_icl_minus_zeroshot"][g] for r in records if g in r.get("delta_icl_minus_zeroshot", {})],
            dtype=float,
        )
        n = int(vals.size)
        if n > 0:
            mean = float(vals.mean())
            variance = float(np.var(vals, ddof=0))
            std = float(np.std(vals, ddof=0))
        else:
            mean = 0.0
            variance = 0.0
            std = 0.0
        stats[g] = {"mean": mean, "variance": variance, "std": std, "n": n}
    return stats


def compute_cross_task_stats(
    per_task_stats: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate across tasks: mean & variance of per-task means (population, ddof=0).

    Each task contributes one observation (its per-group mean) to the aggregate.
    `n` is the number of contributing tasks.
    """
    out: Dict[str, Dict[str, float]] = {}
    for g in BUCKETS:
        task_means = np.array(
            [s[g]["mean"] for s in per_task_stats.values() if s[g]["n"] > 0],
            dtype=float,
        )
        n = int(task_means.size)
        if n > 0:
            mean = float(task_means.mean())
            variance = float(np.var(task_means, ddof=0))
            std = float(np.std(task_means, ddof=0))
        else:
            mean, variance, std = 0.0, 0.0, 0.0
        out[g] = {"mean": mean, "variance": variance, "std": std, "n": n}
    return out


def _write_group_text_table(stats: Dict[str, Dict[str, float]], path: Path) -> str:
    """Write a 4-column text table (group, mean, variance, n) and return its text."""
    header = ["group", "mean", "variance", "n"]
    rows = [
        [g,
         f"{stats[g]['mean']:+.6f}",
         f"{stats[g]['variance']:.6e}",
         str(stats[g]["n"])]
        for g in BUCKETS
    ]
    col_widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    lines = [fmt.format(*header), "-" * (sum(col_widths) + 2 * (len(col_widths) - 1))]
    for r in rows:
        lines.append(fmt.format(*r))
    table_text = "\n".join(lines) + "\n"
    path.write_text(table_text)
    return table_text


def plot_normalized_delta(
    stats: Dict[str, Dict[str, float]],
    output_dir: Path,
    dpi: int = 150,
    title: Optional[str] = None,
    fname: str = "normalized_delta",
) -> None:
    """Horizontal bar chart: groups stacked vertically; bars extend left (negative)
    or right (positive) from x=0, with ±std error bars."""
    means = [stats[g]["mean"] for g in BUCKETS]
    stds = [stats[g]["std"] for g in BUCKETS]

    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(BUCKETS))
    colors = ["tab:red" if m > 0 else "tab:blue" for m in means]
    ax.barh(y, means, xerr=stds, capsize=5, color=colors, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(BUCKETS)
    ax.invert_yaxis()  # first bucket on top
    ax.set_xlabel("delta mean attn per token (ICL − 0-shot)")
    ax.set_title(title or "Normalized per-token attention difference (ICL − 0-shot)")
    ax.grid(axis="x")
    finish_figure(fig, ax)
    fig.savefig(output_dir / f"{fname}.png", dpi=dpi)
    plt.close(fig)

    table_text = _write_group_text_table(stats, output_dir / f"{fname}.txt")
    print(table_text)


def plot_per_task_breakdown(
    per_task_stats: Dict[str, Dict[str, Dict[str, float]]],
    cross_task: Dict[str, Dict[str, float]],
    output_dir: Path,
    dpi: int = 150,
    title_suffix: str = "",
) -> None:
    """One subplot per group; rows = tasks + an 'ALL' row at top with cross-task stats."""
    tasks = sorted(per_task_stats.keys())
    if not tasks:
        return
    row_labels = ["ALL"] + tasks
    fig_height = max(0.5 * len(row_labels) + 2.0, 3.5)
    fig, axes = plt.subplots(
        1, len(BUCKETS), figsize=(4.5 * len(BUCKETS), fig_height), sharey=True,
    )
    axes = np.atleast_1d(axes)
    y = np.arange(len(row_labels))
    for ax, g in zip(axes, BUCKETS):
        means = [cross_task[g]["mean"]] + [per_task_stats[t][g]["mean"] for t in tasks]
        stds = [cross_task[g]["std"]] + [per_task_stats[t][g]["std"] for t in tasks]
        colors = ["tab:red" if m > 0 else "tab:blue" for m in means]
        ax.barh(y, means, xerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(row_labels)
        ax.invert_yaxis()
        ticklabels = ax.get_yticklabels()
        if ticklabels:
            ticklabels[0].set_fontweight("bold")
        ax.set_xlabel("delta (mean ± std)")
        ax.set_title(g)
        ax.grid(axis="x")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3), useMathText=True)
        ax.tick_params(axis="x", labelsize=8)
    suptitle = "Per-task breakdown of ICL − 0-shot delta"
    if title_suffix:
        suptitle += f" — {title_suffix}"
    fig.suptitle(suptitle)
    finish_figure(fig, axes)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / "per_task_breakdown.png", dpi=dpi)
    plt.close(fig)


def write_meta_table(
    per_task_stats: Dict[str, Dict[str, Dict[str, float]]],
    cross_task: Dict[str, Dict[str, float]],
    output_dir: Path,
) -> None:
    """Write meta_table.txt and meta_table.csv combining per-task rows and an ALL row.

    `n` is entry count for per-task rows and task count for the ALL row.
    """
    header = ["task", "group", "mean", "variance", "std", "n"]
    rows: List[List[str]] = []
    for task in sorted(per_task_stats.keys()):
        for g in BUCKETS:
            s = per_task_stats[task][g]
            rows.append([
                task, g,
                f"{s['mean']:+.6f}",
                f"{s['variance']:.6e}",
                f"{s['std']:.6e}",
                str(s["n"]),
            ])
    for g in BUCKETS:
        s = cross_task[g]
        rows.append([
            "ALL", g,
            f"{s['mean']:+.6f}",
            f"{s['variance']:.6e}",
            f"{s['std']:.6e}",
            str(s["n"]),
        ])

    col_widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "-" * (sum(col_widths) + 2 * (len(col_widths) - 1))
    lines = [fmt.format(*header), sep]
    last_task: Optional[str] = None
    for r in rows:
        if last_task is not None and r[0] != last_task:
            lines.append(sep)
        lines.append(fmt.format(*r))
        last_task = r[0]
    text = "\n".join(lines) + "\n"
    (output_dir / "meta_table.txt").write_text(text)

    with open(output_dir / "meta_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(text)


def process_single(input_dir: Path, output_dir: Path, dpi: int) -> Optional[List[Dict]]:
    """Load per-entry results and generate the normalized-delta plot + table."""
    try:
        records = load_records(input_dir)
    except FileNotFoundError as e:
        print(e)
        return None
    if not records:
        print(f"No records in {input_dir}/results.jsonl")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_group_stats(records)
    plot_normalized_delta(stats, output_dir, dpi)
    print(f"Saved plot + table to: {output_dir}")
    return records


def collect_records_by_model_leaf(
    base: Path,
) -> Dict[Tuple[str, str], Dict[str, List[Dict]]]:
    """Walk `task/model/leaf` under `base` and group records by (model, leaf) → {task: records}."""
    grouped: Dict[Tuple[str, str], Dict[str, List[Dict]]] = defaultdict(dict)
    for task_dir in sorted(base.iterdir()):
        if not task_dir.is_dir():
            continue
        for model_dir in sorted(task_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for leaf_dir in sorted(model_dir.iterdir()):
                if not leaf_dir.is_dir():
                    continue
                if not (leaf_dir / "results.jsonl").exists():
                    continue
                try:
                    records = load_records(leaf_dir)
                except FileNotFoundError:
                    continue
                if records:
                    grouped[(model_dir.name, leaf_dir.name)][task_dir.name] = records
    return grouped


def process_aggregated(
    model_name: str,
    leaf_name: str,
    task_records: Dict[str, List[Dict]],
    output_dir: Path,
    dpi: int,
) -> None:
    """Generate cross-task aggregated plots and a meta table for one (model, leaf)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    per_task_stats = {t: compute_group_stats(recs) for t, recs in task_records.items()}
    cross_task = compute_cross_task_stats(per_task_stats)

    title_suffix = f"{model_name} / {leaf_name} (n_tasks={len(task_records)})"
    plot_normalized_delta(
        cross_task, output_dir, dpi,
        title=f"Aggregated ICL − 0-shot across {len(task_records)} tasks\n{model_name} / {leaf_name}",
        fname="aggregated_normalized_delta",
    )
    plot_per_task_breakdown(per_task_stats, cross_task, output_dir, dpi, title_suffix=title_suffix)
    write_meta_table(per_task_stats, cross_task, output_dir)
    print(f"Aggregated to: {output_dir}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize formal_lens_divert results")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Path to a single leaf dir with results.jsonl")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Root dir to auto-discover all tasks/models "
                             "(default: <results_dir>/formal_lens_divert/visual_reasoning_lens)")
    parser.add_argument("--results_dir", type=str, default="output",
                        help="Root of the experiment results")
    parser.add_argument("--output_dir", type=str, default="output_vis/formal_lens_divert")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)
    if not args.input_dir and not args.base_dir:
        args.base_dir = str(Path(args.results_dir) / "formal_lens_divert" / "visual_reasoning_lens")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    setup_style()

    if args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir) / input_dir.parent.name / input_dir.name
        process_single(input_dir, output_dir, args.dpi)

    elif args.base_dir:
        base = Path(args.base_dir)
        leaf_dirs = []
        for task_dir in sorted(base.iterdir()):
            if not task_dir.is_dir():
                continue
            for model_dir in sorted(task_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                for leaf_dir in sorted(model_dir.iterdir()):
                    if not leaf_dir.is_dir():
                        continue
                    if (leaf_dir / "results.jsonl").exists():
                        leaf_dirs.append((task_dir.name, model_dir.name, leaf_dir))
        for task_name, model_name, leaf_dir in tqdm(leaf_dirs, desc="Per-leaf"):
            out = Path(args.output_dir) / task_name / model_name / leaf_dir.name
            process_single(leaf_dir, out, args.dpi)

        grouped = collect_records_by_model_leaf(base)
        for (model_name, leaf_name), task_records in tqdm(grouped.items(), desc="Aggregating"):
            agg_dir = Path(args.output_dir) / "_aggregated" / model_name / leaf_name
            process_aggregated(model_name, leaf_name, task_records, agg_dir, args.dpi)
    else:
        print("Provide either --input_dir or --base_dir")


if __name__ == "__main__":
    main()

# --- Sample running commands ---
# Auto-discover all tasks/models (per-leaf plots + cross-task aggregation):
#   uv run python -m figures.fig_lens_divert --base_dir output/formal_lens_divert/visual_reasoning_lens
#
# Single task/model/leaf — writes normalized_delta.png + normalized_delta.txt:
#   uv run python -m figures.fig_lens_divert --input_dir output/formal_lens_divert/visual_reasoning_lens/counting/Qwen3-VL-4B-Instruct/npt10
