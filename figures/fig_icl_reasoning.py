"""Visualization for formal_icl_reasoning: accuracy plots (standalone + comparison with normal baseline)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figures._plotting.io import (
    attach_variance_to_summaries,
    load_summaries,
    discover_model_summaries,
    aggregate_across_tasks_with_variance,
    to_summaries_format_with_variance,
)
from figures._plotting.plots import (
    plot_accuracy_per_family,
    plot_accuracy_cross_model,
)
from figures._plotting.style import (
    setup_style,
    finish_figure,
)
from figures._plotting.models import (
    get_model_color,
    get_model_display_name,
)

_TITLE_PREFIX = "Accuracy vs. ICL Shots (Thinking)"


# ---------------------------------------------------------------------------
# Table writers
# ---------------------------------------------------------------------------

def _fmt_acc(acc: Optional[float], yerr: Optional[float] = None, width: int = 14) -> str:
    if acc is None:
        return f"{'n/a':>{width}}"
    if yerr is None:
        return f"{acc:>{width}.4f}"
    return f"{acc:.4f}±{yerr:.4f}".rjust(width)


def _accuracy_keys(summaries: List[Dict]) -> List[str]:
    keys: List[str] = []
    for s in summaries:
        for k in (s.get("accuracy") or {}).keys():
            if k not in keys:
                keys.append(k)
    return keys or ["overall"]


def write_leaf_table(
    summaries: List[Dict],
    baseline_summaries: Optional[List[Dict]],
    output_dir: Path,
    *,
    model_name: str,
    task: Optional[str] = None,
    leaf: Optional[str] = None,
) -> None:
    """Write table.txt for a single leaf: thinking accuracy ± yerr per shot,
    plus optional normal-baseline comparison (with delta)."""
    keys = _accuracy_keys(summaries + (baseline_summaries or []))
    shots = sorted({s["n_shots"] for s in summaries} | {s["n_shots"] for s in (baseline_summaries or [])})

    t_by_shot = {s["n_shots"]: s for s in summaries}
    n_by_shot = {s["n_shots"]: s for s in (baseline_summaries or [])}

    lines: List[str] = []
    lines.append(f"Accuracy vs ICL shots — {model_name}")
    if task is not None or leaf is not None:
        lines.append(f"task={task or '-'}  leaf={leaf or '-'}")
    lines.append("=" * 80)

    for key in keys:
        lines.append("")
        lines.append(f"### accuracy[{key}]")
        if baseline_summaries:
            hdr = f"{'n_shots':<10}  {'thinking':>14}  {'normal':>14}  {'delta':>10}"
        else:
            hdr = f"{'n_shots':<10}  {'thinking':>14}"
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for n in shots:
            ts = t_by_shot.get(n)
            t_acc = (ts.get("accuracy") or {}).get(key) if ts else None
            t_yerr = (ts.get("accuracy_yerr") or {}).get(key) if ts else None
            row = f"{n:<10}  {_fmt_acc(t_acc, t_yerr)}"
            if baseline_summaries:
                ns = n_by_shot.get(n)
                n_acc = (ns.get("accuracy") or {}).get(key) if ns else None
                n_yerr = (ns.get("accuracy_yerr") or {}).get(key) if ns else None
                delta = (t_acc - n_acc) if (t_acc is not None and n_acc is not None) else None
                delta_s = "n/a".rjust(10) if delta is None else f"{delta:>+10.4f}"
                row += f"  {_fmt_acc(n_acc, n_yerr)}  {delta_s}"
            lines.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "table.txt").write_text("\n".join(lines) + "\n")


def write_cross_model_table(
    model_summaries: Dict[str, List[Dict]],
    output_dir: Path,
    *,
    title: str,
) -> None:
    """Write table.txt with one column per model, accuracy ± yerr per shot.
    Used for cross-model and aggregated outputs."""
    if not model_summaries:
        return

    models = list(model_summaries.keys())
    keys: List[str] = []
    shots: set[int] = set()
    for sums in model_summaries.values():
        for k in _accuracy_keys(sums):
            if k not in keys:
                keys.append(k)
        shots.update(s["n_shots"] for s in sums)
    shot_list = sorted(shots)

    by_model_shot = {
        m: {s["n_shots"]: s for s in sums} for m, sums in model_summaries.items()
    }

    lines: List[str] = []
    lines.append(title)
    lines.append("=" * 80)

    widths = {m: max(14, len(m)) for m in models}

    for key in keys:
        lines.append("")
        lines.append(f"### accuracy[{key}]")
        hdr = f"{'n_shots':<10}" + "".join(f"  {m:>{widths[m]}}" for m in models)
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for n in shot_list:
            row = f"{n:<10}"
            for m in models:
                s = by_model_shot[m].get(n)
                acc = (s.get("accuracy") or {}).get(key) if s else None
                yerr = (s.get("accuracy_yerr") or {}).get(key) if s else None
                row += f"  {_fmt_acc(acc, yerr, width=widths[m])}"
            lines.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "table.txt").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Standalone accuracy plot
# ---------------------------------------------------------------------------

def plot_accuracy(summaries: List[Dict], output_dir: Path, dpi: int = 150) -> None:
    shots = [s["n_shots"] for s in summaries]
    overall = [s["accuracy"]["overall"] for s in summaries]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(shots, overall, "o-", label="Overall", linewidth=2)
    ax.set_xlabel("Number of ICL shots")
    ax.set_ylabel("Accuracy")
    ax.set_title(_TITLE_PREFIX)
    ax.legend()
    ax.set_xticks(shots)
    ax.grid(True)
    finish_figure(fig, ax)
    fig.savefig(output_dir / "accuracy_vs_shots.png", dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Comparison plot (thinking vs normal)
# ---------------------------------------------------------------------------

def plot_accuracy_comparison(
    thinking_summaries: List[Dict],
    normal_summaries: List[Dict],
    model_name: str,
    output_dir: Path,
    dpi: int = 150,
) -> None:
    """Overlay thinking and normal accuracy curves on one figure."""
    fig, ax = plt.subplots(figsize=(8, 5))

    t_shots = [s["n_shots"] for s in thinking_summaries]
    t_acc = [s["accuracy"]["overall"] for s in thinking_summaries]
    ax.plot(t_shots, t_acc, "o-", label="Thinking", linewidth=2, color=get_model_color(model_name))

    n_shots = [s["n_shots"] for s in normal_summaries]
    n_acc = [s["accuracy"]["overall"] for s in normal_summaries]
    ax.plot(n_shots, n_acc, "s--", label="Normal", linewidth=2, color=get_model_color(model_name), alpha=0.6)

    ax.set_xlabel("Number of ICL shots")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Thinking vs. Normal — {model_name}")
    ax.legend()
    all_shots = sorted(set(t_shots) | set(n_shots))
    if all_shots:
        ax.set_xticks(all_shots)
    ax.grid(True)
    finish_figure(fig, ax)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "accuracy_comparison.png", dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Thinking vs. non-thinking comparison grid (rows=models, cols=splits)
# ---------------------------------------------------------------------------

def plot_thinking_normal_comparison_grid(
    cross_model_thinking: Dict,
    cross_model_normal: Dict,
    output_dir: Path,
    *,
    target_models: Sequence[str],
    target_groups: Sequence[str],
    splits: Sequence[str] = ("simple", "noprior", "complex"),
    dpi: int = 150,
) -> None:
    """4×4 grid: rows=models, cols=splits. Solid line=non-thinking, dotted=thinking.

    Accuracy is aggregated across tasks per split using the standard cross-task
    aggregator, then read off the ``"overall"`` key of the resulting summaries.
    """
    agg_thinking = aggregate_across_tasks_with_variance(cross_model_thinking, list(splits))
    agg_normal = aggregate_across_tasks_with_variance(cross_model_normal, list(splits))

    sums_thinking: Dict[str, Dict[str, List[Dict]]] = {
        g: to_summaries_format_with_variance(agg_thinking.get(g, {})) for g in target_groups
    }
    sums_normal: Dict[str, Dict[str, List[Dict]]] = {
        g: to_summaries_format_with_variance(agg_normal.get(g, {})) for g in target_groups
    }

    group_titles = {
        "overall": "Overall",
        "complex": "Complex",
        "simple": "Simple",
        "noprior": "No-Prior",
    }

    n_rows = len(target_models)
    n_cols = len(target_groups)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.4 * n_rows),
        sharex=True, sharey=False,
        squeeze=False,
    )

    title_fontsize = 26
    axis_label_fontsize = 16
    tick_fontsize = 13
    legend_fontsize = 22

    for r, model in enumerate(target_models):
        color = get_model_color(model)
        for c, group in enumerate(target_groups):
            ax = axes[r][c]
            n_sums = sums_normal.get(group, {}).get(model) or []
            t_sums = sums_thinking.get(group, {}).get(model) or []

            if n_sums:
                ordered = sorted(n_sums, key=lambda s: s["n_shots"])
                xs = [s["n_shots"] for s in ordered]
                ys = [s["accuracy"].get("overall", 0.0) for s in ordered]
                ax.plot(xs, ys, marker="o", linestyle="-", color=color, linewidth=2)
            if t_sums:
                ordered = sorted(t_sums, key=lambda s: s["n_shots"])
                xs = [s["n_shots"] for s in ordered]
                ys = [s["accuracy"].get("overall", 0.0) for s in ordered]
                ax.plot(xs, ys, marker="s", linestyle=":", color=color, linewidth=2)

            if r == 0:
                ax.set_title(group_titles.get(group, group.capitalize()), fontsize=title_fontsize, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{get_model_display_name(model)}\nAccuracy", fontsize=axis_label_fontsize)
            if r == n_rows - 1:
                ax.set_xlabel("Number of ICL shots", fontsize=axis_label_fontsize)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            ax.grid(True, alpha=0.3)

            shots = sorted({s["n_shots"] for s in n_sums} | {s["n_shots"] for s in t_sums})
            if shots:
                ax.set_xticks(shots)

    legend_handles = [
        plt.Line2D([0], [0], color="black", marker="o", linestyle="-", linewidth=3, markersize=10),
        plt.Line2D([0], [0], color="black", marker="s", linestyle=":", linewidth=3, markersize=10),
    ]
    fig.legend(
        legend_handles, ["Non-thinking", "Thinking"],
        loc="lower center", ncol=2, frameon=True, edgecolor="0.8",
        facecolor="white", framealpha=0.9,
        fontsize=legend_fontsize, handlelength=3.0, handletextpad=0.6, columnspacing=2.0,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "thinking_vs_normal_grid.png", dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Compare row (single row across target models, thinking vs non-thinking)
# ---------------------------------------------------------------------------

def plot_compare_row(
    cross_model_thinking: Dict,
    cross_model_normal: Dict,
    output_dir: Path,
    *,
    target_models: Sequence[str],
    group: str,
    splits: Sequence[str],
    filename: str,
    dpi: int = 150,
) -> Dict[str, Dict[str, List[Dict]]]:
    """One row of subplots (one per model). Solid=non-thinking, dotted=thinking.

    Returns ``{"normal": {model: summaries}, "thinking": {model: summaries}}``
    so the table writer can reuse the same aggregated numbers.
    """
    agg_thinking = aggregate_across_tasks_with_variance(cross_model_thinking, list(splits))
    agg_normal = aggregate_across_tasks_with_variance(cross_model_normal, list(splits))

    sums_thinking = to_summaries_format_with_variance(agg_thinking.get(group, {}))
    sums_normal = to_summaries_format_with_variance(agg_normal.get(group, {}))

    n_cols = len(target_models)
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(4.5 * n_cols, 3.6),
        sharex=False, sharey=False,
        squeeze=False,
    )

    title_fontsize = 18
    axis_label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 14

    for c, model in enumerate(target_models):
        ax = axes[0][c]
        color = get_model_color(model)
        n_sums = sums_normal.get(model) or []
        t_sums = sums_thinking.get(model) or []

        if n_sums:
            ordered = sorted(n_sums, key=lambda s: s["n_shots"])
            xs = [s["n_shots"] for s in ordered]
            ys = [s["accuracy"].get("overall", 0.0) for s in ordered]
            ax.plot(xs, ys, marker="o", linestyle="-", color=color, linewidth=2)
        if t_sums:
            ordered = sorted(t_sums, key=lambda s: s["n_shots"])
            xs = [s["n_shots"] for s in ordered]
            ys = [s["accuracy"].get("overall", 0.0) for s in ordered]
            ax.plot(xs, ys, marker="s", linestyle=":", color=color, linewidth=2)

        ax.set_title(get_model_display_name(model), fontsize=title_fontsize, fontweight="bold")
        ax.set_xlabel("Number of ICL shots", fontsize=axis_label_fontsize)
        if c == 0:
            ax.set_ylabel("Accuracy", fontsize=axis_label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.grid(True, alpha=0.3)

        shots = sorted({s["n_shots"] for s in n_sums} | {s["n_shots"] for s in t_sums})
        if shots:
            ax.set_xticks(shots)

    legend_handles = [
        plt.Line2D([0], [0], color="black", marker="o", linestyle="-", linewidth=3, markersize=10),
        plt.Line2D([0], [0], color="black", marker="s", linestyle=":", linewidth=3, markersize=10),
    ]
    fig.legend(
        legend_handles, ["Non-thinking", "Thinking"],
        loc="lower center", ncol=2, frameon=True, edgecolor="0.8",
        facecolor="white", framealpha=0.9,
        fontsize=legend_fontsize, handlelength=3.0, handletextpad=0.6, columnspacing=2.0,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 1])

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=dpi)
    plt.close(fig)

    return {"normal": sums_normal, "thinking": sums_thinking}


def write_compare_row_table(
    sums_normal: Dict[str, List[Dict]],
    sums_thinking: Dict[str, List[Dict]],
    output_dir: Path,
    *,
    target_models: Sequence[str],
    filename: str,
    title: str,
) -> None:
    """Per-model table: n_shots × {non-thinking, thinking, delta} columns."""
    lines: List[str] = []
    lines.append(title)
    lines.append("=" * 80)

    for model in target_models:
        n_sums = sums_normal.get(model) or []
        t_sums = sums_thinking.get(model) or []
        n_by_shot = {s["n_shots"]: s for s in n_sums}
        t_by_shot = {s["n_shots"]: s for s in t_sums}
        shots = sorted(set(n_by_shot.keys()) | set(t_by_shot.keys()))

        lines.append("")
        lines.append(f"### {get_model_display_name(model)}")
        hdr = f"{'n_shots':<10}  {'non-thinking':>14}  {'thinking':>14}  {'delta':>10}"
        lines.append(hdr)
        lines.append("-" * len(hdr))
        if not shots:
            lines.append("(no data)")
            continue
        for n in shots:
            ns = n_by_shot.get(n)
            ts = t_by_shot.get(n)
            n_acc = (ns.get("accuracy") or {}).get("overall") if ns else None
            t_acc = (ts.get("accuracy") or {}).get("overall") if ts else None
            delta = (t_acc - n_acc) if (t_acc is not None and n_acc is not None) else None
            delta_s = "n/a".rjust(10) if delta is None else f"{delta:>+10.4f}"
            lines.append(
                f"{n:<10}  {_fmt_acc(n_acc)}  {_fmt_acc(t_acc)}  {delta_s}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / filename).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_input_dir(input_dir: Path, output_dir: Path, baseline_dir: Optional[Path], dpi: int) -> None:
    summaries = load_summaries(input_dir)
    if not summaries:
        print(f"No summary files found in {input_dir}")
        return
    attach_variance_to_summaries(input_dir, summaries)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy(summaries, output_dir, dpi)
    print(f"Saved plots to: {output_dir}")

    model_name = input_dir.parent.name
    task_name = input_dir.parent.parent.name
    leaf_name = input_dir.name

    baseline_summaries: Optional[List[Dict]] = None
    if baseline_dir is not None and baseline_dir.exists():
        baseline_summaries = load_summaries(baseline_dir)
        if baseline_summaries:
            attach_variance_to_summaries(baseline_dir, baseline_summaries)
            plot_accuracy_comparison(summaries, baseline_summaries, model_name, output_dir, dpi)
            print(f"Saved comparison plot to: {output_dir}")
        else:
            baseline_summaries = None

    write_leaf_table(
        summaries, baseline_summaries, output_dir,
        model_name=model_name, task=task_name, leaf=leaf_name,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize formal_icl_reasoning results")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Path to a single task output dir with shots_*/summary.json files")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Root dir to auto-discover (default: <results_dir>/formal_icl_reasoning/visual_reasoning)")
    parser.add_argument("--baseline_dir", type=str, default=None,
                        help="Root dir for normal baseline (default: <results_dir>/formal_icl_scale/visual_reasoning)")
    parser.add_argument("--results_dir", type=str, default="output",
                        help="Root of the experiment results")
    parser.add_argument("--output_dir", type=str, default="output_vis/formal_icl_reasoning")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)
    results = Path(args.results_dir)
    if not args.input_dir and not args.base_dir:
        args.base_dir = str(results / "formal_icl_reasoning" / "visual_reasoning")
    if not args.baseline_dir:
        args.baseline_dir = str(results / "formal_icl_scale" / "visual_reasoning")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    setup_style()

    if args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir) / input_dir.parent.name / input_dir.name
        baseline = None
        if args.baseline_dir:
            baseline = Path(args.baseline_dir) / input_dir.parent.parent.name / input_dir.parent.name / input_dir.name
        process_input_dir(input_dir, output_dir, baseline, args.dpi)

    elif args.base_dir:
        base = Path(args.base_dir)
        baseline_base = Path(args.baseline_dir) if args.baseline_dir else None
        cross_model = discover_model_summaries(base)

        for leaf_dict in cross_model.values():
            for model_dict in leaf_dict.values():
                for leaf_path, summaries in model_dict.values():
                    attach_variance_to_summaries(leaf_path, summaries)

        # Per-model plots (with optional baseline comparison)
        for task, leaf_dict in cross_model.items():
            for leaf, model_dict in leaf_dict.items():
                for model_name, (leaf_path, summaries) in model_dict.items():
                    out = Path(args.output_dir) / task / model_name / leaf
                    baseline = baseline_base / task / model_name / leaf if baseline_base else None
                    process_input_dir(leaf_path, out, baseline, args.dpi)

        # Cross-model comparison plots
        for task, leaf_dict in cross_model.items():
            for leaf, model_dict in leaf_dict.items():
                if len(model_dict) > 1:
                    sums = {m: s for m, (_, s) in model_dict.items()}
                    out = Path(args.output_dir) / task / leaf
                    plot_accuracy_cross_model(sums, out, task=task, split=leaf, dpi=args.dpi)
                    plot_accuracy_per_family(sums, out, task=task, split=leaf, dpi=args.dpi)
                    write_cross_model_table(
                        sums, out,
                        title=f"Cross-model accuracy (Thinking) — task={task}  leaf={leaf}",
                    )

        # High-level aggregated plots (all tasks combined)
        splits = ["simple", "noprior", "complex"]
        aggregated_var = aggregate_across_tasks_with_variance(cross_model, splits)

        for group in ["overall", "simple", "noprior", "complex"]:
            group_counts = aggregated_var.get(group, {})
            if not group_counts:
                print(f"Skipping aggregated {group}: no data")
                continue
            group_dir = Path(args.output_dir) / "aggregated" / group
            model_summaries_fmt = to_summaries_format_with_variance(group_counts)
            plot_accuracy_cross_model(
                model_summaries_fmt, group_dir,
                split=f"{group.capitalize()} (Thinking)", dpi=args.dpi
            )
            plot_accuracy_per_family(
                model_summaries_fmt, group_dir,
                split=f"{group.capitalize()} (Thinking)",
                use_family_legend=False, dpi=args.dpi
            )
            write_cross_model_table(
                model_summaries_fmt, group_dir,
                title=f"Aggregated accuracy (Thinking) — group={group}",
            )
            print(f"Saved aggregated {group} plots to: {group_dir}")

        # Thinking vs. non-thinking comparison grid (rows=models, cols=splits)
        if baseline_base is not None and baseline_base.exists():
            cross_model_normal = discover_model_summaries(baseline_base)
            target_models = [
                "Qwen3-VL-4B-Instruct",
                "Qwen3-VL-32B-Instruct",
                "gemma-4-E4B-it",
                "gemma-4-31B-it",
            ]
            target_groups = ["overall", "complex", "simple", "noprior"]
            comparison_dir = Path(args.output_dir) / "comparison"
            plot_thinking_normal_comparison_grid(
                cross_model, cross_model_normal, comparison_dir,
                target_models=target_models, target_groups=target_groups,
                splits=splits, dpi=args.dpi,
            )
            print(f"Saved thinking-vs-normal comparison grid to: {comparison_dir}")

            for group, png_name, txt_name, title in (
                ("overall", "compare.png", "compare.txt",
                 "Compare (overall, aggregated across tasks/splits) — non-thinking vs thinking"),
                ("complex", "compare_complex.png", "compare_complex.txt",
                 "Compare (complex split, aggregated across tasks) — non-thinking vs thinking"),
            ):
                sums = plot_compare_row(
                    cross_model, cross_model_normal, comparison_dir,
                    target_models=target_models, group=group,
                    splits=splits, filename=png_name, dpi=args.dpi,
                )
                write_compare_row_table(
                    sums["normal"], sums["thinking"], comparison_dir,
                    target_models=target_models, filename=txt_name, title=title,
                )
                print(f"Saved {png_name} / {txt_name} to: {comparison_dir}")
    else:
        print("Provide either --input_dir or --base_dir")


if __name__ == "__main__":
    main()


# --- Sample running commands ---
# Auto-discover all tasks/models:
# uv run python -m figures.fig_icl_reasoning --base_dir output/formal_icl_reasoning/visual_reasoning --baseline_dir output/formal_icl_scale/visual_reasoning