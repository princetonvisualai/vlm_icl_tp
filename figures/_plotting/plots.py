"""Plot and stats-table helpers for the formal accuracy-vs-shots figures.

Provides reusable plot helpers and an experiment-spec runner used by the
figure entrypoints.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

from .io import (
    LeafFilter,
    ModelFilter,
    ShotAggregate,
    aggregate_across_tasks,
    aggregate_across_tasks_with_variance,
    aggregate_per_task_with_variance,
    attach_variance_to_summaries,
    average_leaf_yerrs,
    discover_model_summaries,
    extract_split_from_leaf,
    has_alignment_keys,
    load_jsonl,
    load_pcorrect_per_sample,
    load_summaries,
    to_summaries_format,
    to_summaries_format_with_variance,
)
from .models import (
    FAMILY_ORDER,
    add_family_legend,
    get_model_color,
    get_model_display_name,
    get_model_family,
)
from .style import finish_figure, setup_style


ShotFilter = Callable[[int], bool]  # n_shots -> keep?


AccuracyBaselineFactory = Callable[
    [list[dict[str, Any]]],
    list["BaselineSpec"],
]


@dataclass(frozen=True)
class AccuracySeriesSpec:
    """Configuration for one accuracy curve in a per-model plot."""

    key: str
    label: str
    style: str = "o-"
    linewidth: float = 2.0
    color: str | None = None
    markersize: float | None = None
    shot_filter: ShotFilter | None = None


@dataclass(frozen=True)
class BaselineSpec:
    """Configuration for an accuracy baseline annotation."""

    kind: str  # "hline" or "point"
    y: float
    label: str | None = None
    x: float | None = None
    color: str = "gray"
    linestyle: str = ":"
    linewidth: float = 1.0
    marker: str = "o"
    markersize: float = 8.0
    zorder: float | None = None


@dataclass(frozen=True)
class ErrorbarConfig:
    """Default-on variance bands on accuracy curves.

    The (task, split) variance unit is the mean over queries of the per-query
    std across the 10 demo sets. Cross-task / cross-split variance is the
    simple average of those leaf units. ``style="bar"`` draws whiskers via
    ``ax.errorbar``; ``"band"`` overlays ``fill_between``; ``"none"`` opts
    out (plain ``ax.plot``).
    """

    enabled: bool = True
    style: str = "band"          # "bar" | "band" | "none"
    capsize: float = 3.0
    band_alpha: float = 0.18
    suppress_at_zero_shot: bool = True


@dataclass(frozen=True)
class ExperimentVisSpec:
    """Declarative configuration for a formal visualization entrypoint."""

    description: str
    output_dir_default: str
    results_subdir: str
    accuracy_title: str = "Accuracy vs. ICL Shots"
    label_probs_title: str = "Per-label Output Probability vs. Shots"
    accuracy_series: tuple[AccuracySeriesSpec, ...] = (
        AccuracySeriesSpec(key="overall", label="Overall"),
    )
    accuracy_baselines_factory: AccuracyBaselineFactory | None = None
    include_label_probs: bool = True
    include_dumbbell: bool = True
    include_violin: bool = True
    include_bar: bool = False
    include_cross_model: bool = True
    include_per_family: bool = True
    per_family_use_family_legend: bool = False
    include_aggregated: bool = False
    include_compare: bool = False
    aggregated_splits: tuple[str, ...] = ("simple", "noprior", "complex")
    leaf_filter: LeafFilter | None = None
    cross_model_filter: ModelFilter | None = None
    errorbar: ErrorbarConfig = field(default_factory=ErrorbarConfig)


def parse_visualization_args(
    argv: list[str] | tuple[str, ...] | None = None,
    *,
    spec: ExperimentVisSpec,
    configure: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.Namespace:
    """Parse the common CLI shared by the figure entrypoints.

    Without ``--input_dir``/``--base_dir``, the results root is
    ``<results_dir>/<spec.results_subdir>``.
    """
    parser = argparse.ArgumentParser(description=spec.description)
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Path to a single task output dir with shots_*/summary.json files",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help=f"Root dir to auto-discover all tasks (default: <results_dir>/{spec.results_subdir})",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="output",
        help="Root of the experiment results",
    )
    parser.add_argument("--output_dir", type=str, default=spec.output_dir_default)
    parser.add_argument("--dpi", type=int, default=150)
    if configure is not None:
        configure(parser)
    args = parser.parse_args(argv)
    if not args.input_dir and not args.base_dir:
        args.base_dir = str(Path(args.results_dir) / spec.results_subdir)
    return args


def plot_curve(
    ax,
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    label: str | None = None,
    color: str | None = None,
    style: str = "o-",
    linewidth: float = 2.0,
    markersize: float | None = None,
    yerr: Sequence[float | None] | None = None,
    errorbar: ErrorbarConfig | None = None,
    zorder: float | None = None,
):
    """Render a single accuracy curve, optionally with whiskers or a band.

    ``yerr`` accepts ``None`` at indices where no whisker should be drawn (e.g.
    ``n_shots == 0`` with only one demo seed); these are converted to ``np.nan``
    so matplotlib drops the whisker at that index. Falls back to ``ax.plot``
    when ``errorbar`` is disabled, missing, or the yerr array is all-nan.

    Returns a legend-handle artist (a Line2D-like object).
    """
    cfg = errorbar if errorbar is not None else ErrorbarConfig()
    has_yerr = yerr is not None and any(v is not None for v in yerr)

    # Plain plot path: error bars off, missing yerr, or style="none"
    if not cfg.enabled or cfg.style == "none" or not has_yerr:
        plot_kwargs: dict[str, Any] = {"label": label, "linewidth": linewidth}
        if color is not None:
            plot_kwargs["color"] = color
        if markersize is not None:
            plot_kwargs["markersize"] = markersize
        if zorder is not None:
            plot_kwargs["zorder"] = zorder
        line, = ax.plot(xs, ys, style, **plot_kwargs)
        return line

    yerr_arr = np.array(
        [np.nan if v is None else float(v) for v in yerr], dtype=float
    )

    if cfg.style == "band":
        plot_kwargs = {"label": label, "linewidth": linewidth}
        if color is not None:
            plot_kwargs["color"] = color
        if markersize is not None:
            plot_kwargs["markersize"] = markersize
        if zorder is not None:
            plot_kwargs["zorder"] = zorder
        line, = ax.plot(xs, ys, style, **plot_kwargs)
        ys_arr = np.asarray(ys, dtype=float)
        # NaN entries in yerr_arr propagate into fill_between, leaving gaps.
        ax.fill_between(
            xs,
            ys_arr - yerr_arr,
            ys_arr + yerr_arr,
            color=color,
            alpha=cfg.band_alpha,
            linewidth=0,
            zorder=(zorder - 0.5) if zorder is not None else 1.5,
        )
        return line

    # Default: errorbar with whiskers
    eb_kwargs: dict[str, Any] = {
        "fmt": style,
        "linewidth": linewidth,
        "capsize": cfg.capsize,
    }
    if label is not None:
        eb_kwargs["label"] = label
    if color is not None:
        eb_kwargs["color"] = color
    if markersize is not None:
        eb_kwargs["markersize"] = markersize
    if zorder is not None:
        eb_kwargs["zorder"] = zorder
    container = ax.errorbar(xs, ys, yerr=yerr_arr, **eb_kwargs)
    return container.lines[0]


def plot_model_accuracy(
    summaries: list[dict[str, Any]],
    output_dir: Path,
    *,
    spec: ExperimentVisSpec,
    dpi: int = 150,
) -> None:
    """Plot the experiment-specific per-model accuracy view."""
    shots = [s["n_shots"] for s in summaries]
    baselines = []
    if spec.accuracy_baselines_factory is not None:
        baselines = spec.accuracy_baselines_factory(summaries)

    fig, ax = plt.subplots(figsize=(8, 5))

    for baseline in baselines:
        if baseline.kind == "hline":
            hline_kwargs: dict[str, Any] = {
                "color": baseline.color,
                "linestyle": baseline.linestyle,
                "linewidth": baseline.linewidth,
                "zorder": baseline.zorder,
            }
            if baseline.label is not None:
                hline_kwargs["label"] = baseline.label
            ax.axhline(
                baseline.y,
                **hline_kwargs,
            )
        elif baseline.kind == "point":
            point_kwargs: dict[str, Any] = {
                "color": baseline.color,
                "markersize": baseline.markersize,
                "zorder": baseline.zorder,
                "linestyle": "None",
            }
            if baseline.label is not None:
                point_kwargs["label"] = baseline.label
            ax.plot(
                baseline.x,
                baseline.y,
                baseline.marker,
                **point_kwargs,
            )
        else:
            raise ValueError(f"Unsupported baseline kind: {baseline.kind}")

    for series in spec.accuracy_series:
        series_summaries = [
            s for s in summaries
            if series.shot_filter is None or series.shot_filter(s["n_shots"])
        ]
        if not series_summaries:
            continue

        series_shots = [s["n_shots"] for s in series_summaries]
        values = [s["accuracy"][series.key] for s in series_summaries]
        yerrs = [
            None if (spec.errorbar.suppress_at_zero_shot and s["n_shots"] == 0)
            else (s.get("accuracy_yerr") or {}).get(series.key)
            for s in series_summaries
        ]
        plot_curve(
            ax,
            series_shots,
            values,
            label=series.label,
            color=series.color,
            style=series.style,
            linewidth=series.linewidth,
            markersize=series.markersize,
            yerr=yerrs,
            errorbar=spec.errorbar,
        )

    ax.set_xlabel("Number of ICL shots")
    ax.set_ylabel("Accuracy")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    ax.set_xticks(shots)
    ax.grid(True)
    finish_figure(fig, ax)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "accuracy_vs_shots.png", dpi=dpi)
    plt.close(fig)


def plot_label_probs(
    summaries: list[dict[str, Any]],
    output_dir: Path,
    *,
    suptitle: str = "Per-label Output Probability vs. Shots",
    dpi: int = 150,
) -> None:
    """Per-label probability subplots with shared legend."""
    label_order = summaries[0]["label_order"]
    gt_labels = sorted(summaries[0]["per_label_mean_distribution"].keys())
    shots = [s["n_shots"] for s in summaries]

    n_gt = len(gt_labels)
    rows, cols = (1, n_gt) if n_gt <= 4 else (2, (n_gt + 1) // 2)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)

    for idx, gt in enumerate(gt_labels):
        ax = axes[idx // cols][idx % cols]
        for li, label in enumerate(label_order):
            probs = [
                (s["per_label_mean_distribution"].get(gt) or [0] * len(label_order))[li]
                for s in summaries
            ]
            ax.plot(shots, probs, "o-", label=f"P({label})", markersize=4)
        ax.set_title(f"GT={gt}")
        ax.set_xlabel("Shots")
        ax.set_ylabel("Mean P(label)")
        ax.set_xticks(shots)
        ax.grid(True)

    for idx in range(n_gt, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(label_order))
    fig.suptitle(suptitle)
    sns.despine(fig=fig)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_dir / "label_probs_vs_shots.png", dpi=dpi)
    plt.close(fig)


def plot_accuracy_per_family(
    model_summaries: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    *,
    task: str = "",
    split: str = "",
    use_family_legend: bool = True,
    legend_fontsize: str | int | None = None,
    errorbar: ErrorbarConfig | None = None,
    dpi: int = 150,
) -> None:
    """One accuracy-vs-shots figure per model family, comparing sizes within."""
    cfg = errorbar or ErrorbarConfig()
    family_models: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for model_name, summaries in model_summaries.items():
        family = get_model_family(model_name)
        family_models.setdefault(family, {})[model_name] = summaries

    ordered_families = [f for f in FAMILY_ORDER if f in family_models]
    ordered_families.extend(f for f in family_models if f not in ordered_families)
    for family in ordered_families:
        models = family_models[family]
        if len(models) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        def _size_key(item: tuple[str, list[dict[str, Any]]]) -> float:
            match = re.search(r"(\d+(?:\.\d+)?)[Bb]", item[0])
            return float(match.group(1)) if match else float("inf")

        for model_name, summaries in sorted(models.items(), key=_size_key):
            shots = [s["n_shots"] for s in summaries]
            overall = [s["accuracy"]["overall"] for s in summaries]
            yerrs = [
                None
                if (cfg.suppress_at_zero_shot and s["n_shots"] == 0)
                else (s.get("accuracy_yerr") or {}).get("overall")
                for s in summaries
            ]
            plot_curve(
                ax,
                shots,
                overall,
                label=model_name,
                color=get_model_color(model_name),
                yerr=yerrs,
                errorbar=cfg,
            )

        ax.set_xlabel("Number of ICL shots", fontsize=16)
        ax.set_ylabel("Accuracy", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        all_shots = sorted({s["n_shots"] for sums in models.values() for s in sums})
        if all_shots:
            ax.set_xticks(all_shots)
        ax.grid(True)
        finish_figure(fig, ax)
        if use_family_legend:
            add_family_legend(fig, ax, fontsize=legend_fontsize or 14)
        else:
            handles, labels = ax.get_legend_handles_labels()
            ncol = min(len(handles), 3)
            n_rows = math.ceil(len(handles) / ncol) if ncol else 1
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=ncol,
                fontsize=legend_fontsize or 14,
            )
            fig.tight_layout(rect=[0, 0.06 + 0.06 * n_rows, 1, 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"accuracy_{family}.png", dpi=dpi)
        plt.close(fig)


def plot_accuracy_cross_model(
    model_summaries: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    *,
    task: str = "",
    split: str = "",
    model_filter: ModelFilter | None = None,
    legend_fontsize: str | int | None = None,
    errorbar: ErrorbarConfig | None = None,
    filename: str = "accuracy_cross_model.png",
    dpi: int = 150,
) -> None:
    """Accuracy vs shots comparing all models on one figure."""
    cfg = errorbar or ErrorbarConfig()
    if model_filter is not None:
        model_summaries = {
            name: sums
            for name, sums in model_summaries.items()
            if model_filter(name, get_model_family(name))
        }

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, summaries in sorted(model_summaries.items()):
        shots = [s["n_shots"] for s in summaries]
        overall = [s["accuracy"]["overall"] for s in summaries]
        yerrs = [
            None
            if (cfg.suppress_at_zero_shot and s["n_shots"] == 0)
            else (s.get("accuracy_yerr") or {}).get("overall")
            for s in summaries
        ]
        plot_curve(
            ax,
            shots,
            overall,
            label=model_name,
            color=get_model_color(model_name),
            yerr=yerrs,
            errorbar=cfg,
        )
    ax.set_xlabel("Number of ICL shots")
    ax.set_ylabel("Accuracy")
    all_shots = sorted({s["n_shots"] for sums in model_summaries.values() for s in sums})
    if all_shots:
        ax.set_xticks(all_shots)
    ax.grid(True)
    finish_figure(fig, ax)
    add_family_legend(fig, ax, fontsize=legend_fontsize)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=dpi)
    plt.close(fig)


def write_cross_model_accuracy_table(
    model_summaries: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    *,
    filename: str = "overall_cross_model.txt",
    title: str = "Cross-model accuracy vs. ICL shots",
    model_filter: ModelFilter | None = None,
    errorbar: ErrorbarConfig | None = None,
) -> None:
    """Write the model x shots accuracy table that backs the cross-model plot.

    Cells are rendered as ``acc ± yerr`` when a yerr is available; the yerr at
    ``n_shots == 0`` is suppressed when ``errorbar.suppress_at_zero_shot`` is
    set, matching the plot. Rows are ordered by model family / size so the
    table reads in the same order as the plot legend.
    """
    cfg = errorbar or ErrorbarConfig()
    if model_filter is not None:
        model_summaries = {
            name: sums
            for name, sums in model_summaries.items()
            if model_filter(name, get_model_family(name))
        }
    if not model_summaries:
        return

    all_shots = sorted(
        {s["n_shots"] for sums in model_summaries.values() for s in sums}
    )
    ordered_models = sorted(model_summaries.keys(), key=_model_family_size_key)

    name_w = max(len("Model"), *(len(m) for m in ordered_models))
    cell_w = 16
    header = f"{'Model':<{name_w}}" + "".join(
        f"  {f'{n}-shot':>{cell_w}}" for n in all_shots
    )

    lines: list[str] = []
    lines.append(title)
    lines.append("=" * len(header))
    lines.append("")
    lines.append(header)
    lines.append("-" * len(header))
    for model_name in ordered_models:
        shot_to_s = {s["n_shots"]: s for s in model_summaries[model_name]}
        row = f"{model_name:<{name_w}}"
        for n in all_shots:
            s = shot_to_s.get(n)
            if s is None:
                row += f"  {'n/a':>{cell_w}}"
                continue
            acc = (s.get("accuracy") or {}).get("overall")
            if acc is None:
                row += f"  {'n/a':>{cell_w}}"
                continue
            yerr = (s.get("accuracy_yerr") or {}).get("overall")
            if cfg.suppress_at_zero_shot and n == 0:
                yerr = None
            cell = f"{acc:.4f}" if yerr is None else f"{acc:.4f} ± {yerr:.4f}"
            row += f"  {cell:>{cell_w}}"
        lines.append(row)

    lines.append("")
    lines.append(
        "Notes: cells are ``acc ± yerr`` where yerr is the simple average of "
        "per-(task, split) variance units (mean over queries of per-query std "
        "across demo seeds). yerr is omitted at 0-shot."
    )
    lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / filename).write_text("\n".join(lines))


def _model_family_size_key(model_name: str) -> tuple[int, float, str]:
    family = get_model_family(model_name)
    fam_idx = (
        FAMILY_ORDER.index(family) if family in FAMILY_ORDER else len(FAMILY_ORDER)
    )
    match = re.search(r"(\d+(?:\.\d+)?)[Bb]", model_name)
    size = float(match.group(1)) if match else float("inf")
    return (fam_idx, size, model_name)


def _resolve_panel_yerr(
    summary: dict[str, Any],
    key: str,
    cfg: ErrorbarConfig,
) -> float | None:
    """Pick the yerr to render at one panel point.

    Returns ``accuracy_yerr[key]`` already pre-computed by the aggregator: at
    leaf level this is the (task, split) variance unit (mean over queries of
    per-query demo-set std); at cross-task / cross-split level this is the
    simple average of those leaf-level units. ``shots == 0`` is suppressed
    when ``cfg.suppress_at_zero_shot``.
    """
    if cfg.suppress_at_zero_shot and summary["n_shots"] == 0:
        return None
    return (summary.get("accuracy_yerr") or {}).get(key)


def _plot_cross_family_panels(
    panel_summaries: dict[str, dict[str, list[dict[str, Any]]]],
    output_dir: Path,
    *,
    layout: list[list[str]],
    display_names: dict[str, str],
    filename: str,
    show_titles: bool = True,
    legend_fontsize: str | int | None = None,
    errorbar: ErrorbarConfig | None = None,
    baseline: BaselineSpec | None = None,
    disable_variance: bool = False,
    dpi: int = 150,
) -> None:
    """Render a 2D grid of accuracy-vs-shots panels with a shared bottom legend."""
    cfg = errorbar or ErrorbarConfig()
    cfg_no_var = replace(cfg, style="none")
    n_rows = len(layout)
    n_cols = len(layout[0])
    fontsize_resolved = (
        legend_fontsize if isinstance(legend_fontsize, (int, float)) else 16
    )
    legend_band = fontsize_resolved / 36 + 0.25  # inches reserved for legend
    fig_h = 4 * n_rows + legend_band
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, fig_h),
        sharey=True,
        squeeze=False,
    )

    handle_map: dict[str, Any] = {}
    baseline_handle: Any = None
    for r, row in enumerate(layout):
        for c, key in enumerate(row):
            ax = axes[r][c]
            data = panel_summaries.get(key, {})
            panel_cfg = cfg_no_var if (disable_variance or key == "overall") else cfg

            if baseline is not None:
                line = ax.axhline(
                    baseline.y,
                    color=baseline.color,
                    linestyle=baseline.linestyle,
                    linewidth=baseline.linewidth,
                    zorder=baseline.zorder if baseline.zorder is not None else 1,
                )
                if baseline_handle is None:
                    baseline_handle = line

            for model_name in sorted(data.keys(), key=_model_family_size_key):
                summaries = data[model_name]
                if not summaries:
                    continue
                ordered = sorted(summaries, key=lambda s: s["n_shots"])
                shots = [s["n_shots"] for s in ordered]
                accs = [s["accuracy"]["overall"] for s in ordered]
                yerrs = [
                    _resolve_panel_yerr(s, "overall", panel_cfg)
                    for s in ordered
                ]
                line = plot_curve(
                    ax,
                    shots,
                    accs,
                    label=model_name,
                    color=get_model_color(model_name),
                    style="o-",
                    linewidth=2.5,
                    markersize=7,
                    yerr=yerrs,
                    errorbar=panel_cfg,
                )
                handle_map.setdefault(model_name, line)

            if show_titles:
                ax.set_title(display_names.get(key, key), fontsize=20)
            if r == n_rows - 1:
                ax.set_xlabel("Number of ICL shots", fontsize=18)
            if c == 0:
                ax.set_ylabel("Accuracy", fontsize=18)
            ax.tick_params(axis="both", labelsize=15)
            ax.grid(True, alpha=0.3)
            all_shots = sorted(
                {s["n_shots"] for sums in data.values() for s in sums}
            )
            if all_shots:
                ax.set_xticks(all_shots)

    axes[0][0].set_ylim(0, 1.05)
    sns.despine(fig=fig)

    if handle_map:
        ordered_names = sorted(handle_map.keys(), key=_model_family_size_key)
        legend_handles = [handle_map[n] for n in ordered_names]
        legend_labels = [get_model_display_name(n) for n in ordered_names]
        if baseline is not None and baseline_handle is not None and baseline.label:
            legend_handles.append(baseline_handle)
            legend_labels.append(baseline.label)
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=len(legend_handles),
            fontsize=fontsize_resolved,
            frameon=True,
            edgecolor="0.8",
            facecolor="white",
            framealpha=0.9,
            handletextpad=0.5,
            columnspacing=1.5,
        )
        fig.tight_layout(rect=[0, legend_band / fig_h, 1, 1])
    else:
        fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=dpi)
    plt.close(fig)


def plot_cross_family_task_grid(
    cross_model: dict[str, dict[str, dict[str, tuple[Path, list[dict[str, Any]]]]]],
    output_dir: Path,
    *,
    layout: list[list[str]],
    display_names: dict[str, str],
    target_splits: tuple[str, ...] = ("simple",),
    model_filter: ModelFilter | None = None,
    filename: str = "cross_family_task_grid.png",
    show_titles: bool = True,
    legend_fontsize: str | int | None = None,
    errorbar: ErrorbarConfig | None = None,
    baseline: BaselineSpec | None = None,
    dpi: int = 150,
) -> None:
    """Cross-family accuracy-vs-shots grid: one subplot per task plus an overall panel.

    Each task panel sample-weighted-aggregates across ``target_splits``. The
    ``"overall"`` key aggregates across all tasks AND splits.
    """
    cfg = errorbar or ErrorbarConfig()

    per_task_var = aggregate_per_task_with_variance(
        cross_model, list(target_splits), model_filter=model_filter
    )
    panel_summaries: dict[str, dict[str, list[dict[str, Any]]]] = {
        task: to_summaries_format_with_variance(model_counts)
        for task, model_counts in per_task_var.items()
    }

    aggregated_var = aggregate_across_tasks_with_variance(
        cross_model, list(target_splits)
    )

    def _keep(model_name: str) -> bool:
        return model_filter is None or model_filter(
            model_name, get_model_family(model_name)
        )

    panel_summaries["overall"] = {
        model: sums
        for model, sums in to_summaries_format_with_variance(
            aggregated_var.get("overall", {}),
        ).items()
        if _keep(model)
    }

    _plot_cross_family_panels(
        panel_summaries,
        output_dir,
        layout=layout,
        display_names=display_names,
        filename=filename,
        show_titles=show_titles,
        legend_fontsize=legend_fontsize,
        errorbar=cfg,
        baseline=baseline,
        dpi=dpi,
    )


def plot_cross_family_split_grid(
    cross_model: dict[str, dict[str, dict[str, tuple[Path, list[dict[str, Any]]]]]],
    output_dir: Path,
    *,
    layout: list[list[str]],
    display_names: dict[str, str],
    target_splits: tuple[str, ...] = ("simple", "noprior", "complex"),
    model_filter: ModelFilter | None = None,
    filename: str = "cross_family_split_grid.png",
    show_titles: bool = True,
    legend_fontsize: str | int | None = None,
    errorbar: ErrorbarConfig | None = None,
    baseline: BaselineSpec | None = None,
    disable_variance: bool = False,
    dpi: int = 150,
) -> None:
    """Cross-family accuracy-vs-shots grid: one subplot per split plus an overall panel.

    Each split panel sample-weighted-aggregates across all tasks. The
    ``"overall"`` key aggregates across all tasks AND all listed splits.
    """
    cfg = errorbar or ErrorbarConfig()

    def _keep(model_name: str) -> bool:
        return model_filter is None or model_filter(
            model_name, get_model_family(model_name)
        )

    aggregated_var = aggregate_across_tasks_with_variance(
        cross_model, list(target_splits)
    )
    panel_summaries: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for group in (*target_splits, "overall"):
        panel_summaries[group] = {
            model: sums
            for model, sums in to_summaries_format_with_variance(
                aggregated_var.get(group, {}),
            ).items()
            if _keep(model)
        }

    _plot_cross_family_panels(
        panel_summaries,
        output_dir,
        layout=layout,
        display_names=display_names,
        filename=filename,
        show_titles=show_titles,
        legend_fontsize=legend_fontsize,
        errorbar=cfg,
        baseline=baseline,
        disable_variance=disable_variance,
        dpi=dpi,
    )


def plot_complex_alignment_panels(
    counts_by_model: dict[str, dict[int, ShotAggregate]],
    output_dir: Path,
    *,
    panel_models: list[str],
    panel_titles: list[str],
    filename: str,
    series_keys: tuple[str, ...] = ("overall", "real", "counterfactual"),
    series_labels: tuple[str, ...] = (
        "Overall accuracy",
        "Real",
        "Counterfactual",
    ),
    series_styles: tuple[str, ...] = ("-", "--", ":"),
    series_markers: tuple[str, ...] = ("o", "s", "^"),
    errorbar: ErrorbarConfig | None = None,
    share_y: bool = True,
    dpi: int = 150,
) -> None:
    """Side-by-side accuracy-vs-shots panels with three line-style-distinguished curves.

    Each panel plots one model's three accuracy curves (one per ``series_keys``
    entry) using the model's color and the per-series ``linestyle`` + ``marker``.
    A single bottom-of-figure legend shows neutral-color line samples for the
    three series so the legend reads independently of model color. The yerr is
    the simple average of per-(task, complex) leaf yerrs.

    Set ``share_y=False`` to give each panel its own auto-scaled y-axis so
    fine-grain accuracy differences are visible.
    """
    cfg = errorbar or ErrorbarConfig()
    n_panels = len(panel_models)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(6 * n_panels, 5), sharey=share_y, squeeze=False
    )

    all_shots: set[int] = set()
    for model_name in panel_models:
        all_shots.update(counts_by_model.get(model_name, {}).keys())
    xticks = sorted(all_shots)

    def _resolve_shot(
        shot_entry: ShotAggregate | None, key: str,
    ) -> tuple[int, int, float | None]:
        """Return (correct, total, leaf_yerr_avg_or_None) for one ShotAggregate slot."""
        if not shot_entry:
            return 0, 0, None
        by_demo = shot_entry["counts"].get(key, {})
        c = sum(cc for cc, _ in by_demo.values())
        t = sum(tt for _, tt in by_demo.values())
        return c, t, average_leaf_yerrs(shot_entry["leaf_yerrs"].get(key, []))

    for i, (model_name, title) in enumerate(zip(panel_models, panel_titles)):
        ax = axes[0][i]
        counts = counts_by_model.get(model_name, {})
        color = get_model_color(model_name)
        for key, style, marker in zip(series_keys, series_styles, series_markers):
            xs: list[int] = []
            ys: list[float] = []
            yerrs: list[float | None] = []
            for n in sorted(counts.keys()):
                c, t, leaf_yerr = _resolve_shot(counts[n], key)
                if t == 0:
                    continue
                xs.append(n)
                ys.append(c / t)
                if cfg.suppress_at_zero_shot and n == 0:
                    yerrs.append(None)
                else:
                    yerrs.append(leaf_yerr)
            if xs:
                plot_curve(
                    ax,
                    xs,
                    ys,
                    color=color,
                    style=f"{marker}{style}",
                    linewidth=2.5,
                    markersize=7,
                    yerr=yerrs,
                    errorbar=cfg,
                )
        ax.set_title(title, fontsize=24)
        ax.set_xlabel("Number of ICL shots", fontsize=20)
        if i == 0 or not share_y:
            ax.set_ylabel("Accuracy", fontsize=20)
        ax.tick_params(axis="both", labelsize=17)
        ax.grid(True, alpha=0.3)
        if xticks:
            ax.set_xticks(xticks)

    if share_y:
        axes[0][0].set_ylim(0, 1.05)
    sns.despine(fig=fig)

    legend_handles = [
        Line2D(
            [0], [0],
            linestyle=style, marker=marker, color="black",
            linewidth=3.5, markersize=12, label=label,
        )
        for label, style, marker in zip(series_labels, series_styles, series_markers)
    ]
    fig.legend(
        legend_handles,
        list(series_labels),
        loc="lower center",
        ncol=len(series_labels),
        fontsize=24,
        frameon=True,
        edgecolor="0.8",
        facecolor="white",
        framealpha=0.9,
        handletextpad=0.5,
        columnspacing=2.0,
    )
    fig.tight_layout(rect=[0, 0.16, 1, 1])

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=dpi)
    plt.close(fig)


def write_complex_alignment_stats(
    counts_by_model: dict[str, dict[int, ShotAggregate]],
    output_path: Path,
    *,
    panel_models: list[str],
    panel_titles: list[str],
    series_keys: tuple[str, ...] = ("overall", "real", "counterfactual"),
    series_labels: tuple[str, ...] = ("Overall", "Real", "Counterfactual"),
) -> None:
    """Write a per-model accuracy table (acc + correct/total per series, per shot)."""
    def _resolve(shot_entry: ShotAggregate | None, key: str) -> tuple[int, int]:
        if not shot_entry:
            return 0, 0
        by_demo = shot_entry["counts"].get(key, {})
        c = sum(cc for cc, _ in by_demo.values())
        t = sum(tt for _, tt in by_demo.values())
        return c, t

    lines: list[str] = []
    header = f"{'n_shots':>8}"
    for label in series_labels:
        header += f"  {label + '_acc':>16}  {label + '_correct/total':>22}"
    sep = "-" * len(header)

    for model_name, title in zip(panel_models, panel_titles):
        counts = counts_by_model.get(model_name, {})
        lines.append(f"{title}  ({model_name})")
        lines.append(sep)
        lines.append(header)
        lines.append(sep)
        if not counts:
            lines.append("  (no data)")
            lines.append("")
            continue
        for n in sorted(counts.keys()):
            row = f"{n:>8}"
            for key in series_keys:
                c, t = _resolve(counts[n], key)
                acc_str = f"{c / t:.4f}" if t > 0 else "-"
                row += f"  {acc_str:>16}  {f'{c}/{t}':>22}"
            lines.append(row)
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def _delta_stats_from_shot_aggregate(
    per_shots: dict[int, ShotAggregate],
    key: str,
    *,
    zero_shots: int,
    target_shots: int,
) -> dict[str, float | None]:
    """Compute (acc_0, acc_target, delta, sem_target) for one task/key.

    SEM at the target shot count uses per-demo-set accuracies
    (``std(ddof=1) / sqrt(n)``). The 0-shot baseline is treated as a fixed
    point (single demo set), so ``sem_delta == sem_target``. Returns a dict
    with ``None`` accuracies if the slot or counts are missing.
    """
    def _acc(by_demo: dict[int, tuple[int, int]]) -> float | None:
        c = sum(cc for cc, _ in by_demo.values())
        t = sum(tt for _, tt in by_demo.values())
        return (c / t) if t > 0 else None

    slot0 = per_shots.get(zero_shots)
    slot_t = per_shots.get(target_shots)
    by_demo_0 = (slot0 or {"counts": {}})["counts"].get(key, {})
    by_demo_t = (slot_t or {"counts": {}})["counts"].get(key, {})
    acc0 = _acc(by_demo_0)
    acct = _acc(by_demo_t)

    sem: float | None = None
    per_demo_t = [c / t for c, t in by_demo_t.values() if t > 0]
    if len(per_demo_t) >= 2:
        arr = np.asarray(per_demo_t, dtype=float)
        sem = float(arr.std(ddof=1) / math.sqrt(len(arr)))

    delta = (acct - acc0) if (acc0 is not None and acct is not None) else None
    return {
        "acc_zero": acc0,
        "acc_target": acct,
        "delta": delta,
        "sem_target": sem,
        "n_demo_target": len(per_demo_t),
    }


def plot_complex_alignment_delta_bars(
    per_task_counts: dict[str, dict[str, dict[int, ShotAggregate]]],
    output_dir: Path,
    *,
    model_name: str,
    task_order: list[str],
    task_titles: list[str],
    filename: str,
    title: str,
    real_key: str = "original",
    counterfactual_key: str = "perturbed",
    real_label: str = "Real (gt = bias)",
    counterfactual_label: str = "Counterfactual (gt ≠ bias)",
    zero_shots: int = 0,
    target_shots: int = 16,
    counterfactual_color: str = "#ff7f0e",
    counterfactual_edge: str = "#cc6600",
    dpi: int = 150,
) -> None:
    """Grouped bar chart: ICL accuracy delta (target_shots - zero_shots) per task.

    Each task gets two bars: Real (``real_key``) in the model's color and
    Counterfactual (``counterfactual_key``) in ``counterfactual_color``.
    Error bars use SEM over the per-demo-set 16-shot accuracies (the 0-shot
    baseline is a single demo set, so its variance contribution is 0).

    Each bar is annotated with its signed delta value and a small "(0s→16s)"
    pair of absolute accuracies, so readers see both the change and the
    absolute floor/ceiling.
    """
    bar_color = get_model_color(model_name)
    bar_edge = mcolors.to_hex(np.clip(np.array(mcolors.to_rgb(bar_color)) * 0.7, 0, 1))

    deltas_real: list[float] = []
    deltas_cf: list[float] = []
    sem_real: list[float] = []
    sem_cf: list[float] = []
    annot_real: list[str] = []
    annot_cf: list[str] = []
    has_real: list[bool] = []
    has_cf: list[bool] = []

    for task in task_order:
        per_shots = per_task_counts.get(task, {}).get(model_name, {})
        for key, deltas, sems, annots, has in (
            (real_key, deltas_real, sem_real, annot_real, has_real),
            (counterfactual_key, deltas_cf, sem_cf, annot_cf, has_cf),
        ):
            stats = _delta_stats_from_shot_aggregate(
                per_shots, key,
                zero_shots=zero_shots, target_shots=target_shots,
            )
            if stats["delta"] is None:
                deltas.append(0.0)
                sems.append(0.0)
                annots.append("n/a")
                has.append(False)
            else:
                deltas.append(stats["delta"])
                sems.append(stats["sem_target"] or 0.0)
                annots.append(
                    f"({stats['acc_zero']:.2f}→{stats['acc_target']:.2f})"
                )
                has.append(True)

    n = len(task_order)
    x = np.arange(n, dtype=float)
    bw = 0.38

    fig, ax = plt.subplots(figsize=(max(8.0, 1.6 * n + 2.0), 5.5))

    real_bars = ax.bar(
        x - bw / 2, deltas_real, bw,
        yerr=sem_real, capsize=4,
        label=real_label,
        color=bar_color, alpha=0.85,
        edgecolor=bar_edge, linewidth=1.2,
        error_kw={"elinewidth": 1.2, "ecolor": "0.25"},
    )
    cf_bars = ax.bar(
        x + bw / 2, deltas_cf, bw,
        yerr=sem_cf, capsize=4,
        label=counterfactual_label,
        color=counterfactual_color, alpha=0.85,
        edgecolor=counterfactual_edge, linewidth=1.2,
        error_kw={"elinewidth": 1.2, "ecolor": "0.25"},
    )

    ax.axhline(0, color="0.3", linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(task_titles, fontsize=12)
    ax.set_ylabel(
        f"Accuracy delta ({target_shots}-shot − {zero_shots}-shot)",
        fontsize=12,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=11)

    all_deltas = [d for d, h in zip(deltas_real, has_real) if h]
    all_deltas += [d for d, h in zip(deltas_cf, has_cf) if h]
    all_sems = [s for s, h in zip(sem_real, has_real) if h]
    all_sems += [s for s, h in zip(sem_cf, has_cf) if h]
    if all_deltas:
        lo = min(all_deltas) - max(all_sems + [0.0]) - 0.05
        hi = max(all_deltas) + max(all_sems + [0.0]) + 0.10
        lo = min(lo, -0.05)
        hi = max(hi, 0.05)
        ax.set_ylim(lo, hi)

    y_lo, y_hi = ax.get_ylim()
    label_pad = (y_hi - y_lo) * 0.018
    annot_pad = (y_hi - y_lo) * 0.05

    for bars, deltas, sems, annots, has in (
        (real_bars, deltas_real, sem_real, annot_real, has_real),
        (cf_bars, deltas_cf, sem_cf, annot_cf, has_cf),
    ):
        for bar, delta, sem, annot, ok in zip(bars, deltas, sems, annots, has):
            cx = bar.get_x() + bar.get_width() / 2
            if not ok:
                ax.text(cx, label_pad, "n/a", ha="center", va="bottom",
                        fontsize=9, color="0.4")
                continue
            top = delta + (sem if delta >= 0 else 0.0)
            bot = delta - (sem if delta < 0 else 0.0)
            if delta >= 0:
                ax.text(cx, top + label_pad,
                        f"{delta:+.3f}", ha="center", va="bottom",
                        fontsize=10, fontweight="bold", color="0.15")
                ax.text(cx, top + label_pad + annot_pad,
                        annot, ha="center", va="bottom",
                        fontsize=8, color="0.4")
            else:
                ax.text(cx, bot - label_pad,
                        f"{delta:+.3f}", ha="center", va="top",
                        fontsize=10, fontweight="bold", color="0.15")
                ax.text(cx, bot - label_pad - annot_pad,
                        annot, ha="center", va="top",
                        fontsize=8, color="0.4")

    finish_figure(fig, ax)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=dpi)
    plt.close(fig)


def write_complex_alignment_delta_stats(
    per_task_counts: dict[str, dict[str, dict[int, ShotAggregate]]],
    output_path: Path,
    *,
    model_name: str,
    task_order: list[str],
    task_titles: list[str],
    real_key: str = "original",
    counterfactual_key: str = "perturbed",
    real_label: str = "Real",
    counterfactual_label: str = "Counterfactual",
    zero_shots: int = 0,
    target_shots: int = 16,
) -> None:
    """Companion table for plot_complex_alignment_delta_bars.

    Per task, prints acc_0, acc_target, delta, and SEM (over per-demo-set
    accuracies at the target shot count) for both Real and Counterfactual.
    """
    cell_w = 14
    header_cols = (
        "task",
        f"{real_label}_acc0",
        f"{real_label}_acc{target_shots}",
        f"{real_label}_delta",
        f"{real_label}_sem",
        f"{counterfactual_label}_acc0",
        f"{counterfactual_label}_acc{target_shots}",
        f"{counterfactual_label}_delta",
        f"{counterfactual_label}_sem",
    )
    header = f"{header_cols[0]:>20}" + "".join(
        f"  {h:>{cell_w}}" for h in header_cols[1:]
    )
    sep = "-" * len(header)

    lines: list[str] = []
    lines.append(f"{model_name}  ({zero_shots}-shot → {target_shots}-shot)")
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    def _fmt(v: float | None, *, signed: bool = False) -> str:
        if v is None:
            return "n/a"
        return f"{v:+.4f}" if signed else f"{v:.4f}"

    for task, title in zip(task_order, task_titles):
        per_shots = per_task_counts.get(task, {}).get(model_name, {})
        r = _delta_stats_from_shot_aggregate(
            per_shots, real_key,
            zero_shots=zero_shots, target_shots=target_shots,
        )
        c = _delta_stats_from_shot_aggregate(
            per_shots, counterfactual_key,
            zero_shots=zero_shots, target_shots=target_shots,
        )
        row_vals = (
            _fmt(r["acc_zero"]),
            _fmt(r["acc_target"]),
            _fmt(r["delta"], signed=True),
            _fmt(r["sem_target"]),
            _fmt(c["acc_zero"]),
            _fmt(c["acc_target"]),
            _fmt(c["delta"], signed=True),
            _fmt(c["sem_target"]),
        )
        row = f"{title:>20}" + "".join(f"  {v:>{cell_w}}" for v in row_vals)
        lines.append(row)

    lines.append(sep)
    lines.append(
        f"# SEM = std(per-demo-set acc at {target_shots}-shot, ddof=1) / sqrt(n_demos); "
        f"acc0 has a single demo set so its variance contribution is 0."
    )
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def plot_complex_alignment_delta_heatmap(
    per_task_counts: dict[str, dict[str, dict[int, ShotAggregate]]],
    output_dir: Path,
    *,
    panel_models: list[str],
    panel_titles: list[str],
    task_order: list[str],
    task_titles: list[str],
    filename: str,
    panel_keys: tuple[str, ...] = ("overall", "perturbed", "original"),
    panel_labels: tuple[str, ...] = ("Overall", "Counterfactual", "Real"),
    zero_shots: int = 0,
    target_shots: int = 16,
    cmap: str = "RdBu_r",
    dpi: int = 150,
) -> None:
    """3-panel heatmap of (target_shots − zero_shots) accuracy delta per (task, model).

    Each panel shows one alignment slice; rows are tasks, columns are models.
    Missing (task, model) cells are rendered as gray with an "n/a" annotation.
    All panels share a symmetric diverging colorbar centered at 0 so the cells
    can be compared across panels.
    """
    n_tasks = len(task_order)
    n_models = len(panel_models)
    n_panels = len(panel_keys)

    deltas = np.full((n_panels, n_tasks, n_models), np.nan, dtype=float)
    for k_idx, key in enumerate(panel_keys):
        for t_idx, task in enumerate(task_order):
            for m_idx, model_name in enumerate(panel_models):
                per_shots = per_task_counts.get(task, {}).get(model_name, {})
                stats = _delta_stats_from_shot_aggregate(
                    per_shots, key,
                    zero_shots=zero_shots, target_shots=target_shots,
                )
                if stats["delta"] is not None:
                    deltas[k_idx, t_idx, m_idx] = stats["delta"]

    finite = deltas[np.isfinite(deltas)]
    if finite.size:
        vmax = float(np.nanmax(np.abs(finite)))
        vmax = max(vmax, 0.05)
    else:
        vmax = 0.5
    vmin = -vmax

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(2.4 * n_models * n_panels + 1.5, 0.7 * n_tasks + 2.0),
        squeeze=False,
        sharey=True,
    )

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="0.85")
    im = None
    for k_idx, (key, label) in enumerate(zip(panel_keys, panel_labels)):
        ax = axes[0][k_idx]
        data = np.ma.masked_invalid(deltas[k_idx])
        im = ax.imshow(
            data, cmap=cmap_obj, vmin=vmin, vmax=vmax,
            aspect="auto", origin="upper",
        )
        ax.set_xticks(np.arange(n_models))
        ax.set_xticklabels(panel_titles, rotation=30, ha="right", fontsize=11)
        ax.set_yticks(np.arange(n_tasks))
        if k_idx == 0:
            ax.set_yticklabels(task_titles, fontsize=11)
            ax.set_ylabel("Task", fontsize=12)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xticks(np.arange(n_models + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_tasks + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", length=0)

        for t_idx in range(n_tasks):
            for m_idx in range(n_models):
                v = deltas[k_idx, t_idx, m_idx]
                if not np.isfinite(v):
                    ax.text(m_idx, t_idx, "n/a",
                            ha="center", va="center",
                            fontsize=9, color="0.3")
                    continue
                bg = cmap_obj((v - vmin) / (vmax - vmin) if vmax > vmin else 0.5)
                lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
                txt_color = "white" if lum < 0.5 else "0.1"
                ax.text(m_idx, t_idx, f"{v:+.3f}",
                        ha="center", va="center",
                        fontsize=10, color=txt_color, fontweight="bold")

    fig.suptitle(
        f"ICL accuracy delta ({target_shots}-shot − {zero_shots}-shot)",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
    cbar_ax = fig.add_axes([0.94, 0.18, 0.014, 0.66])
    fig.colorbar(im, cax=cbar_ax, label=f"Δ accuracy")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=dpi)
    plt.close(fig)


def write_complex_alignment_delta_heatmap_stats(
    per_task_counts: dict[str, dict[str, dict[int, ShotAggregate]]],
    output_path: Path,
    *,
    panel_models: list[str],
    panel_titles: list[str],
    task_order: list[str],
    task_titles: list[str],
    panel_keys: tuple[str, ...] = ("overall", "perturbed", "original"),
    panel_labels: tuple[str, ...] = ("Overall", "Counterfactual", "Real"),
    zero_shots: int = 0,
    target_shots: int = 16,
) -> None:
    """Companion table for ``plot_complex_alignment_delta_heatmap``.

    Per (alignment, task, model), prints the 0-shot and target-shot accuracy
    plus the signed delta. ``n/a`` is emitted when either shot count is
    missing for that cell.
    """
    def _fmt(v: float | None, *, signed: bool = False) -> str:
        if v is None:
            return "n/a"
        return f"{v:+.4f}" if signed else f"{v:.4f}"

    cell_w = 10
    header_cols = ["task"]
    for title in panel_titles:
        header_cols += [
            f"{title}_acc{zero_shots}",
            f"{title}_acc{target_shots}",
            f"{title}_delta",
        ]
    header = f"{header_cols[0]:>20}" + "".join(
        f"  {h:>{cell_w}}" for h in header_cols[1:]
    )
    sep = "-" * len(header)

    lines: list[str] = []
    for key, label in zip(panel_keys, panel_labels):
        lines.append(f"=== {label} ({zero_shots}-shot → {target_shots}-shot) ===")
        lines.append(sep)
        lines.append(header)
        lines.append(sep)
        for task, t_title in zip(task_order, task_titles):
            row_vals: list[str] = []
            for model_name in panel_models:
                per_shots = per_task_counts.get(task, {}).get(model_name, {})
                stats = _delta_stats_from_shot_aggregate(
                    per_shots, key,
                    zero_shots=zero_shots, target_shots=target_shots,
                )
                row_vals += [
                    _fmt(stats["acc_zero"]),
                    _fmt(stats["acc_target"]),
                    _fmt(stats["delta"], signed=True),
                ]
            row = f"{t_title:>20}" + "".join(
                f"  {v:>{cell_w}}" for v in row_vals
            )
            lines.append(row)
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def plot_dumbbell(
    summaries: list[dict[str, Any]],
    model_name: str,
    output_dir: Path,
    dpi: int = 150,
    raw_distributions: dict[int, dict[str, dict[str, dict[str, list[float]]]]] | None = None,
) -> None:
    """Horizontal dumbbell chart: P(correct label) at 0-shot vs 16-shot per GT label."""
    shot_map = {s["n_shots"]: s for s in summaries}
    s0 = shot_map.get(0)
    s16 = shot_map.get(16)
    if s0 is None or s16 is None:
        return

    label_order = s0["label_order"]
    color = get_model_color(model_name)

    if raw_distributions is not None:
        gt_labels = sorted(
            set(raw_distributions.get(0, {})) & set(raw_distributions.get(16, {}))
        )
        p0_vals = [float(np.mean(_flatten(raw_distributions[0][gt]["all"]))) for gt in gt_labels]
        p16_vals = [float(np.mean(_flatten(raw_distributions[16][gt]["all"]))) for gt in gt_labels]
    else:
        gt_labels = sorted(s0["per_label_mean_distribution"].keys())
        p0_vals, p16_vals = [], []
        for gt in gt_labels:
            gt_idx = label_order.index(gt)
            dist0 = s0["per_label_mean_distribution"].get(gt, [])
            dist16 = s16["per_label_mean_distribution"].get(gt, [])
            p0_vals.append(dist0[gt_idx] if gt_idx < len(dist0) else 0)
            p16_vals.append(dist16[gt_idx] if gt_idx < len(dist16) else 0)

    y_pos = np.arange(len(gt_labels))
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.75 * len(gt_labels) + 1.8)))

    for i in range(1, len(gt_labels)):
        ax.axhline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3, zorder=1)

    for idx, (x0, x1) in enumerate(zip(p0_vals, p16_vals)):
        ax.plot([x0, x1], [y_pos[idx], y_pos[idx]], "-", color=color, linewidth=3.0, zorder=2)
        ax.plot(
            x0,
            y_pos[idx],
            "o",
            color=color,
            markersize=10,
            zorder=3,
            markeredgecolor="white",
            markeredgewidth=1.0,
        )
        ax.plot(
            x1,
            y_pos[idx],
            "D",
            color=color,
            markersize=11,
            zorder=3,
            markeredgecolor="white",
            markeredgewidth=1.0,
        )
        improved = x1 > x0
        marker_char = "\u2713" if improved else "\u2717"
        marker_color = "#2ca02c" if improved else "#d62728"
        ax.text(
            max(x0, x1) + 0.03,
            y_pos[idx],
            marker_char,
            color=marker_color,
            fontsize=22,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=4,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(gt_labels, fontsize=14)
    ax.set_ylabel("Answer Labels", fontsize=15)
    ax.set_xlabel("P(correct label)", fontsize=15)
    ax.set_title(f"P(correct): {model_name}", fontsize=17)
    ax.tick_params(axis="x", labelsize=13)
    ax.set_xlim(0, 1.12)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            linestyle="None",
            markersize=12,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label="0-shot",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color=color,
            linestyle="None",
            markersize=13,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label="16-shot",
        ),
        Line2D([0], [0], marker="None", linestyle="None", label="\u2713 improved"),
        Line2D([0], [0], marker="None", linestyle="None", label="\u2717 worsened"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=14)

    finish_figure(fig, ax)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "dumbbell_pcorrect.png", dpi=dpi)
    plt.close(fig)


def _draw_dumbbell(
    ax,
    gts: list[str],
    a0: dict[str, dict[str, dict[str, list[float]]]],
    a16: dict[str, dict[str, dict[str, list[float]]]],
    alignment: str,
    color: str,
) -> None:
    """Draw dumbbell glyphs on *ax* for a single alignment."""
    y_pos = np.arange(len(gts))

    for i in range(1, len(gts)):
        ax.axhline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3, zorder=1)

    for idx, gt in enumerate(gts):
        x0 = float(np.mean(_flatten(a0[gt][alignment])))
        x1 = float(np.mean(_flatten(a16[gt][alignment])))
        ax.plot([x0, x1], [y_pos[idx], y_pos[idx]], "-", color=color, linewidth=3.0, zorder=2)
        ax.plot(x0, y_pos[idx], "o", color=color, markersize=10, zorder=3,
                markeredgecolor="white", markeredgewidth=1.0)
        ax.plot(x1, y_pos[idx], "D", color=color, markersize=11, zorder=3,
                markeredgecolor="white", markeredgewidth=1.0)
        improved = x1 > x0
        ax.text(
            max(x0, x1) + 0.03, y_pos[idx],
            "\u2713" if improved else "\u2717",
            color="#2ca02c" if improved else "#d62728",
            fontsize=22, fontweight="bold", ha="left", va="center", zorder=4,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(gts, fontsize=14)
    ax.set_ylabel("Answer Labels", fontsize=15)
    ax.set_xlabel("P(correct label)", fontsize=15)
    ax.tick_params(axis="x", labelsize=13)
    ax.set_xlim(0, 1.12)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()


def _dumbbell_legend(color: str) -> list:
    return [
        Line2D([0], [0], marker="o", color=color, linestyle="None",
               markersize=12, markeredgecolor="white", markeredgewidth=1.0,
               label="0-shot"),
        Line2D([0], [0], marker="D", color=color, linestyle="None",
               markersize=13, markeredgecolor="white", markeredgewidth=1.0,
               label="16-shot"),
        Line2D([0], [0], marker="None", linestyle="None", label="\u2713 improved"),
        Line2D([0], [0], marker="None", linestyle="None", label="\u2717 worsened"),
    ]


def _plot_dumbbell_split(
    raw_distributions: dict[int, dict[str, dict[str, dict[str, list[float]]]]],
    label_order: list[str],
    model_name: str,
    output_dir: Path,
    dpi: int,
) -> None:
    """One dumbbell figure per alignment (real, counterfactual)."""
    a0 = raw_distributions.get(0, {})
    a16 = raw_distributions.get(16, {})
    color = get_model_color(model_name)

    output_dir.mkdir(parents=True, exist_ok=True)

    for alignment, tag in (("real", "Real"), ("counterfactual", "Counterfactual")):
        gts = sorted(
            gt for gt in set(a0) & set(a16)
            if alignment in a0.get(gt, {}) and alignment in a16.get(gt, {})
        )
        if not gts:
            continue

        fig, ax = plt.subplots(figsize=(8, max(3.5, 0.75 * len(gts) + 1.8)))
        _draw_dumbbell(ax, gts, a0, a16, alignment, color)
        ax.set_title(f"P(correct): {model_name}\n[{tag}]", fontsize=17)

        fig.legend(handles=_dumbbell_legend(color), loc="lower center", ncol=4, fontsize=14)
        finish_figure(fig, ax)
        fig.tight_layout(rect=[0, 0.1, 1, 1])
        fig.savefig(output_dir / f"dumbbell_pcorrect_{alignment}.png", dpi=dpi)
        plt.close(fig)


def _plot_dumbbell_both(
    raw_distributions: dict[int, dict[str, dict[str, dict[str, list[float]]]]],
    model_name: str,
    output_dir: Path,
    dpi: int,
) -> None:
    """Two-panel dumbbell: counterfactual (left) + real (right), shared x-axis."""
    a0 = raw_distributions.get(0, {})
    a16 = raw_distributions.get(16, {})

    gts_by_alignment: dict[str, list[str]] = {}
    for alignment in ("counterfactual", "real"):
        gts_by_alignment[alignment] = sorted(
            gt for gt in set(a0) & set(a16)
            if alignment in a0.get(gt, {}) and alignment in a16.get(gt, {})
        )
    n_cf = len(gts_by_alignment["counterfactual"])
    n_real = len(gts_by_alignment["real"])
    if n_cf == 0 or n_real == 0:
        return

    color = get_model_color(model_name)
    total_rows = n_cf + n_real
    height = max(7, 0.75 * max(n_cf, n_real) + 3)
    fig, axes = plt.subplots(
        2, 1,
        figsize=(8, height),
        gridspec_kw={"height_ratios": [n_cf, n_real], "hspace": 0.15},
        sharex=True,
    )

    display_tag = {"counterfactual": "Counterfactual", "real": "Real"}
    for ax, alignment in zip(axes, ("counterfactual", "real")):
        gts = gts_by_alignment[alignment]
        _draw_dumbbell(ax, gts, a0, a16, alignment, color)
        ax.set_title(f"[{display_tag[alignment]}]", fontsize=15)

    axes[0].set_xlabel("")
    axes[1].tick_params(axis="x", labelsize=13)

    fig.suptitle(f"P(correct): {model_name}", fontsize=17)

    fig.legend(
        handles=_dumbbell_legend(color), loc="lower center",
        bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=14,
    )

    sns.despine(fig=fig)
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "dumbbell_pcorrect_both.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _flatten(by_entry: dict[str, list[float]]) -> list[float]:
    """Flatten ``{image_id: [vals]}`` to a single flat list of sample values."""
    return [v for vs in by_entry.values() for v in vs]


def _collect_accuracy_keys(summaries: list[dict[str, Any]]) -> list[str]:
    """Return accuracy keys in insertion order across all summaries."""
    keys: list[str] = []
    seen: set[str] = set()
    for s in summaries or []:
        for k in (s.get("accuracy") or {}).keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def _overall_vals_by_shot(
    raw: dict[int, dict[str, dict[str, dict[str, list[float]]]]],
    alignment: str,
    shot_counts: list[int],
) -> dict[int, list[float]]:
    """Flatten P(correct) across all GTs per shot count for a given alignment."""
    out: dict[int, list[float]] = {}
    for n in shot_counts:
        combined: list[float] = []
        for _, aligns in raw.get(n, {}).items():
            by_entry = aligns.get(alignment)
            if by_entry:
                combined.extend(_flatten(by_entry))
        if combined:
            out[n] = combined
    return out


def write_stats_table(
    raw: dict[int, dict[str, dict[str, dict[str, list[float]]]]],
    model_name: str,
    output_dir: Path,
    *,
    summaries: list[dict[str, Any]] | None = None,
) -> None:
    """Write table.txt with overall statistics and per-plot statistics.

    Overall section: accuracy per shot (from summaries) + P(correct) per shot
    (overall and per-GT) for every available shot count. Per-plot section:
    the slices consumed by each figure (0-shot vs 16-shot for dumbbell,
    violin, bar).
    """
    has_align = has_alignment_keys(raw)
    alignments = ("counterfactual", "real") if has_align else ("all",)
    shot_counts = sorted(raw.keys())
    summaries = summaries or []
    acc_keys = _collect_accuracy_keys(summaries)

    lines: list[str] = []
    lines.append(f"P(correct label) statistics: {model_name}")
    lines.append("=" * 80)

    # =====================================================================
    # Overall statistics
    # =====================================================================
    lines.append("")
    lines.append("## Overall statistics")
    lines.append("=" * 80)

    # --- Accuracy vs number of ICL demonstrations (from summaries) ---
    lines.append("")
    lines.append("### Accuracy across number of ICL demonstrations")
    lines.append("-" * 80)
    if summaries and acc_keys:
        header = f"{'n_shots':<10}" + "".join(f"  {k:>12}" for k in acc_keys)
        lines.append(header)
        lines.append("-" * len(header))
        shot_to_s = {s["n_shots"]: s for s in summaries}
        for n in sorted(shot_to_s):
            row = f"{n:<10}"
            for k in acc_keys:
                v = (shot_to_s[n].get("accuracy") or {}).get(k)
                row += f"  {'n/a':>12}" if v is None else f"  {v:>12.4f}"
            lines.append(row)
    else:
        lines.append("(no summary data available)")

    # --- P(correct label) across all shots ---
    for alignment in alignments:
        tag = alignment.capitalize() if has_align else "All"

        lines.append("")
        lines.append(f"### P(correct label) across all shots [{tag}]")
        lines.append("-" * 80)

        # Overall P(correct) flattened across GTs per shot
        lines.append("Overall (flattened across ground-truth labels):")
        hdr = f"{'n_shots':<10}  {'mean':>10}  {'var':>10}  {'N':>8}"
        lines.append(hdr)
        lines.append("-" * len(hdr))
        overall_by_shot = _overall_vals_by_shot(raw, alignment, shot_counts)
        for n in shot_counts:
            vals = overall_by_shot.get(n)
            if vals is None:
                lines.append(f"{n:<10}  {'n/a':>10}  {'n/a':>10}  {'n/a':>8}")
            else:
                lines.append(
                    f"{n:<10}  {float(np.mean(vals)):>10.4f}  "
                    f"{float(np.var(vals)):>10.4f}  {len(vals):>8d}"
                )

        all_gts = sorted({
            gt for n in shot_counts
            for gt, aligns in raw.get(n, {}).items() if alignment in aligns
        })
        if not all_gts:
            continue

        col_hdr = f"{'GT Label':<12}" + "".join(f"  {f'{n}-shot':>10}" for n in shot_counts)

        lines.append("")
        lines.append("Per-GT mean P(correct):")
        lines.append(col_hdr)
        lines.append("-" * len(col_hdr))
        for gt in all_gts:
            row = f"{gt:<12}"
            for n in shot_counts:
                by_entry = raw.get(n, {}).get(gt, {}).get(alignment)
                if by_entry is None:
                    row += f"  {'n/a':>10}"
                else:
                    row += f"  {float(np.mean(_flatten(by_entry))):>10.4f}"
            lines.append(row)

        lines.append("")
        lines.append("Per-GT variance P(correct):")
        lines.append(col_hdr)
        lines.append("-" * len(col_hdr))
        for gt in all_gts:
            row = f"{gt:<12}"
            for n in shot_counts:
                by_entry = raw.get(n, {}).get(gt, {}).get(alignment)
                if by_entry is None:
                    row += f"  {'n/a':>10}"
                else:
                    row += f"  {float(np.var(_flatten(by_entry))):>10.4f}"
            lines.append(row)

    # =====================================================================
    # Per-plot statistics
    # =====================================================================
    lines.append("")
    lines.append("")
    lines.append("## Per-plot statistics")
    lines.append("=" * 80)

    lines.append("")
    lines.append("### accuracy_vs_shots.png — Accuracy vs. number of ICL shots")
    lines.append("Source: summaries[n].accuracy[*] at every shot count.")
    lines.append("(See 'Accuracy across number of ICL demonstrations' above.)")

    have_ref = 0 in shot_counts and 16 in shot_counts
    if not have_ref:
        lines.append("")
        lines.append(
            "(dumbbell/violin/bar plots require both 0-shot and 16-shot data; skipping)"
        )
    else:
        for alignment in alignments:
            tag = alignment.capitalize() if has_align else "All"
            gts = sorted(
                gt for gt in set(raw[0]) & set(raw[16])
                if alignment in raw[0].get(gt, {}) and alignment in raw[16].get(gt, {})
            )
            if not gts:
                continue

            lines.append("")
            lines.append(
                f"### dumbbell_pcorrect.png [{tag}] — 0-shot vs 16-shot mean P(correct) per GT"
            )
            lines.append("-" * 80)
            hdr = (
                f"{'GT Label':<12}  {'0-shot mean':>12}  "
                f"{'16-shot mean':>13}  {'delta':>10}"
            )
            lines.append(hdr)
            lines.append("-" * len(hdr))
            for gt in gts:
                m0 = float(np.mean(_flatten(raw[0][gt][alignment])))
                m16 = float(np.mean(_flatten(raw[16][gt][alignment])))
                lines.append(
                    f"{gt:<12}  {m0:>12.4f}  {m16:>13.4f}  {m16 - m0:>+10.4f}"
                )

            lines.append("")
            lines.append(
                f"### violin_pcorrect.png [{tag}] — 0-shot vs 16-shot per-sample distribution"
            )
            lines.append("-" * 80)
            hdr = (
                f"{'GT Label':<12}  {'0-mean':>8}  {'0-med':>8}  {'0-std':>8}  "
                f"{'0-N':>6}  {'16-mean':>8}  {'16-med':>8}  {'16-std':>8}  {'16-N':>6}"
            )
            lines.append(hdr)
            lines.append("-" * len(hdr))
            for gt in gts:
                v0 = _flatten(raw[0][gt][alignment])
                v16 = _flatten(raw[16][gt][alignment])
                lines.append(
                    f"{gt:<12}  "
                    f"{float(np.mean(v0)):>8.4f}  {float(np.median(v0)):>8.4f}  "
                    f"{float(np.std(v0)):>8.4f}  {len(v0):>6d}  "
                    f"{float(np.mean(v16)):>8.4f}  {float(np.median(v16)):>8.4f}  "
                    f"{float(np.std(v16)):>8.4f}  {len(v16):>6d}"
                )

            lines.append("")
            lines.append(
                f"### bar_pcorrect.png [{tag}] — 0-shot vs 16-shot per-entry mean ± SEM"
            )
            lines.append("-" * 80)
            hdr = (
                f"{'GT Label':<12}  {'0-mean':>8}  {'0-SEM':>8}  {'0-N':>6}  "
                f"{'16-mean':>8}  {'16-SEM':>8}  {'16-N':>6}"
            )
            lines.append(hdr)
            lines.append("-" * len(hdr))
            for gt in gts:
                e0 = _entry_means(raw[0][gt][alignment])
                e16 = _entry_means(raw[16][gt][alignment])
                m0, s0 = _bar_stats(raw[0][gt][alignment])
                m16, s16 = _bar_stats(raw[16][gt][alignment])
                lines.append(
                    f"{gt:<12}  {m0:>8.4f}  {s0:>8.4f}  {len(e0):>6d}  "
                    f"{m16:>8.4f}  {s16:>8.4f}  {len(e16):>6d}"
                )

    lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "table.txt").write_text("\n".join(lines) + "\n")


def write_accuracy_breakdown(
    input_dir: Path,
    model_name: str,
    output_dir: Path,
    *,
    shot_counts: tuple[int, ...],
) -> None:
    """Write accuracy_breakdown.txt: per-GT-label accuracy across shots.

    Always emits a "(all examples)" section. If any record carries an
    ``expected_bias`` (i.e. the leaf is the complex split), additionally
    emits "[Real - gt == expected_bias]" and
    "[Counterfactual - gt != expected_bias]" sections. Each cell is
    rendered as ``acc (N)`` so the reader sees both accuracy and sample
    size at every shot.
    """
    counts: dict[str, dict[str, dict[int, list[int]]]] = {
        "all": {},
        "real": {},
        "counterfactual": {},
    }
    has_alignment = False
    available_shots: list[int] = []
    for n in shot_counts:
        path = input_dir / f"shots_{n}" / "results.jsonl"
        records = load_jsonl(path)
        if not records:
            continue
        available_shots.append(n)
        for r in records:
            gt = r.get("ground_truth")
            correct = r.get("correct")
            if gt is None or correct is None:
                continue
            inc = 1 if bool(correct) else 0
            counts["all"].setdefault(gt, {}).setdefault(n, [0, 0])
            counts["all"][gt][n][0] += inc
            counts["all"][gt][n][1] += 1
            bias = r.get("expected_bias") or ""
            if bias:
                has_alignment = True
                key = "real" if bias == gt else "counterfactual"
                counts[key].setdefault(gt, {}).setdefault(n, [0, 0])
                counts[key][gt][n][0] += inc
                counts[key][gt][n][1] += 1

    if not available_shots:
        return

    sections: list[tuple[str, dict[str, dict[int, list[int]]]]] = [
        ("Per-label accuracy (all examples)", counts["all"]),
    ]
    if has_alignment:
        sections.append(
            ("Per-label accuracy [Real - gt == expected_bias]", counts["real"])
        )
        sections.append(
            (
                "Per-label accuracy [Counterfactual - gt != expected_bias]",
                counts["counterfactual"],
            )
        )

    cell_w = 14
    label_w = 14
    shots_sorted = sorted(available_shots)

    lines: list[str] = []
    lines.append(f"Accuracy breakdown: {model_name}")
    lines.append("=" * 80)

    for title, by_gt in sections:
        lines.append("")
        lines.append(f"## {title}")
        lines.append("-" * 80)
        if not by_gt:
            lines.append("(no records in this slice)")
            continue
        header = f"{'GT Label':<{label_w}}" + "".join(
            f"  {f'{n}-shot':>{cell_w}}" for n in shots_sorted
        )
        lines.append(header)
        lines.append("-" * len(header))
        for gt in sorted(by_gt.keys()):
            row = f"{gt:<{label_w}}"
            for n in shots_sorted:
                pair = by_gt[gt].get(n)
                if pair is None or pair[1] == 0:
                    row += f"  {'n/a':>{cell_w}}"
                else:
                    acc = pair[0] / pair[1]
                    row += f"  {f'{acc:.4f} ({pair[1]})':>{cell_w}}"
            lines.append(row)

        lines.append("-" * len(header))
        overall_row = f"{'Overall':<{label_w}}"
        for n in shots_sorted:
            tot_correct = sum(
                by_gt[gt].get(n, [0, 0])[0] for gt in by_gt
            )
            tot_n = sum(by_gt[gt].get(n, [0, 0])[1] for gt in by_gt)
            if tot_n == 0:
                overall_row += f"  {'n/a':>{cell_w}}"
            else:
                acc = tot_correct / tot_n
                overall_row += f"  {f'{acc:.4f} ({tot_n})':>{cell_w}}"
        lines.append(overall_row)

    lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "accuracy_breakdown.txt").write_text("\n".join(lines) + "\n")


def _entry_means(by_entry: dict[str, list[float]]) -> list[float]:
    """One mean per image_id — collapses per-seed repetitions to per-entry scalars."""
    return [float(np.mean(vs)) for vs in by_entry.values() if vs]


def _bar_stats(by_entry: dict[str, list[float]]) -> tuple[float, float]:
    """Return (mean, SEM) over per-entry means for a single group."""
    vals = _entry_means(by_entry)
    if len(vals) < 2:
        return (float(np.mean(vals)) if vals else 0.0, 0.0)
    arr = np.asarray(vals)
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr)))


def _violin_dots(ax, vals, x, side, *, color, width, polygon=None, size=24):
    """Scatter entry-level dots inside a half-violin.

    When ``polygon`` (Nx2 verts from ``_half_violin``) is given, jitter is
    clamped to the local violin half-width at each dot's y. Otherwise falls
    back to uniform rectangular jitter within ``width/2``.
    """
    if not vals:
        return
    rng = np.random.default_rng(0)
    sign = -1.0 if side == "left" else 1.0
    vals_arr = np.asarray(vals, dtype=float)

    if polygon is not None and len(polygon) >= 2:
        dx = np.abs(polygon[:, 0] - x)
        mask = dx > 1e-9
        if mask.sum() >= 2:
            order = np.argsort(polygon[mask, 1])
            y_curve = polygon[mask, 1][order]
            dx_curve = dx[mask][order]
            local_hw = np.interp(vals_arr, y_curve, dx_curve)
            hw = np.maximum(local_hw - 0.02, 0.01)
            jitter = rng.uniform(0, 1, size=len(vals_arr)) * hw * sign
            ax.scatter(x + jitter, vals_arr, s=size, color=color, alpha=0.7,
                       edgecolors="white", linewidths=0.4, zorder=3)
            return

    jitter = rng.uniform(0.05, max(0.07, width / 2 - 0.02), size=len(vals_arr)) * sign
    ax.scatter(x + jitter, vals_arr, s=size, color=color, alpha=0.7,
               edgecolors="white", linewidths=0.4, zorder=3)


def _half_violin(ax, data, x, half, facecolor, *, width=0.7, alpha=0.6):
    """Draw a half violin to the left (``half='left'``) or right of ``x``.

    Overlays dashed Q1/Q3 ticks from ``x`` (centerline) to the polygon boundary.
    Returns the clipped polygon vertices (Nx2) for downstream dot clamping.
    """
    if not data:
        return None
    parts = ax.violinplot(
        [data], positions=[x], vert=True, widths=width,
        showextrema=False, showmedians=False,
    )
    out_verts = None
    for body in parts["bodies"]:
        verts = body.get_paths()[0].vertices
        if half == "right":
            verts[:, 0] = np.clip(verts[:, 0], x, None)
        else:
            verts[:, 0] = np.clip(verts[:, 0], None, x)
        body.set_facecolor(facecolor)
        body.set_edgecolor(facecolor)
        body.set_alpha(alpha)
        q1, q3 = np.percentile(data, [25, 75])
        for qy in (q1, q3):
            qx = verts[int(np.argmin(np.abs(verts[:, 1] - qy))), 0]
            ax.plot([x, qx], [qy, qy], linestyle="--", color="black",
                    linewidth=0.8, alpha=0.5, zorder=3)
        out_verts = verts.copy()
    return out_verts


def _motif(
    ax,
    vals0,
    vals16,
    x,
    *,
    color,
    linestyle="-",
    check_y=1.02,
    linewidth=3.0,
    marker_scale=1.0,
    check_fontsize=18,
):
    """Dumbbell motif (circle, diamond, line, ✓/✗) on centerline ``x``.

    Uses mean (not median) to match ``plot_dumbbell``. ✓/✗ is placed at the
    fixed ``check_y`` above the violin so markers don't overlap the body.
    """
    m0 = float(np.mean(vals0)) if vals0 else 0.0
    m16 = float(np.mean(vals16)) if vals16 else 0.0
    ax.plot([x, x], [m0, m16], linestyle, color=color, linewidth=linewidth, zorder=4)
    ax.plot(x, m0, "o", color=color, markersize=10 * marker_scale, zorder=5,
            markeredgecolor="white", markeredgewidth=1.0)
    ax.plot(x, m16, "D", color=color, markersize=11 * marker_scale, zorder=5,
            markeredgecolor="white", markeredgewidth=1.0)
    improved = m16 > m0
    ax.text(
        x, check_y,
        "\u2713" if improved else "\u2717",
        color="#2ca02c" if improved else "#d62728",
        fontsize=check_fontsize, fontweight="bold",
        ha="center", va="bottom", zorder=6,
    )


def plot_violin(
    raw_distributions: dict[int, dict[str, dict[str, dict[str, list[float]]]]],
    label_order: list[str],
    model_name: str,
    output_dir: Path,
    dpi: int,
) -> None:
    """Split violin of P(correct) per GT: top half = 0-shot, bottom half = 16-shot."""
    r0 = raw_distributions.get(0, {})
    r16 = raw_distributions.get(16, {})
    gts = sorted(gt for gt in set(r0) & set(r16)
                 if "all" in r0.get(gt, {}) and "all" in r16.get(gt, {}))
    if not gts:
        return

    color = get_model_color(model_name)
    color_0 = mcolors.to_rgba(color, 0.5)
    color_16 = "#ff7f0e"

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(gts) + 2), 4.5))
    for i in range(1, len(gts)):
        ax.axvline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3, zorder=1)

    for idx, gt in enumerate(gts):
        vals0 = _entry_means(r0[gt]["all"])
        vals16 = _entry_means(r16[gt]["all"])
        if len(vals0) < 3 or len(vals16) < 3:
            _violin_dots(ax, vals0, idx, "left", color=color_0, width=0.2)
            _violin_dots(ax, vals16, idx, "right", color=color_16, width=0.2)
        else:
            poly0 = _half_violin(ax, vals0, idx, "left", color_0, width=0.7)
            poly16 = _half_violin(ax, vals16, idx, "right", color_16, width=0.7)
            _violin_dots(ax, vals0, idx, "left", color=color_0, width=0.7, polygon=poly0)
            _violin_dots(ax, vals16, idx, "right", color=color_16, width=0.7, polygon=poly16)
        _motif(ax, vals0, vals16, idx, color=color)

    ax.set_xticks(np.arange(len(gts)))
    ax.set_xticklabels(gts, fontsize=18)
    ax.set_ylabel("P(correct label)", fontsize=19)
    ax.set_title(f"P(correct): {model_name}", fontsize=22)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(0, 1.12)
    ax.grid(True, axis="y", alpha=0.3)

    legend_handles = [
        Line2D([0], [0], marker="s", color=color_0, linestyle="None",
               markersize=12, label="0-shot"),
        Line2D([0], [0], marker="s", color=color_16, linestyle="None",
               markersize=12, label="16-shot"),
        Line2D([0], [0], marker="o", color=color, linestyle="None",
               markersize=10, markeredgecolor="white", markeredgewidth=1.0,
               label="0-shot mean"),
        Line2D([0], [0], marker="D", color=color, linestyle="None",
               markersize=11, markeredgecolor="white", markeredgewidth=1.0,
               label="16-shot mean"),
        Line2D([0], [0], marker="None", linestyle="None", label="\u2713 improved"),
        Line2D([0], [0], marker="None", linestyle="None", label="\u2717 worsened"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=18)

    finish_figure(fig, ax)
    fig.tight_layout(rect=[0, 0.22, 1, 1])
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "violin_pcorrect.png", dpi=dpi)
    plt.close(fig)


def _draw_bars(
    ax,
    gts: list[str],
    r0: dict[str, dict[str, dict[str, list[float]]]],
    r16: dict[str, dict[str, dict[str, list[float]]]],
    alignment: str,
    color_0,
    color_16: str,
    edge_0: str,
) -> None:
    """Draw grouped bars with error bars and value labels on *ax*."""
    bar_width = 0.35
    x = np.arange(len(gts))
    means_0, sems_0, means_16, sems_16 = [], [], [], []
    for gt in gts:
        m0, s0 = _bar_stats(r0[gt][alignment])
        m16, s16 = _bar_stats(r16[gt][alignment])
        means_0.append(m0)
        sems_0.append(s0)
        means_16.append(m16)
        sems_16.append(s16)

    ax.bar(
        x - 0.2, means_0, bar_width, yerr=sems_0, color=color_0,
        edgecolor=edge_0, linewidth=1.0, capsize=4,
        error_kw={"linewidth": 1.2}, label="0-shot", zorder=2,
    )
    ax.bar(
        x + 0.2, means_16, bar_width, yerr=sems_16, color=color_16,
        edgecolor="#cc6600", linewidth=1.0, capsize=4,
        error_kw={"linewidth": 1.2}, label="16-shot", zorder=2,
    )
    for i, (m, s) in enumerate(zip(means_0, sems_0)):
        ax.text(i - 0.2, m + s + 0.02, f"{m:.2f}", ha="center", va="bottom",
                fontsize=9, color="gray")
    for i, (m, s) in enumerate(zip(means_16, sems_16)):
        ax.text(i + 0.2, m + s + 0.02, f"{m:.2f}", ha="center", va="bottom",
                fontsize=9, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(gts, fontsize=14)
    ax.set_xlabel("Answer Labels", fontsize=15)
    ax.set_ylim(0, 1.12)
    ax.grid(True, axis="y", alpha=0.3)


def plot_bar(
    raw_distributions: dict[int, dict[str, dict[str, dict[str, list[float]]]]],
    label_order: list[str],
    model_name: str,
    output_dir: Path,
    dpi: int,
) -> None:
    """Grouped bar chart of P(correct) per GT: 0-shot vs 16-shot."""
    r0 = raw_distributions.get(0, {})
    r16 = raw_distributions.get(16, {})
    gts = sorted(
        gt for gt in set(r0) & set(r16)
        if "all" in r0.get(gt, {}) and "all" in r16.get(gt, {})
    )
    if not gts:
        return

    color = get_model_color(model_name)
    color_0 = mcolors.to_rgba(color, 0.5)
    color_16 = "#ff7f0e"

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(gts) + 2), 6))
    _draw_bars(ax, gts, r0, r16, "all", color_0, color_16, color)
    ax.set_ylabel("P(correct label)", fontsize=15)
    ax.set_title(f"P(correct): {model_name}", fontsize=17)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=13)
    finish_figure(fig, ax)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "bar_pcorrect.png", dpi=dpi)
    plt.close(fig)


def _plot_bar_split(
    raw_distributions: dict[int, dict[str, dict[str, dict[str, list[float]]]]],
    label_order: list[str],
    model_name: str,
    output_dir: Path,
    dpi: int,
) -> None:
    """One bar chart per alignment (real, counterfactual)."""
    r0 = raw_distributions.get(0, {})
    r16 = raw_distributions.get(16, {})
    color = get_model_color(model_name)
    color_0 = mcolors.to_rgba(color, 0.5)
    color_16 = "#ff7f0e"
    output_dir.mkdir(parents=True, exist_ok=True)

    for alignment, tag in (("real", "Real"), ("counterfactual", "Counterfactual")):
        gts = sorted(
            gt for gt in set(r0) & set(r16)
            if alignment in r0.get(gt, {}) and alignment in r16.get(gt, {})
        )
        if not gts:
            continue

        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(gts) + 2), 6))
        _draw_bars(ax, gts, r0, r16, alignment, color_0, color_16, color)
        ax.set_ylabel("P(correct label)", fontsize=15)
        ax.set_title(f"P(correct): {model_name}\n[{tag}]", fontsize=17)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=13)
        finish_figure(fig, ax)
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        fig.savefig(output_dir / f"bar_pcorrect_{alignment}.png", dpi=dpi)
        plt.close(fig)


def _plot_bar_both(
    raw_distributions: dict[int, dict[str, dict[str, dict[str, list[float]]]]],
    model_name: str,
    output_dir: Path,
    dpi: int,
) -> None:
    """Combined bar chart: counterfactual (left) + real (right), shared y-axis."""
    r0 = raw_distributions.get(0, {})
    r16 = raw_distributions.get(16, {})

    gts_by_alignment: dict[str, list[str]] = {}
    for alignment in ("counterfactual", "real"):
        gts_by_alignment[alignment] = sorted(
            gt for gt in set(r0) & set(r16)
            if alignment in r0.get(gt, {}) and alignment in r16.get(gt, {})
        )
    n_cf = len(gts_by_alignment["counterfactual"])
    n_real = len(gts_by_alignment["real"])
    if n_cf == 0 or n_real == 0:
        return

    color = get_model_color(model_name)
    color_0 = mcolors.to_rgba(color, 0.5)
    color_16 = "#ff7f0e"

    total_cols = n_cf + n_real
    width = max(8, 1.2 * total_cols + 2)
    fig, axes = plt.subplots(
        1, 2,
        figsize=(width, 7),
        gridspec_kw={"width_ratios": [n_cf, n_real], "wspace": 0.05},
        sharey=True,
    )

    display_tag = {"counterfactual": "Counterfactual", "real": "Real"}
    for ax, alignment in zip(axes, ("counterfactual", "real")):
        gts = gts_by_alignment[alignment]
        _draw_bars(ax, gts, r0, r16, alignment, color_0, color_16, color)
        ax.set_xlim(-0.5, len(gts) - 0.5)
        ax.set_title(f"[{display_tag[alignment]}]", fontsize=15)

    axes[0].set_ylabel("P(correct label)", fontsize=15)
    axes[0].tick_params(axis="y", labelsize=13)
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    fig.suptitle(f"P(correct): {model_name}", fontsize=17)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center",
        bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=16,
    )

    sns.despine(fig=fig)
    fig.tight_layout(rect=[0, 0.10, 1, 0.94])
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "bar_pcorrect_both.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_violin_split(
    raw_distributions: dict[int, dict[str, dict[str, dict[str, list[float]]]]],
    label_order: list[str],
    model_name: str,
    output_dir: Path,
    dpi: int,
) -> None:
    """One split-violin figure per alignment (real, counterfactual).

    GTs lacking data for a given alignment are dropped from that figure's
    x-axis; the figure width scales with the retained GT count.
    """
    r0 = raw_distributions.get(0, {})
    r16 = raw_distributions.get(16, {})

    color = get_model_color(model_name)
    color_0 = mcolors.to_rgba(color, 0.5)
    color_16 = "#ff7f0e"

    output_dir.mkdir(parents=True, exist_ok=True)

    for alignment, tag in (
        ("real", "Real"),
        ("counterfactual", "Counterfactual"),
    ):
        gts = sorted(
            gt for gt in set(r0) & set(r16)
            if alignment in r0.get(gt, {}) and alignment in r16.get(gt, {})
        )
        if not gts:
            continue

        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(gts) + 2), 4.5))
        for i in range(1, len(gts)):
            ax.axvline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3, zorder=1)

        for idx, gt in enumerate(gts):
            vals0 = _entry_means(r0[gt][alignment])
            vals16 = _entry_means(r16[gt][alignment])
            if len(vals0) < 3 or len(vals16) < 3:
                _violin_dots(ax, vals0, idx, "left", color=color_0, width=0.2)
                _violin_dots(ax, vals16, idx, "right", color=color_16, width=0.2)
            else:
                poly0 = _half_violin(ax, vals0, idx, "left", color_0, width=0.7)
                poly16 = _half_violin(ax, vals16, idx, "right", color_16, width=0.7)
                _violin_dots(ax, vals0, idx, "left", color=color_0, width=0.7, polygon=poly0)
                _violin_dots(ax, vals16, idx, "right", color=color_16, width=0.7, polygon=poly16)
            _motif(ax, vals0, vals16, idx, color=color)

        ax.set_xticks(np.arange(len(gts)))
        ax.set_xticklabels(gts, fontsize=18)
        ax.set_ylabel("P(correct label)", fontsize=19)
        ax.set_title(f"P(correct): {model_name}\n[{tag}]", fontsize=22)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_ylim(0, 1.12)
        ax.grid(True, axis="y", alpha=0.3)

        legend_handles = [
            Line2D([0], [0], marker="s", color=color_0, linestyle="None",
                   markersize=12, label="0-shot"),
            Line2D([0], [0], marker="s", color=color_16, linestyle="None",
                   markersize=12, label="16-shot"),
            Line2D([0], [0], marker="o", color=color, linestyle="None",
                   markersize=10, markeredgecolor="white", markeredgewidth=1.0,
                   label="0-shot mean"),
            Line2D([0], [0], marker="D", color=color, linestyle="None",
                   markersize=11, markeredgecolor="white", markeredgewidth=1.0,
                   label="16-shot mean"),
            Line2D([0], [0], marker="None", linestyle="None", label="\u2713 improved"),
            Line2D([0], [0], marker="None", linestyle="None", label="\u2717 worsened"),
        ]
        fig.legend(
            handles=legend_handles, loc="lower center",
            bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=18,
        )

        finish_figure(fig, ax)
        fig.tight_layout(rect=[0, 0.18, 1, 1])
        fig.savefig(output_dir / f"violin_pcorrect_{alignment}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def _alignment_gts(
    raw_distributions: dict[int, dict[str, dict[str, dict[str, list[float]]]]],
    alignment: str,
) -> list[str]:
    r0 = raw_distributions.get(0, {})
    r16 = raw_distributions.get(16, {})
    return sorted(
        gt for gt in set(r0) & set(r16)
        if alignment in r0.get(gt, {}) and alignment in r16.get(gt, {})
    )


def _draw_violin_alignment_panel(
    ax,
    raw_distributions: dict[int, dict[str, dict[str, dict[str, list[float]]]]],
    alignment: str,
    *,
    color,
    color_0,
    color_16,
    title: str | None = None,
    title_fontsize: int = 19,
    xtick_fontsize: int = 18,
    ytick_fontsize: int = 16,
    dot_size: int = 24,
    motif_linewidth: float = 3.0,
    motif_marker_scale: float = 1.0,
    motif_check_fontsize: int = 18,
) -> list[str]:
    r0 = raw_distributions.get(0, {})
    r16 = raw_distributions.get(16, {})
    gts = _alignment_gts(raw_distributions, alignment)

    for i in range(1, len(gts)):
        ax.axvline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3, zorder=1)

    for idx, gt in enumerate(gts):
        vals0 = _entry_means(r0[gt][alignment])
        vals16 = _entry_means(r16[gt][alignment])
        if len(vals0) < 3 or len(vals16) < 3:
            _violin_dots(ax, vals0, idx, "left", color=color_0, width=0.2, size=dot_size)
            _violin_dots(ax, vals16, idx, "right", color=color_16, width=0.2, size=dot_size)
        else:
            poly0 = _half_violin(ax, vals0, idx, "left", color_0, width=0.7)
            poly16 = _half_violin(ax, vals16, idx, "right", color_16, width=0.7)
            _violin_dots(
                ax, vals0, idx, "left", color=color_0, width=0.7,
                polygon=poly0, size=dot_size,
            )
            _violin_dots(
                ax, vals16, idx, "right", color=color_16, width=0.7,
                polygon=poly16, size=dot_size,
            )
        _motif(
            ax, vals0, vals16, idx, color=color,
            linewidth=motif_linewidth,
            marker_scale=motif_marker_scale,
            check_fontsize=motif_check_fontsize,
        )

    ax.set_xticks(np.arange(len(gts)))
    ax.set_xticklabels(gts, fontsize=xtick_fontsize)
    ax.set_xlim(-0.5, len(gts) - 0.5)
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    ax.set_ylim(0, 1.12)
    ax.tick_params(axis="y", labelsize=ytick_fontsize)
    ax.grid(True, axis="y", alpha=0.3)
    return gts


def _plot_violin_both(
    raw_distributions: dict[int, dict[str, dict[str, dict[str, list[float]]]]],
    model_name: str,
    output_dir: Path,
    dpi: int,
) -> None:
    """Combined figure: counterfactual (left) + real (right), shared y-axis.

    Width ratios match GT counts so per-label horizontal spacing is identical
    across the two subplots.
    """
    r0 = raw_distributions.get(0, {})
    r16 = raw_distributions.get(16, {})

    gts_by_alignment: dict[str, list[str]] = {}
    for alignment in ("counterfactual", "real"):
        gts_by_alignment[alignment] = sorted(
            gt for gt in set(r0) & set(r16)
            if alignment in r0.get(gt, {}) and alignment in r16.get(gt, {})
        )
    n_cf = len(gts_by_alignment["counterfactual"])
    n_real = len(gts_by_alignment["real"])
    if n_cf == 0 or n_real == 0:
        return

    color = get_model_color(model_name)
    color_0 = mcolors.to_rgba(color, 0.5)
    color_16 = "#ff7f0e"

    total_cols = n_cf + n_real
    width = max(8, 1.2 * total_cols + 2)
    fig, axes = plt.subplots(
        1, 2,
        figsize=(width, 4.9),
        gridspec_kw={"width_ratios": [n_cf, n_real], "wspace": 0.05},
        sharey=True,
    )

    display_tag = {"counterfactual": "Counterfactual", "real": "Real"}
    for ax, alignment in zip(axes, ("counterfactual", "real")):
        _draw_violin_alignment_panel(
            ax, raw_distributions, alignment,
            color=color, color_0=color_0, color_16=color_16,
            title=f"[{display_tag[alignment]}]",
        )

    axes[0].set_ylabel("P(correct label)", fontsize=19)
    axes[0].tick_params(axis="y", labelsize=16)
    axes[1].tick_params(axis="y", left=False, labelleft=False)

    fig.suptitle(f"P(correct): {model_name}", fontsize=22)

    legend_handles = [
        Line2D([0], [0], marker="s", color=color_0, linestyle="None",
               markersize=12, label="0-shot"),
        Line2D([0], [0], marker="s", color=color_16, linestyle="None",
               markersize=12, label="16-shot"),
        Line2D([0], [0], marker="o", color=color, linestyle="None",
               markersize=10, markeredgecolor="white", markeredgewidth=1.0,
               label="0-shot mean"),
        Line2D([0], [0], marker="D", color=color, linestyle="None",
               markersize=11, markeredgecolor="white", markeredgewidth=1.0,
               label="16-shot mean"),
        Line2D([0], [0], marker="None", linestyle="None", label="\u2713 improved"),
        Line2D([0], [0], marker="None", linestyle="None", label="\u2717 worsened"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=18)

    sns.despine(fig=fig)
    fig.tight_layout(rect=[0, 0.30, 1, 0.92])
    for ax in axes:
        pos = ax.get_position()
        new_height = pos.height * 0.70
        ax.set_position([
            pos.x0,
            pos.y0 + (pos.height - new_height) * 0.5,
            pos.width,
            new_height,
        ])
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "violin_pcorrect_both.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _resolve_model_name(
    model_dict: dict[str, tuple[Path, list[dict[str, Any]]]],
    candidates: Sequence[str],
) -> str | None:
    for candidate in candidates:
        if candidate in model_dict:
            return candidate
    lowered = {model.lower(): model for model in model_dict}
    for candidate in candidates:
        model = lowered.get(candidate.lower())
        if model is not None:
            return model
    return None


def _resolve_leaf_name(
    leaf_dict: dict[str, dict[str, tuple[Path, list[dict[str, Any]]]]],
    split: str,
) -> str | None:
    if split in leaf_dict:
        return split
    for leaf in sorted(leaf_dict):
        if extract_split_from_leaf(leaf) == split:
            return leaf
    return None


def plot_answer_dist_compare_violin(
    cross_model: dict[str, dict[str, dict[str, tuple[Path, list[dict[str, Any]]]]]],
    output_dir: Path,
    *,
    row_specs: Sequence[tuple[str, str, str]],
    model_specs: Sequence[tuple[str, Sequence[str]]],
    filename: str = "compare_violin.png",
    dpi: int = 150,
) -> None:
    """Comparison grid of compact ``violin_pcorrect_both`` panels.

    ``row_specs`` entries are ``(task_key, split_key, display_title)``.
    ``model_specs`` entries are ``(display_title, acceptable_model_names)``.
    Each model column is internally split into counterfactual and real panels.
    """
    if not row_specs or not model_specs:
        return

    n_rows = len(row_specs)
    n_axes_cols = len(model_specs) * 2
    fig, axes = plt.subplots(
        n_rows,
        n_axes_cols,
        figsize=(4.1 * n_axes_cols, 2.65 * n_rows + 1.15),
        sharey=True,
        squeeze=False,
        gridspec_kw={"wspace": 0.12, "hspace": 0.62},
    )

    display_tag = {"counterfactual": "Counterfactual", "real": "Real"}
    legend_color = "#4c4c4c"
    legend_handles = [
        Line2D([0], [0], marker="s", color=mcolors.to_rgba(legend_color, 0.5),
               linestyle="None", markersize=14, label="0-shot"),
        Line2D([0], [0], marker="s", color="#ff7f0e", linestyle="None",
               markersize=14, label="16-shot"),
        Line2D([0], [0], marker="o", color=legend_color, linestyle="None",
               markersize=13, markeredgecolor="white", markeredgewidth=1.0,
               label="0-shot mean"),
        Line2D([0], [0], marker="D", color=legend_color, linestyle="None",
               markersize=13, markeredgecolor="white", markeredgewidth=1.0,
               label="16-shot mean"),
        Line2D([0], [0], marker="None", linestyle="None", label="\u2713 improved"),
        Line2D([0], [0], marker="None", linestyle="None", label="\u2717 worsened"),
    ]

    missing: list[str] = []
    for row_idx, (task, split, row_title) in enumerate(row_specs):
        leaf_dict = cross_model.get(task, {})
        leaf = _resolve_leaf_name(leaf_dict, split)
        row_axes = axes[row_idx]
        if leaf is None:
            missing.append(f"{task}/{split}")
            for ax in row_axes:
                ax.axis("off")
            continue

        model_dict = leaf_dict[leaf]
        for model_idx, (model_title, candidates) in enumerate(model_specs):
            model_name = _resolve_model_name(model_dict, candidates)
            pair_axes = row_axes[model_idx * 2:model_idx * 2 + 2]
            if model_name is None:
                missing.append(f"{task}/{leaf}: {model_title}")
                for ax in pair_axes:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                continue

            leaf_path, summaries = model_dict[model_name]
            label_order = summaries[0].get("label_order", []) if summaries else []
            raw = load_pcorrect_per_sample(leaf_path, label_order, shot_counts=(0, 16))
            if raw is None or not has_alignment_keys(raw):
                missing.append(f"{task}/{leaf}: {model_name} has no aligned violin data")
                for ax in pair_axes:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                continue

            color = get_model_color(model_name)
            color_0 = mcolors.to_rgba(color, 0.5)
            color_16 = "#ff7f0e"
            for align_idx, (ax, alignment) in enumerate(zip(pair_axes, ("counterfactual", "real"))):
                if not _alignment_gts(raw, alignment):
                    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue
                title = display_tag[alignment] if row_idx == 0 else f"[{display_tag[alignment]}]"
                _draw_violin_alignment_panel(
                    ax, raw, alignment,
                    color=color, color_0=color_0, color_16=color_16,
                    title=title,
                    title_fontsize=12,
                    xtick_fontsize=10,
                    ytick_fontsize=10,
                    dot_size=10,
                    motif_linewidth=1.8,
                    motif_marker_scale=0.65,
                    motif_check_fontsize=11,
                )
                if model_idx > 0 or align_idx > 0:
                    ax.tick_params(axis="y", left=False, labelleft=False)

        row_axes[0].set_ylabel("P(correct label)", fontsize=11)
        row_axes[0].text(
            -0.36, 0.5, row_title,
            transform=row_axes[0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    sns.despine(fig=fig)
    fig.subplots_adjust(
        left=0.08,
        right=0.995,
        bottom=0.19,
        top=0.89,
        wspace=0.12,
        hspace=0.62,
    )

    for model_idx, (model_title, _) in enumerate(model_specs):
        left = axes[0, model_idx * 2].get_position().x0
        right = axes[0, model_idx * 2 + 1].get_position().x1
        fig.text(
            (left + right) / 2,
            0.94,
            model_title,
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=6,
        fontsize=18,
        frameon=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    for item in missing:
        print(f"[answer_dist compare violin] missing: {item}")
    print(f"Saved answer distribution comparison violin to: {output_dir / filename}")


def process_input_dir(
    input_dir: Path,
    output_dir: Path,
    *,
    spec: ExperimentVisSpec,
    dpi: int,
    model_name: str | None = None,
) -> None:
    """Render all per-model plots for one experiment leaf directory."""
    summaries = load_summaries(input_dir)
    if not summaries:
        print(f"No summary files found in {input_dir}")
        return

    if model_name is None:
        model_name = input_dir.parent.name

    if spec.errorbar.enabled:
        attach_variance_to_summaries(input_dir, summaries)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_model_accuracy(summaries, output_dir, spec=spec, dpi=dpi)
    if spec.include_label_probs:
        plot_label_probs(summaries, output_dir, suptitle=spec.label_probs_title, dpi=dpi)
    label_order = summaries[0].get("label_order", [])
    all_shot_counts = tuple(s["n_shots"] for s in summaries)
    write_accuracy_breakdown(
        input_dir,
        model_name,
        output_dir,
        shot_counts=all_shot_counts,
    )
    raw = (
        load_pcorrect_per_sample(input_dir, label_order, shot_counts=all_shot_counts)
        if (spec.include_dumbbell or spec.include_violin or spec.include_bar)
        else None
    )
    if raw is not None:
        write_stats_table(raw, model_name, output_dir, summaries=summaries)
    if spec.include_dumbbell:
        if raw is not None and has_alignment_keys(raw):
            _plot_dumbbell_split(raw, label_order, model_name, output_dir, dpi)
            if extract_split_from_leaf(output_dir.name) == "complex":
                _plot_dumbbell_both(raw, model_name, output_dir, dpi)
        else:
            plot_dumbbell(
                summaries,
                model_name,
                output_dir,
                dpi,
                raw_distributions=raw,
            )
    if spec.include_violin:
        if raw is None:
            print(f"[violin] skipping (no results.jsonl in {input_dir})")
        elif has_alignment_keys(raw):
            _plot_violin_split(raw, label_order, model_name, output_dir, dpi)
            if extract_split_from_leaf(output_dir.name) == "complex":
                _plot_violin_both(raw, model_name, output_dir, dpi)
        else:
            plot_violin(raw, label_order, model_name, output_dir, dpi)
    if spec.include_bar:
        if raw is None:
            print(f"[bar] skipping (no results.jsonl in {input_dir})")
        elif has_alignment_keys(raw):
            _plot_bar_split(raw, label_order, model_name, output_dir, dpi)
            if extract_split_from_leaf(output_dir.name) == "complex":
                _plot_bar_both(raw, model_name, output_dir, dpi)
        else:
            plot_bar(raw, label_order, model_name, output_dir, dpi)
    print(f"Saved plots to: {output_dir}")


def run_experiment_visualization(
    spec: ExperimentVisSpec,
    *,
    input_dir: str | Path | None = None,
    base_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    dpi: int = 150,
) -> None:
    """Shared visualization runner used by the formal experiment wrappers."""
    setup_style()
    output_root = Path(output_dir or spec.output_dir_default)

    if input_dir:
        input_path = Path(input_dir)
        model_output_dir = output_root / input_path.parent.name / input_path.name
        process_input_dir(input_path, model_output_dir, spec=spec, dpi=dpi)
        return

    if not base_dir:
        print("Provide either --input_dir or --base_dir")
        return

    base_path = Path(base_dir)
    cross_model = discover_model_summaries(base_path, leaf_filter=spec.leaf_filter)

    if spec.errorbar.enabled:
        for leaf_dict in cross_model.values():
            for model_dict in leaf_dict.values():
                for leaf_path, summaries in model_dict.values():
                    attach_variance_to_summaries(leaf_path, summaries)

    for task, leaf_dict in cross_model.items():
        for leaf, model_dict in leaf_dict.items():
            for model_name, (leaf_path, _) in model_dict.items():
                model_output_dir = output_root / task / model_name / leaf
                process_input_dir(
                    leaf_path,
                    model_output_dir,
                    spec=spec,
                    dpi=dpi,
                    model_name=model_name,
                )

    for task, leaf_dict in cross_model.items():
        for leaf, model_dict in leaf_dict.items():
            if len(model_dict) <= 1:
                continue
            model_summaries = {model: summaries for model, (_, summaries) in model_dict.items()}
            group_output_dir = output_root / task / leaf
            if spec.include_cross_model:
                plot_accuracy_cross_model(
                    model_summaries,
                    group_output_dir,
                    model_filter=spec.cross_model_filter,
                    task=task,
                    split=leaf,
                    errorbar=spec.errorbar,
                    dpi=dpi,
                )
            if spec.include_per_family:
                plot_accuracy_per_family(
                    model_summaries,
                    group_output_dir,
                    use_family_legend=spec.per_family_use_family_legend,
                    task=task,
                    split=leaf,
                    errorbar=spec.errorbar,
                    dpi=dpi,
                )

    if not spec.include_aggregated:
        return

    if spec.errorbar.enabled:
        aggregated_var = aggregate_across_tasks_with_variance(
            cross_model, spec.aggregated_splits
        )
        per_group_summaries = {
            group: to_summaries_format_with_variance(aggregated_var.get(group, {}))
            for group in ("overall", *spec.aggregated_splits)
        }
    else:
        aggregated = aggregate_across_tasks(cross_model, spec.aggregated_splits)
        per_group_summaries = {
            group: to_summaries_format(aggregated.get(group, {}))
            for group in ("overall", *spec.aggregated_splits)
        }

    for group in ("overall", *spec.aggregated_splits):
        model_summaries = per_group_summaries.get(group, {})
        if not model_summaries:
            print(f"Skipping aggregated {group}: no data")
            continue
        group_dir = output_root / "aggregated" / group
        if spec.include_cross_model:
            plot_accuracy_cross_model(
                model_summaries,
                group_dir,
                model_filter=spec.cross_model_filter,
                split=group.capitalize(),
                errorbar=spec.errorbar,
                dpi=dpi,
            )
        if spec.include_per_family:
            plot_accuracy_per_family(
                model_summaries,
                group_dir,
                split=group.capitalize(),
                use_family_legend=spec.per_family_use_family_legend,
                errorbar=spec.errorbar,
                dpi=dpi,
            )
        print(f"Saved aggregated {group} plots to: {group_dir}")

        if spec.include_compare and group == "overall":
            compare_dir = output_root / "compare"
            plot_accuracy_cross_model(
                model_summaries,
                compare_dir,
                model_filter=spec.cross_model_filter,
                split=group.capitalize(),
                errorbar=spec.errorbar,
                filename="overall_cross_model.png",
                dpi=dpi,
            )
            write_cross_model_accuracy_table(
                model_summaries,
                compare_dir,
                filename="overall_cross_model.txt",
                title="Overall cross-model accuracy vs. ICL shots",
                model_filter=spec.cross_model_filter,
                errorbar=spec.errorbar,
            )
            print(f"Saved compare plot + table to: {compare_dir}")
