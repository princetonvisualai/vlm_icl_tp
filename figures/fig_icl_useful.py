"""Visualization wrapper for formal_icl_useful.

Sample commands:
    # Single task (single model + task combination):
    uv run python -m figures.fig_icl_useful \
        --input_dir output/formal_icl_scale/visual_reasoning/counting/qwen3vl_4b/default

    # Auto-discover all tasks from the results directory:
    uv run python -m figures.fig_icl_useful --results_dir output

    # Stress-test preset (formal_icl_stresstest results and output dir):
    uv run python -m figures.fig_icl_useful --results_dir output --preset stresstest

    # Custom output directory and DPI:
    uv run python -m figures.fig_icl_useful \
        --base_dir output/formal_icl_scale/visual_reasoning \
        --output_dir output_vis/formal_icl_useful_custom \
        --dpi 300
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Sequence

from dataclasses import replace

from figures._plotting.plots import (
    AccuracySeriesSpec,
    BaselineSpec,
    ExperimentVisSpec,
    parse_visualization_args,
    plot_cross_family_split_grid,
    plot_cross_family_task_grid,
    plot_answer_dist_compare_violin,
    run_experiment_visualization,
)
from figures._plotting.io import (
    discover_model_summaries,
)


_CROSS_FAMILY_BASELINE = BaselineSpec(
    kind="hline",
    y=0.164,
    label="Baseline",
    color="lightgrey",
    linestyle=":",
    linewidth=1.5,
)


def _qwen3vl_filter(model_name: str, family: str) -> bool:
    """Keep only 4B and 32B for Qwen3-VL in cross-model plots."""
    if family == "qwen3vl":
        return any(size in model_name.lower() for size in {"4b", "32b"})
    return True


def _around_30b_filter(model_name: str, family: str) -> bool:
    """Keep only the ~30B variant of each family (size in [25, 40] B)."""
    match = re.search(r"(\d+(?:\.\d+)?)[Bb]", model_name)
    if not match:
        return False
    size = float(match.group(1))
    return 25.0 <= size <= 40.0


_CROSS_FAMILY_LAYOUT: list[list[str]] = [
    ["overall", "direction", "relation"],
    ["counting", "color_identification", "shape_identification"],
]

_CROSS_FAMILY_DISPLAY_NAMES: dict[str, str] = {
    "overall": "Overall",
    "direction": "Object Direction",
    "relation": "Spatial Relation",
    "counting": "Counting",
    "color_identification": "Color",
    "shape_identification": "Shape",
}


_CROSS_FAMILY_SPLIT_LAYOUT: list[list[str]] = [
    ["overall", "complex", "simple", "noprior"],
]

_CROSS_FAMILY_SPLIT_DISPLAY_NAMES: dict[str, str] = {
    "overall": "Overall",
    "complex": "Complex",
    "simple": "Simple",
    "noprior": "No-prior",
}


_ANSWER_DIST_COMPARE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("color_identification", "complex", "Color Identification"),
    ("counting", "complex", "Counting"),
)


_ANSWER_DIST_COMPARE_MODELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Qwen3-VL-32B-Instruct", ("Qwen3-VL-32B-Instruct",)),
    ("Gemma4-31B", ("gemma-4-31B-it", "Gemma4-31B", "Gemma4-31B-it")),
)


SPEC = ExperimentVisSpec(
    description="Visualize formal_icl_scale experiment results (simple/noprior)",
    output_dir_default="output_vis/formal_icl_useful",
    results_subdir="formal_icl_scale/visual_reasoning",
    accuracy_series=(AccuracySeriesSpec(key="overall", label="Overall"),),
    cross_model_filter=_qwen3vl_filter,
    per_family_use_family_legend=False,
    include_bar=True,
)


STRESSTEST_SPEC = replace(
    SPEC,
    description="Visualize formal_icl_stresstest experiment results",
    output_dir_default="output_vis/formal_icl_stresstest",
    results_subdir="formal_icl_stresstest/visual_reasoning_stresstest",
)

PRESETS = {"default": SPEC, "stresstest": STRESSTEST_SPEC}


def _add_preset_arg(parser) -> None:
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="default",
        help="'stresstest' reads <results_dir>/formal_icl_stresstest/visual_reasoning_stresstest "
        "and writes to output_vis/formal_icl_stresstest",
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    pre = argparse.ArgumentParser(add_help=False)
    _add_preset_arg(pre)
    preset = pre.parse_known_args(argv)[0].preset
    SPEC = PRESETS[preset]
    args = parse_visualization_args(argv, spec=SPEC, configure=_add_preset_arg)
    run_experiment_visualization(
        SPEC,
        input_dir=args.input_dir,
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    if args.base_dir:
        base_path = Path(args.base_dir)
        output_root = Path(args.output_dir or SPEC.output_dir_default)
        cross_model = discover_model_summaries(base_path)
        across_task_eb = replace(SPEC.errorbar, style="band")
        plot_cross_family_task_grid(
            cross_model,
            output_root,
            layout=_CROSS_FAMILY_LAYOUT,
            display_names=_CROSS_FAMILY_DISPLAY_NAMES,
            target_splits=("simple", "noprior", "complex"),
            model_filter=_around_30b_filter,
            filename="cross_family_30b_grid.png",
            errorbar=across_task_eb,
            baseline=_CROSS_FAMILY_BASELINE,
            dpi=args.dpi,
        )
        for split in ("simple", "noprior", "complex"):
            plot_cross_family_task_grid(
                cross_model,
                output_root,
                layout=_CROSS_FAMILY_LAYOUT,
                display_names=_CROSS_FAMILY_DISPLAY_NAMES,
                target_splits=(split,),
                model_filter=_around_30b_filter,
                filename=f"cross_family_30b_grid_{split}.png",
                errorbar=across_task_eb,
                baseline=_CROSS_FAMILY_BASELINE,
                dpi=args.dpi,
            )
        plot_cross_family_split_grid(
            cross_model,
            output_root,
            layout=_CROSS_FAMILY_SPLIT_LAYOUT,
            display_names=_CROSS_FAMILY_SPLIT_DISPLAY_NAMES,
            target_splits=("simple", "noprior", "complex"),
            model_filter=_around_30b_filter,
            filename="cross_family_30b_grid_2.png",
            legend_fontsize=20,
            errorbar=across_task_eb,
            baseline=_CROSS_FAMILY_BASELINE,
            disable_variance=True,
            dpi=args.dpi,
        )
        plot_answer_dist_compare_violin(
            cross_model,
            output_root / "answer_dist",
            row_specs=_ANSWER_DIST_COMPARE_ROWS,
            model_specs=_ANSWER_DIST_COMPARE_MODELS,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
