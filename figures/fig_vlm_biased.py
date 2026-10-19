"""Visualization wrapper for formal_vlm_biased.

Sample commands:
    # Single task (single model + task combination):
    uv run python -m figures.fig_vlm_biased \
        --input_dir output/formal_icl_scale/visual_reasoning/counting/qwen3vl_4b/default_complex

    # Auto-discover all tasks from base directory:
    uv run python -m figures.fig_vlm_biased \
        --base_dir output/formal_icl_scale/visual_reasoning

    # Custom output directory and DPI:
    uv run python -m figures.fig_vlm_biased \
        --base_dir output/formal_icl_scale/visual_reasoning \
        --output_dir output_vis/formal_vlm_biased_custom \
        --dpi 300
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from figures._plotting.plots import (
    AccuracySeriesSpec,
    ExperimentVisSpec,
    parse_visualization_args,
    plot_complex_alignment_delta_bars,
    plot_complex_alignment_delta_heatmap,
    plot_complex_alignment_panels,
    run_experiment_visualization,
    write_complex_alignment_delta_heatmap_stats,
    write_complex_alignment_delta_stats,
    write_complex_alignment_stats,
)
from figures._plotting.io import (
    aggregate_complex_alignments_from_results_with_variance,
    aggregate_per_task_with_variance,
    discover_model_summaries,
)


SPEC = ExperimentVisSpec(
    description="Visualize formal_icl_scale experiment results (complex split)",
    output_dir_default="output_vis/formal_vlm_biased",
    results_subdir="formal_icl_scale/visual_reasoning",
    accuracy_series=(
        AccuracySeriesSpec(key="overall", label="Overall"),
        AccuracySeriesSpec(
            key="original",
            label="Original (gt==bias)",
            style="s--",
            linewidth=1.5,
        ),
        AccuracySeriesSpec(
            key="perturbed",
            label="Perturbed (gt!=bias)",
            style="^--",
            linewidth=1.5,
        ),
    ),
    leaf_filter=lambda path: path.name.endswith("_complex"),
    per_family_use_family_legend=False,
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_visualization_args(argv, spec=SPEC)
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
        cross_model = discover_model_summaries(base_path, leaf_filter=SPEC.leaf_filter)
        counts = aggregate_complex_alignments_from_results_with_variance(cross_model)
        qwen3vl_panel_models = [
            "Qwen3-VL-4B-Instruct",
            "Qwen3-VL-8B-Instruct",
            "Qwen3-VL-32B-Instruct",
            "Qwen3-VL-235B-A22B-Instruct",
        ]
        qwen3vl_panel_titles = [
            "Qwen3-VL 4B",
            "Qwen3-VL 8B",
            "Qwen3-VL 32B",
            "Qwen3-VL 235B",
        ]
        plot_complex_alignment_panels(
            counts,
            output_root,
            panel_models=qwen3vl_panel_models,
            panel_titles=qwen3vl_panel_titles,
            filename="qwen3vl_complex_aggregated.png",
            errorbar=replace(SPEC.errorbar, style="none"),
            share_y=False,
            dpi=args.dpi,
        )
        write_complex_alignment_stats(
            counts,
            output_root / "qwen3vl_complex_aggregated.txt",
            panel_models=qwen3vl_panel_models,
            panel_titles=qwen3vl_panel_titles,
        )
        complex_panel_models = [
            "Qwen3-VL-4B-Instruct",
            "Qwen3-VL-32B-Instruct",
            "gemma-4-E4B-it",
            "gemma-4-31B-it",
        ]
        complex_panel_titles = ["Qwen3-VL 4B", "Qwen3-VL 32B", "Gemma4 E4B", "Gemma4 31B"]
        plot_complex_alignment_panels(
            counts,
            output_root,
            panel_models=complex_panel_models,
            panel_titles=complex_panel_titles,
            filename="complex_aggregated.png",
            errorbar=SPEC.errorbar,
            dpi=args.dpi,
        )
        write_complex_alignment_stats(
            counts,
            output_root / "complex_aggregated.txt",
            panel_models=complex_panel_models,
            panel_titles=complex_panel_titles,
        )
        thirtyB_panel_models = [
            "Qwen3-VL-32B-Instruct",
            "Qwen2.5-VL-32B-Instruct",
            "Qwen3.5-27B",
            "InternVL3_5-38B",
            "gemma-4-31B-it",
        ]
        thirtyB_panel_titles = [
            "Qwen3-VL 32B",
            "Qwen2.5-VL 32B",
            "Qwen3.5 27B",
            "InternVL3.5 38B",
            "Gemma4 31B",
        ]
        write_complex_alignment_stats(
            counts,
            output_root / "30B_complex_aggregated.txt",
            panel_models=thirtyB_panel_models,
            panel_titles=thirtyB_panel_titles,
        )

        qwen32b_model = "Qwen3-VL-32B-Instruct"
        qwen32b_task_order = [
            "counting",
            "color_identification",
            "shape_identification",
            "direction",
            "relation",
        ]
        qwen32b_task_titles = [
            "Counting",
            "Color ID",
            "Shape ID",
            "Direction",
            "Relation",
        ]
        per_task_counts = aggregate_per_task_with_variance(
            cross_model,
            ("complex",),
            model_filter=lambda m, _f: m == qwen32b_model,
        )
        plot_complex_alignment_delta_bars(
            per_task_counts,
            output_root,
            model_name=qwen32b_model,
            task_order=qwen32b_task_order,
            task_titles=qwen32b_task_titles,
            filename="task_qwen3vl_complex.png",
            title="Qwen3-VL 32B: ICL accuracy delta (16-shot − 0-shot) per task",
            dpi=args.dpi,
        )
        write_complex_alignment_delta_stats(
            per_task_counts,
            output_root / "task_qwen3vl_complex.txt",
            model_name=qwen32b_model,
            task_order=qwen32b_task_order,
            task_titles=qwen32b_task_titles,
        )

        qwen3vl_heatmap_per_task = aggregate_per_task_with_variance(
            cross_model,
            ("complex",),
            model_filter=lambda m, _f: m in qwen3vl_panel_models,
        )
        qwen3vl_heatmap_titles = ["4B", "8B", "32B", "235B"]
        plot_complex_alignment_delta_heatmap(
            qwen3vl_heatmap_per_task,
            output_root,
            panel_models=qwen3vl_panel_models,
            panel_titles=qwen3vl_heatmap_titles,
            task_order=qwen32b_task_order,
            task_titles=qwen32b_task_titles,
            filename="qwen3vl_complex_tasks.png",
            dpi=args.dpi,
        )
        write_complex_alignment_delta_heatmap_stats(
            qwen3vl_heatmap_per_task,
            output_root / "qwen3vl_complex_tasks.txt",
            panel_models=qwen3vl_panel_models,
            panel_titles=qwen3vl_heatmap_titles,
            task_order=qwen32b_task_order,
            task_titles=qwen32b_task_titles,
        )


if __name__ == "__main__":
    main()
