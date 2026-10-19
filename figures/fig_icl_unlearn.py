"""Visualization wrapper for formal_icl_unlearn.

Sample commands:
    # Single task (single model + task combination):
    uv run python -m figures.fig_icl_unlearn \
        --input_dir output/formal_icl_unlearn/visual_reasoning/counting/qwen3vl_4b/default

    # Auto-discover all tasks from base directory:
    uv run python -m figures.fig_icl_unlearn \
        --base_dir output/formal_icl_unlearn/visual_reasoning

    # Custom output directory and DPI:
    uv run python -m figures.fig_icl_unlearn \
        --base_dir output/formal_icl_unlearn/visual_reasoning \
        --output_dir output_vis/formal_icl_unlearn_custom \
        --dpi 300
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from figures._plotting.plots import (
    AccuracySeriesSpec,
    BaselineSpec,
    ExperimentVisSpec,
    parse_visualization_args,
    run_experiment_visualization,
)


def _corrupt_icl_baselines(summaries: list[dict[str, Any]]) -> list[BaselineSpec]:
    """Uniform and zero-shot baselines used by the corrupt ICL plot."""
    n_labels = len(summaries[0]["label_order"])
    baselines = [
        BaselineSpec(
            kind="hline",
            y=1.0 / n_labels,
            label=f"Uniform ({1.0 / n_labels:.2f})",
            color="gray",
        )
    ]

    zero_shot_summary = next((s for s in summaries if s["n_shots"] == 0), None)
    if zero_shot_summary is None:
        return baselines

    zero_shot_acc = zero_shot_summary["accuracy"]["overall"]
    baselines.append(
        BaselineSpec(
            kind="hline",
            y=zero_shot_acc,
            label=f"Zero-shot ({zero_shot_acc:.2f})",
            color="orange",
        )
    )
    baselines.append(
        BaselineSpec(
            kind="point",
            x=0,
            y=zero_shot_acc,
            color="tab:blue",
            label=None,
            zorder=5,
        )
    )
    return baselines


SPEC = ExperimentVisSpec(
    description="Visualize formal_icl_unlearn experiment results",
    output_dir_default="output_vis/formal_icl_unlearn",
    results_subdir="formal_icl_unlearn/visual_reasoning",
    accuracy_title="Accuracy vs. Corrupt ICL Shots",
    label_probs_title="Per-label Output Probability vs. Corrupt ICL Shots",
    accuracy_series=(
        AccuracySeriesSpec(
            key="overall",
            label="Accuracy (corrupt ICL)",
            color="tab:blue",
            shot_filter=lambda n_shots: n_shots >= 1,
        ),
    ),
    accuracy_baselines_factory=_corrupt_icl_baselines,
    include_aggregated=True,
    include_compare=True,
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


if __name__ == "__main__":
    main()
