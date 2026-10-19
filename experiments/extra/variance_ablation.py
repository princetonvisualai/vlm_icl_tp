"""Variance ablation: thin wrapper around evaluate --num_eval_runs."""

import argparse
from typing import Optional

from experiments.extra.evaluate import evaluate
from vlm_icl.data import TASK_PARTITIONS


def variance_ablation(
    model_path: str,
    backend: str = "default",
    tensor_parallel_size: int = 1,
    max_num_images: int = 64,
    max_model_len: int = 32768,
    icl: bool = False,
    num_icl_examples: int = 2,
    num_icl_sample_sets: int = 1,
    output_dir: str = "./output",
    num_runs: int = 10,
    sample_per_split: int = 100,
    k: int = 1,
    special: Optional[str] = None,
    cot: bool = False,
    query_gt_demo: Optional[str] = None,
    dataset: str = "visual_reasoning",
    enable_thinking: bool = False,
    **data_kwargs,
) -> None:
    """Run evaluation N times with different seeds via evaluate(num_eval_runs=...)."""
    print("Note: variance_ablation now delegates to experiments.extra.evaluate --num_eval_runs")

    evaluate(
        model_path=model_path,
        backend=backend,
        tensor_parallel_size=tensor_parallel_size,
        max_num_images=max_num_images,
        max_model_len=max_model_len,
        icl=icl,
        num_icl_examples=num_icl_examples,
        num_icl_sample_sets=num_icl_sample_sets,
        output_dir=output_dir,
        seed=0,  # ignored in multi-run mode
        k=k,
        special=special,
        cot=cot,
        query_gt_demo=query_gt_demo,
        should_save=False,
        dataset=dataset,
        enable_thinking=enable_thinking,
        num_eval_runs=num_runs,
        sample_per_split=sample_per_split,
        **data_kwargs,
    )


def main():
    """CLI entry point."""
    from vlm_icl.args import add_model_args, add_data_args, add_icl_args, add_run_config_args

    parser = argparse.ArgumentParser(
        description="Variance ablation: evaluate N times with different seeds"
    )
    add_model_args(parser)
    add_data_args(parser)
    add_icl_args(parser)
    add_run_config_args(parser)
    parser.set_defaults(max_model_len=80000, sample_per_split=100)

    # Script-specific
    parser.add_argument("--num_runs", type=int, default=10,
                        help="Number of evaluation runs with different seeds (default: 10)")

    args = parser.parse_args()

    data_kwargs = {}
    if args.topics:
        data_kwargs["topics"] = args.topics
    if args.tasks:
        data_kwargs["tasks"] = args.tasks

    variance_ablation(
        model_path=args.model_path,
        backend=args.backend,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_images=args.max_num_images,
        max_model_len=args.max_model_len,
        icl=args.icl,
        num_icl_examples=args.num_icl_examples,
        num_icl_sample_sets=args.num_icl_sample_sets,
        output_dir=args.output_dir,
        num_runs=args.num_runs,
        sample_per_split=args.sample_per_split,
        k=args.k,
        special=args.special,
        cot=args.cot,
        query_gt_demo=args.query_gt_demo,
        dataset=args.dataset,
        enable_thinking=args.enable_thinking,
        **data_kwargs,
    )

    print("\nVariance ablation completed successfully!")


if __name__ == "__main__":
    main()
