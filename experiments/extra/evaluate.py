"""VLM evaluation: runs inference then computes and saves evaluation metrics."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional

from vlm_icl.evaluation import save_metrics, save_results, display_results
from vlm_icl.metrics import (
    compute_metrics, augment_metrics_with_zero_variance, aggregate_multi_run_metrics,
)
from vlm_icl.icl import validate_special_mode
from vlm_icl.paths import build_output_path, derive_run_config
from vlm_icl.models import load_model
from experiments.extra.inference import inference, cleanup_model


def _single_run_evaluate(
    model_path: str,
    backend: str,
    tensor_parallel_size: int,
    max_num_images: int,
    max_model_len: int,
    icl: bool,
    num_icl_examples: int,
    num_icl_sample_sets: int,
    seed: int,
    k: int,
    special: Optional[str],
    cot: bool,
    query_gt_demo: Optional[str],
    demo_pool: Optional[Dict],
    test: bool,
    dataset: str,
    enable_thinking: bool = False,
    data_root: Optional[str] = None,
    data_source: str = "auto",
    model=None,
    **data_kwargs,
) -> Dict[str, Dict[str, Any]]:
    """Run a single evaluation pass and return per-task results plus metrics."""
    random.seed(seed)

    max_samples = 10 if test else None

    results = inference(
        model_path=model_path,
        backend=backend,
        tensor_parallel_size=tensor_parallel_size,
        max_num_images=max_num_images,
        max_model_len=max_model_len,
        seed=seed,
        output_dir=None,
        icl=icl,
        num_icl_examples=num_icl_examples,
        num_icl_sample_sets=num_icl_sample_sets,
        max_samples=max_samples,
        k=k,
        special=special,
        cot=cot,
        query_gt_demo=query_gt_demo,
        demo_pool=demo_pool,
        dataset=dataset,
        enable_thinking=enable_thinking,
        data_root=data_root,
        data_source=data_source,
        model=model,
        **data_kwargs,
    )

    for result in results:
        result['icl_mode'] = icl

    if len(results) == 0:
        raise ValueError("No results generated! All samples were skipped.")

    results_by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        results_by_task[result['task']].append(result)

    return {
        task_name: {
            "results": task_results,
            "metrics": compute_metrics(task_results, k=k),
        }
        for task_name, task_results in results_by_task.items()
    }


def evaluate(
    model_path: str,
    backend: str = "default",
    tensor_parallel_size: int = 1,
    max_num_images: int = 64,
    max_model_len: int = 32768,
    icl: bool = False,
    num_icl_examples: int = 2,
    num_icl_sample_sets: int = 1,
    output_dir: str = "./evaluation_results",
    seed: int = 42,
    test: bool = False,
    k: int = 4,
    special: Optional[str] = None,
    cot: bool = False,
    query_gt_demo: Optional[str] = None,
    demo_pool: Optional[Dict] = None,
    should_save: bool = True,
    dataset: str = "visual_reasoning",
    num_eval_runs: int = 1,
    enable_thinking: bool = False,
    data_root: Optional[str] = None,
    data_source: str = "auto",
    model=None,
    **data_kwargs
) -> Dict[str, Any]:
    """Run full evaluation on all samples with optional ICL. Returns metrics dict."""
    validate_special_mode(special, icl, num_icl_examples)

    samples_note = f" (pass@{k})" if k > 1 else ""
    print(f"\nRunning {'ICL' if icl else 'standard'} evaluation{samples_note}...")

    common_kwargs = dict(
        model_path=model_path,
        backend=backend,
        tensor_parallel_size=tensor_parallel_size,
        max_num_images=max_num_images,
        max_model_len=max_model_len,
        icl=icl,
        num_icl_examples=num_icl_examples,
        num_icl_sample_sets=num_icl_sample_sets,
        k=k,
        special=special,
        cot=cot,
        query_gt_demo=query_gt_demo,
        demo_pool=demo_pool,
        test=test,
        dataset=dataset,
        enable_thinking=enable_thinking,
        data_root=data_root,
        data_source=data_source,
        **data_kwargs,
    )

    run_config = derive_run_config(special, cot)

    if num_eval_runs == 1:
        # --- Single run (default): current behavior + zero-variance fields ---
        task_data = _single_run_evaluate(seed=seed, model=model, **common_kwargs)

        all_metrics = {}
        for task_name, task_payload in task_data.items():
            task_results = task_payload["results"]
            metrics = augment_metrics_with_zero_variance(task_payload["metrics"])
            display_results(metrics)

            if should_save:
                run_output_dir = build_output_path(
                    base_dir=output_dir,
                    analysis_type="evaluation",
                    model_path=model_path,
                    task_name=task_name,
                    icl=icl,
                    num_icl_examples=num_icl_examples if icl else 0,
                    run_config=run_config,
                    dataset=dataset,
                    query_gt_demo=query_gt_demo,
                    enable_thinking=enable_thinking,
                )

                print(f"\nSaving results for task: {task_name}")
                save_results(
                    task_results, run_output_dir, model_path=model_path,
                    extra_metadata={"icl_mode": icl, "seed": seed, "k": k, "special_mode": special, "data_filters": data_kwargs},
                )
                save_metrics(metrics, run_output_dir / "metrics.json")
                print(f"Task '{task_name}' results saved to: {run_output_dir}")

            all_metrics[task_name] = metrics

        print(f"\nEvaluation complete! {len(task_data)} task(s) processed.")
        return all_metrics

    # --- Multi-run (num_eval_runs > 1) ---
    if 'sample_per_split' not in data_kwargs:
        print(f"Warning: Running {num_eval_runs} evaluation runs on the full dataset. "
              "Consider using --sample_per_split to reduce cost.")

    # Load model once for all runs
    owns_model = model is None
    if owns_model:
        model_kwargs = {}
        if backend == "vllm":
            model_kwargs['tensor_parallel_size'] = tensor_parallel_size
            model_kwargs['max_num_images'] = max_num_images
            model_kwargs['max_model_len'] = max_model_len
        model = load_model(model_path, backend=backend, enable_thinking=enable_thinking, **model_kwargs)
        print(f"Model loaded once for {num_eval_runs} runs (Type: {type(model).__name__})")

    seeds = list(range(num_eval_runs))
    all_runs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    run_seeds: Dict[str, List[int]] = defaultdict(list)

    try:
        for run_idx, run_seed in enumerate(seeds):
            print(f"\n{'=' * 72}")
            print(f"Run {run_idx + 1}/{num_eval_runs} (seed={run_seed})")
            print(f"{'=' * 72}")

            task_data = _single_run_evaluate(seed=run_seed, model=model, **common_kwargs)
            for task_name, task_payload in task_data.items():
                metrics = task_payload["metrics"]
                display_results(metrics)
                all_runs[task_name].append(metrics)
                run_seeds[task_name].append(run_seed)

        # Aggregate and save per task
        all_metrics = {}
        for task_name, per_run_list in all_runs.items():
            aggregated = aggregate_multi_run_metrics(per_run_list)

            run_output_dir = build_output_path(
                base_dir=output_dir,
                analysis_type="evaluation",
                model_path=model_path,
                task_name=task_name,
                icl=icl,
                num_icl_examples=num_icl_examples if icl else 0,
                run_config=run_config,
                dataset=dataset,
                query_gt_demo=query_gt_demo,
                stable=True,
                sample_per_split=data_kwargs.get('sample_per_split'),
                enable_thinking=enable_thinking,
            )

            # Embed per-run metrics
            aggregated["per_run_metrics"] = [
                {"seed": run_seeds[task_name][i], **m}
                for i, m in enumerate(per_run_list)
            ]

            extra_metadata = {
                "num_eval_runs": num_eval_runs,
                "seeds": run_seeds[task_name],
                "sample_per_split": data_kwargs.get("sample_per_split"),
            }
            save_metrics(aggregated, run_output_dir / "metrics.json", extra_metadata=extra_metadata)

            # Print variance summary
            overall = aggregated["overall"]
            print(f"\n{'=' * 72}")
            print(f"Variance Summary — {task_name}")
            print(f"{'=' * 72}")
            print(f"  Runs: {num_eval_runs}")
            print(f"  Accuracy:  {overall['accuracy_mean']*100:.2f}% ± {overall['accuracy_std']*100:.2f}%")
            print(f"  Bias Rate: {overall['bias_rate_mean']*100:.2f}% ± {overall['bias_rate_std']*100:.2f}%")
            for split, split_stats in aggregated.get("by_task_split", {}).get(task_name, {}).items():
                if "accuracy_mean" in split_stats:
                    print(f"  {split:>10s} acc: {split_stats['accuracy_mean']*100:.2f}% ± {split_stats['accuracy_std']*100:.2f}%")
            print(f"  Saved to: {run_output_dir / 'metrics.json'}")

            all_metrics[task_name] = aggregated

        print(f"\nEvaluation complete! {len(all_runs)} task(s) processed across {num_eval_runs} runs.")
        return all_metrics
    finally:
        if owns_model:
            cleanup_model(model)


def main():
    """CLI entry point."""
    from vlm_icl.args import add_model_args, add_data_args, add_icl_args, add_run_config_args, add_eval_args

    parser = argparse.ArgumentParser(
        description="Comprehensive VLM evaluation with optional ICL"
    )
    add_model_args(parser)
    add_data_args(parser)
    add_icl_args(parser)
    add_run_config_args(parser)
    add_eval_args(parser)
    parser.set_defaults(max_model_len=80000)

    # Script-specific
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick sanity evaluation on 10 samples"
    )

    args = parser.parse_args()

    # Prepare data kwargs
    data_kwargs = {}
    if args.topics:
        data_kwargs['topics'] = args.topics
    if args.tasks:
        data_kwargs['tasks'] = args.tasks
    if args.sample_per_split:
        data_kwargs['sample_per_split'] = args.sample_per_split

    # Run evaluation
    metrics = evaluate(
        model_path=args.model_path,
        backend=args.backend,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_images=args.max_num_images,
        max_model_len=args.max_model_len,
        icl=args.icl,
        num_icl_examples=args.num_icl_examples,
        num_icl_sample_sets=args.num_icl_sample_sets,
        output_dir=args.output_dir,
        seed=args.seed,
        test=args.test,
        k=args.k,
        special=args.special,
        cot=args.cot,
        query_gt_demo=args.query_gt_demo,
        dataset=args.dataset,
        num_eval_runs=args.num_eval_runs,
        enable_thinking=args.enable_thinking,
        data_root=args.data_root,
        data_source=args.data_source,
        **data_kwargs
    )

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
