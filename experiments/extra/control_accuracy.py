"""Control accuracy analysis — fixed demo set, clean vs corrupt, across k values and multiple runs."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from vlm_icl.data import prepare_data
from vlm_icl.models import load_model
from vlm_icl.images import resolve_image
from vlm_icl.prompts import format_icl_answer
from vlm_icl.icl import (
    build_control_demo_set,
    build_icl_context_pools,
    process_icl_context_images,
    sample_per_label,
)
from vlm_icl.paths import build_analysis_path
from vlm_icl.metrics import normalize_answer

PartitionKey = Tuple[str, str]


@dataclass(frozen=True)
class ControlAccuracyConfig:
    model_path: str
    backend: str = "default"
    tensor_parallel_size: int = 1
    max_num_images: int = 64
    max_model_len: int = 32768
    seed: int = 42
    output_dir: Optional[str] = None
    tasks: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    sample_per_label_n: int = 50
    max_demos: int = 32
    k_values: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    num_runs: int = 10
    enable_thinking: bool = False
    cot: bool = False
    data_root: Optional[str] = None
    data_source: str = "auto"


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _evaluate_queries(
    model,
    test_data: List[Dict[str, Any]],
    icl_context_by_pk: Optional[Dict[PartitionKey, List[Dict[str, Any]]]],
    config: ControlAccuracyConfig,
    prefer_pil: bool,
) -> Dict[str, Any]:
    """Generate answers for all test queries and compute accuracy."""
    results_by_gt: Dict[str, Dict[str, int]] = {}
    total = 0
    correct = 0

    for entry in tqdm(test_data, desc="Evaluating", unit="entry"):
        image_obj = entry.get("image")
        image = resolve_image(image_obj, prefer_pil=prefer_pil) if image_obj is not None else None
        extra_imgs = {}
        for extra_key in ("image2", "image3", "image4"):
            extra_obj = entry.get(extra_key)
            if extra_obj is not None:
                extra_imgs[extra_key] = resolve_image(extra_obj, prefer_pil=prefer_pil)

        pk = (entry["task"], entry["split"])
        context = icl_context_by_pk.get(pk) if icl_context_by_pk else None

        for question in entry.get("questions", []):
            gt = normalize_answer(question["ground_truth"])
            try:
                answer = model.generate(
                    image=image,
                    prompt=question["prompt"],
                    context=context,
                    cot=config.cot,
                    **extra_imgs,
                )
            except Exception as e:
                tqdm.write(f"  Error: {e}, skipping image_id={entry['image_id']}")
                continue

            pred = normalize_answer(answer)
            is_correct = int(pred == gt)

            total += 1
            correct += is_correct

            if gt not in results_by_gt:
                results_by_gt[gt] = {"count": 0, "correct": 0}
            results_by_gt[gt]["count"] += 1
            results_by_gt[gt]["correct"] += is_correct

    accuracy = correct / total if total > 0 else 0.0
    by_gt = {
        gt: {**stats, "accuracy": stats["correct"] / stats["count"] if stats["count"] > 0 else 0.0}
        for gt, stats in sorted(results_by_gt.items())
    }
    return {"accuracy": accuracy, "total": total, "correct": correct, "by_ground_truth": by_gt}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_control_accuracy(config: ControlAccuracyConfig) -> Dict[str, Any]:
    # 1. Load full data for demo pool
    full_data = prepare_data(tasks=config.tasks, topics=config.topics, seed=config.seed, data_root=config.data_root, data_source=config.data_source)

    # 2. Sample balanced test queries
    test_data = sample_per_label(full_data, config.sample_per_label_n, seed=config.seed)
    test_ids = {e["image_id"] for e in test_data}
    print(f"Test queries: {len(test_data)} entries ({config.sample_per_label_n} per label)")

    # 3. Build context pools excluding test queries
    pool_data = [e for e in full_data if e["image_id"] not in test_ids]
    context_pools = build_icl_context_pools(pool_data)

    # 4. Control demos + corrupted answers
    control_demos, corrupted_answers = build_control_demo_set(
        context_pools, max_demos=config.max_demos, seed=config.seed,
    )

    # 5. Load model
    if config.backend == "vllm":
        model = load_model(
            config.model_path,
            backend=config.backend,
            tensor_parallel_size=config.tensor_parallel_size,
            max_num_images=config.max_num_images,
            max_model_len=config.max_model_len,
            enable_thinking=config.enable_thinking,
        )
    else:
        model = load_model(config.model_path, enable_thinking=config.enable_thinking)

    prefer_pil = config.backend != "vllm"

    # 6. Output base
    leaf = f"spl{config.sample_per_label_n}_standard"
    output_dirs = []

    tasks_in_data = sorted({e["task"] for e in test_data})
    for task in tasks_in_data:
        base = build_analysis_path(
            config.output_dir or "./output",
            "control_accuracy", "visual_reasoning", task,
            config.model_path, leaf,
        )
        base.mkdir(parents=True, exist_ok=True)
        with open(base / "run_config.json", "w") as f:
            json.dump(asdict(config), f, indent=2, default=str)
        # Save control demo identifiers
        for key, demos in control_demos.items():
            if key[0] == task:
                with open(base / "control_demos.json", "w") as f:
                    json.dump([
                        {"image_id": d["entry"]["image_id"], "gt": d["question"]["ground_truth"]}
                        for d in demos
                    ], f, indent=2)
        output_dirs.append(str(base))

    def base_for(task: str) -> Path:
        return build_analysis_path(
            config.output_dir or "./output",
            "control_accuracy", "visual_reasoning", task,
            config.model_path, leaf,
        )

    # 7. Run experiments
    # Collect per-run accuracies for aggregation: {task: {condition: {k: [acc_per_run]}}}
    all_runs: Dict[str, Dict[str, Dict[int, List[Dict]]]] = {}

    # Pre-split test data by task
    test_by_task: Dict[str, List[Dict]] = {}
    for e in test_data:
        test_by_task.setdefault(e["task"], []).append(e)

    for run_idx in range(config.num_runs):
        run_seed = config.seed + run_idx
        random.seed(run_seed)
        print(f"\n{'='*60}")
        print(f"Run {run_idx}/{config.num_runs} (seed={run_seed})")
        print(f"{'='*60}")

        # k=0 (zero-shot) — shared across conditions
        print(f"  k=0 (zero-shot)")
        for task in tasks_in_data:
            task_zs = _evaluate_queries(model, test_by_task[task], None, config, prefer_pil)
            for condition in ("clean", "corrupt"):
                all_runs.setdefault(task, {}).setdefault(condition, {}).setdefault(0, []).append(task_zs)
                out = base_for(task) / condition / "k0" / f"run{run_idx}"
                out.mkdir(parents=True, exist_ok=True)
                with open(out / "accuracy.json", "w") as f:
                    json.dump(task_zs, f, indent=2)

        # k > 0
        for condition in ("clean", "corrupt"):
            for k in config.k_values:
                if k > config.max_demos:
                    continue
                print(f"  {condition} k={k}")

                fixed_contexts: Dict[PartitionKey, List[Dict[str, Any]]] = {}
                for key, demos in control_demos.items():
                    subset = demos[:k]
                    if condition == "clean":
                        answers = [d["question"]["ground_truth"] for d in subset]
                    else:
                        answers = corrupted_answers[key][:k]
                    fixed_contexts[key] = process_icl_context_images(
                        subset, answers, None, format_icl_answer,
                        prefer_pil=prefer_pil,
                    )

                for task in tasks_in_data:
                    task_results = _evaluate_queries(model, test_by_task[task], fixed_contexts, config, prefer_pil)

                    all_runs.setdefault(task, {}).setdefault(condition, {}).setdefault(k, []).append(task_results)

                    out = base_for(task) / condition / f"k{k}" / f"run{run_idx}"
                    out.mkdir(parents=True, exist_ok=True)
                    with open(out / "accuracy.json", "w") as f:
                        json.dump(task_results, f, indent=2)

    # 8. Aggregate across runs
    for task in tasks_in_data:
        for condition in ("clean", "corrupt"):
            for k_val, run_results in all_runs.get(task, {}).get(condition, {}).items():
                accs = [r["accuracy"] for r in run_results]
                summary = {
                    "k": k_val,
                    "condition": condition,
                    "num_runs": len(accs),
                    "accuracy": {
                        "mean": float(np.mean(accs)),
                        "std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                        "values": accs,
                    },
                }
                out = base_for(task) / condition / f"k{k_val}"
                out.mkdir(parents=True, exist_ok=True)
                with open(out / "accuracy_summary.json", "w") as f:
                    json.dump(summary, f, indent=2)

    return {"output_dirs": output_dirs}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> ControlAccuracyConfig:
    from vlm_icl.args import add_model_args, add_data_args, add_run_config_args

    parser = argparse.ArgumentParser(description="Control accuracy analysis.")
    add_model_args(parser)
    add_data_args(parser)
    add_run_config_args(parser)

    parser.add_argument("--sample_per_label", type=int, default=50)
    parser.add_argument("--max_demos", type=int, default=32)
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--num_runs", type=int, default=10)

    args = parser.parse_args(argv)
    return ControlAccuracyConfig(
        model_path=args.model_path,
        backend=args.backend,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_images=args.max_num_images,
        max_model_len=args.max_model_len,
        seed=args.seed,
        output_dir=args.output_dir,
        tasks=args.tasks,
        topics=args.topics,
        sample_per_label_n=args.sample_per_label,
        enable_thinking=args.enable_thinking,
        cot=args.cot,
        max_demos=args.max_demos,
        k_values=tuple(args.k_values),
        num_runs=args.num_runs,
        data_root=args.data_root,
        data_source=args.data_source,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    result = run_control_accuracy(config)
    for d in result["output_dirs"]:
        print(f"Saved to: {d}")


if __name__ == "__main__":
    main()
