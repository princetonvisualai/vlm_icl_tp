"""Control output distribution analysis — fixed demo set, clean vs corrupt, across k values."""

from __future__ import annotations

import argparse
import json
import random
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Literal

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None

from vlm_icl.data import prepare_data
from vlm_icl.prompts import format_icl_answer
from vlm_icl.icl import (
    build_control_demo_set,
    build_icl_context_pools,
    process_icl_context_images,
)
from vlm_icl.paths import build_analysis_path, leaf_parts_prefix
from vlm_icl.metrics import normalize_answer
from vlm_icl.scoring import HFAnswerScorer, VLLMAnswerScorer, build_label_vocab_by_partition
from experiments.extra.answer_distribution import (
    AggregationState,
    iter_queries,
    save_label_order,
    save_by_ground_truth,
    save_overall,
)

PartitionKey = Tuple[str, str]
BraceVariant = Literal["{", "{ "]


@dataclass(frozen=True)
class ControlOutputDistConfig:
    model_path: str
    backend: str = "default"
    tensor_parallel_size: int = 1
    max_num_images: int = 64
    max_model_len: int = 32768
    seed: int = 42
    output_dir: Optional[str] = None
    tasks: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    sample_per_split: int = 20
    enable_thinking: bool = False
    cot: bool = False
    max_demos: int = 32
    k_values: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    vllm_logprobs_k: int = 20
    brace_variant: BraceVariant = "{"
    data_root: Optional[str] = None
    data_source: str = "auto"


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_queries(
    *,
    data: Sequence[Mapping[str, Any]],
    label_vocab: Mapping[PartitionKey, List[str]],
    scorer,
    config: ControlOutputDistConfig,
    fixed_icl_contexts: Optional[Mapping[PartitionKey, List[Dict[str, Any]]]] = None,
) -> Dict[PartitionKey, Dict[str, Any]]:
    """Score all test queries, returning per-partition aggregation + per-query records."""
    prefer_pil = config.backend != "vllm"
    queries = iter_queries(
        data=data,
        label_vocab_by_partition=label_vocab,
        icl=fixed_icl_contexts is not None,
        num_icl_examples=0,
        num_icl_sample_sets=1,
        seed=config.seed,
        special=None,
        cot=config.cot,
        brace_variant=config.brace_variant,
        prefer_pil=prefer_pil,
        fixed_icl_contexts=fixed_icl_contexts,
    )

    results: Dict[PartitionKey, Dict[str, Any]] = {}
    for query in tqdm(queries, desc="Scoring"):
        pk = query.partition_key
        if pk not in results:
            results[pk] = {
                "agg": AggregationState(label_order=list(label_vocab[pk])),
                "per_query": [],
            }
        state = results[pk]
        label_order = state["agg"].label_order

        result = scorer.score_answers(
            image=query.image, prompt=query.prompt,
            labels=label_order, context=query.context,
            **({"logprobs_k": config.vllm_logprobs_k} if isinstance(scorer, VLLMAnswerScorer) else {}),
        )

        logp = np.array([result["label_logprobs"].get(l, float("-inf")) for l in label_order], dtype=np.float32)
        if np.isneginf(logp).all():
            continue
        dist = softmax(logp)

        gt_norm = normalize_answer(query.ground_truth)
        gt_idx = label_order.index(gt_norm) if gt_norm in label_order else None
        p_correct = float(dist[gt_idx]) if gt_idx is not None else 0.0
        entropy = -float(np.sum(dist * np.log(dist + 1e-12)))

        state["agg"].update(gt_norm, dist)
        state["per_query"].append({
            "image_id": query.image_id,
            "gt": gt_norm,
            "p_correct": p_correct,
            "entropy": entropy,
            "distribution": dist.tolist(),
        })

    return results


def _save_k_results(
    base_dir: Path,
    pk: PartitionKey,
    state: Dict[str, Any],
    config: ControlOutputDistConfig,
) -> None:
    """Save aggregation + per-query results for one partition."""
    task, split = pk
    agg: AggregationState = state["agg"]
    per_query: List[Dict] = state["per_query"]

    out = base_dir / split
    out.mkdir(parents=True, exist_ok=True)

    mean_by_gt = agg.compute_mean_by_gt()
    total = sum(agg.count_by_gt.values())
    mean_p_correct = float(np.mean([q["p_correct"] for q in per_query])) if per_query else 0.0
    mean_entropy = float(np.mean([q["entropy"] for q in per_query])) if per_query else 0.0
    std_entropy = float(np.std([q["entropy"] for q in per_query])) if per_query else 0.0

    save_label_order(out, label_order=agg.label_order)
    save_by_ground_truth(out, by_ground_truth={
        gt: {
            "count": int(agg.count_by_gt[gt]),
            "mean_distribution": mean_by_gt[gt].tolist(),
            "mean_p_correct": float(np.mean([
                q["p_correct"] for q in per_query if q["gt"] == gt
            ])) if any(q["gt"] == gt for q in per_query) else 0.0,
            "mean_entropy": float(np.mean([
                q["entropy"] for q in per_query if q["gt"] == gt
            ])) if any(q["gt"] == gt for q in per_query) else 0.0,
            "std_entropy": float(np.std([
                q["entropy"] for q in per_query if q["gt"] == gt
            ])) if any(q["gt"] == gt for q in per_query) else 0.0,
        }
        for gt in agg.sum_by_gt
    })
    save_overall(out, overall={
        "macro_mean_distribution": agg.compute_macro_average().tolist(),
        "mean_p_correct": mean_p_correct,
        "mean_entropy": mean_entropy,
        "std_entropy": std_entropy,
        "n_labels": len(agg.label_order),
        "n_samples": total,
        "partition": {"task": task, "split": split},
    })
    with open(out / "per_query.json", "w") as f:
        json.dump(per_query, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_control_output_dist(config: ControlOutputDistConfig) -> Dict[str, Any]:
    # 1. Load test data
    test_data = prepare_data(
        tasks=config.tasks, topics=config.topics,
        sample_per_split=config.sample_per_split, seed=config.seed,
        data_root=config.data_root,
        data_source=config.data_source,
    )
    # 2. Full data for demo pool
    full_data = prepare_data(tasks=config.tasks, topics=config.topics, seed=config.seed, data_root=config.data_root, data_source=config.data_source)
    context_pools = build_icl_context_pools(full_data)

    # 3. Control demos + corrupted answers per (task, split)
    control_demos, corrupted_answers = build_control_demo_set(
        context_pools, max_demos=config.max_demos, seed=config.seed,
    )

    # 4. Label vocab
    label_vocab = build_label_vocab_by_partition(test_data)

    # 5. Scorer (loaded once)
    scorer_kw = {"model_path": config.model_path, "cot": config.cot, "enable_thinking": config.enable_thinking}
    if config.backend == "vllm":
        scorer = VLLMAnswerScorer(
            **scorer_kw, tensor_parallel_size=config.tensor_parallel_size,
            max_num_images=config.max_num_images, max_model_len=config.max_model_len,
        )
    else:
        scorer = HFAnswerScorer(**scorer_kw)

    # Output base
    leaf_parts = leaf_parts_prefix(config.sample_per_split, "default", config.enable_thinking)
    leaf_parts.append("standard")
    leaf = "_".join(leaf_parts)

    def base_for(task: str) -> Path:
        return build_analysis_path(
            config.output_dir or "./output",
            "control_output_dist", "visual_reasoning", task,
            config.model_path, leaf,
        )

    # Save run config + control demo identifiers
    output_dirs = []
    for key, demos in control_demos.items():
        task, _ = key
        bdir = base_for(task)
        bdir.mkdir(parents=True, exist_ok=True)
        with open(bdir / "run_config.json", "w") as f:
            json.dump(asdict(config), f, indent=2, default=str)
        with open(bdir / "control_demos.json", "w") as f:
            json.dump([
                {"image_id": d["entry"]["image_id"], "gt": d["question"]["ground_truth"]}
                for d in demos
            ], f, indent=2)
        output_dirs.append(str(bdir))

    # 6. k=0 (zero-shot, shared between conditions)
    print("=== k=0 (zero-shot) ===")
    zs_results = _score_queries(
        data=test_data, label_vocab=label_vocab,
        scorer=scorer, config=config,
    )
    for pk, state in zs_results.items():
        task, _ = pk
        for cond in ("clean", "corrupt"):
            _save_k_results(base_for(task) / cond / "k0", pk, state, config)

    # 7. For each condition × k
    for condition in ("clean", "corrupt"):
        for k in config.k_values:
            if k > config.max_demos:
                continue
            print(f"=== {condition} k={k} ===")

            # Build fixed ICL contexts
            fixed_contexts: Dict[PartitionKey, List[Dict[str, Any]]] = {}
            for key, demos in control_demos.items():
                subset = demos[:k]
                if condition == "clean":
                    answers = [d["question"]["ground_truth"] for d in subset]
                else:
                    answers = corrupted_answers[key][:k]
                fixed_contexts[key] = process_icl_context_images(
                    subset, answers, None, format_icl_answer,
                    prefer_pil=config.backend != "vllm",
                )

            results = _score_queries(
                data=test_data, label_vocab=label_vocab,
                scorer=scorer, config=config,
                fixed_icl_contexts=fixed_contexts,
            )
            for pk, state in results.items():
                task, _ = pk
                _save_k_results(base_for(task) / condition / f"k{k}", pk, state, config)

    return {"output_dirs": output_dirs}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> ControlOutputDistConfig:
    from vlm_icl.args import add_model_args, add_data_args, add_run_config_args

    parser = argparse.ArgumentParser(description="Control output distribution analysis.")
    add_model_args(parser)
    add_data_args(parser)
    add_run_config_args(parser)

    parser.add_argument("--max_demos", type=int, default=32)
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--vllm_logprobs_k", type=int, default=20)
    parser.add_argument("--brace_variant", type=str, default="{", choices=["{", "{ "])

    args = parser.parse_args(argv)
    return ControlOutputDistConfig(
        model_path=args.model_path,
        backend=args.backend,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_images=args.max_num_images,
        max_model_len=args.max_model_len,
        seed=args.seed,
        output_dir=args.output_dir,
        tasks=args.tasks,
        topics=args.topics,
        sample_per_split=args.sample_per_split or 20,
        enable_thinking=args.enable_thinking,
        cot=args.cot,
        max_demos=args.max_demos,
        k_values=tuple(args.k_values),
        vllm_logprobs_k=args.vllm_logprobs_k,
        brace_variant=args.brace_variant,
        data_root=args.data_root,
        data_source=args.data_source,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    result = run_control_output_dist(config)
    for d in result["output_dirs"]:
        print(f"Saved to: {d}")


if __name__ == "__main__":
    main()
