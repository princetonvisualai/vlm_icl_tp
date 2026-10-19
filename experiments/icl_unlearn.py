"""Formal ICL Unlearn experiment: measure whether corrupting ICL demo labels shifts output toward uniform."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.special import softmax
from tqdm import tqdm

from vlm_icl.data import prepare_data
from vlm_icl.images import resolve_image
from vlm_icl.prompts import format_icl_answer
from vlm_icl.icl import (
    build_demo_pool,
    build_icl_context_pools,
    process_icl_context_images,
    sample_per_label,
    select_from_demo_pool,
    shuffle_icl_answers,
)
from vlm_icl.paths import build_output_path, derive_run_config
from vlm_icl.metrics import cross_entropy, normalize_answer, one_hot
from vlm_icl.args import add_model_args, add_data_args, add_run_config_args
from vlm_icl.scoring import (
    VLLMAnswerScorer,
    HFAnswerScorer,
    build_label_vocab_by_partition,
)

TARGET_SPLITS = ("simple", "complex")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal ICL Unlearn: corrupted-label ICL effect on simple split")
    add_model_args(parser)
    add_data_args(parser)
    add_run_config_args(parser)

    parser.add_argument("--shot_counts", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32])
    parser.add_argument("--n_per_label", type=int, default=40)
    parser.add_argument("--num_demo_sets", type=int, default=10)
    parser.add_argument("--splits", type=str, nargs="+", default=None,
                        help="Splits to run (default: all of simple, complex, no-prior)")
    parser.add_argument("--vllm_logprobs_k", type=int, default=20)

    args = parser.parse_args(argv)
    if args.tasks is None:
        args.tasks = ["counting", "shape_identification", "color_identification", "direction", "relation"]
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    all_data = prepare_data(tasks=args.tasks, dataset="visual_reasoning", seed=args.seed, data_root=args.data_root, data_source=args.data_source)
    splits = tuple(args.splits) if args.splits else TARGET_SPLITS
    filtered_data = [e for e in all_data if e["split"] in splits]

    query_data = sample_per_label(filtered_data, n_per_label=args.n_per_label, seed=args.seed)
    context_pools = build_icl_context_pools(filtered_data)
    demo_pool = build_demo_pool(context_pools, seed=args.seed)
    label_vocab = build_label_vocab_by_partition(filtered_data, include_expected_bias=False)

    prefer_pil = args.backend != "vllm"
    if args.backend == "vllm":
        scorer = VLLMAnswerScorer(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_num_images=args.max_num_images,
            max_model_len=args.max_model_len,
        )
    else:
        scorer = HFAnswerScorer(model_path=args.model_path)

    run_config = derive_run_config(args.special, args.cot)

    for task in args.tasks:
        for split in splits:
            partition_key = (task, split)
            if partition_key not in label_vocab:
                print(f"No label vocab for {partition_key}, skipping")
                continue
            label_order = label_vocab[partition_key]
            uniform_dist = np.ones(len(label_order)) / len(label_order)

            leaf = f"npl{args.n_per_label}_{split}"
            out_base = build_output_path(
                base_dir=args.output_dir,
                analysis_type="formal_icl_unlearn",
                model_path=args.model_path,
                task_name=task,
                run_config=run_config,
                stable=True,
            )
            out_dir = out_base.parent / leaf
            out_dir.mkdir(parents=True, exist_ok=True)

            with open(out_dir / "run_config.json", "w") as f:
                json.dump(vars(args), f, indent=2, default=str)

            task_queries = [e for e in query_data if e["task"] == task and e["split"] == split]

            for n_shots in args.shot_counts:
                records: List[Dict[str, Any]] = []
                desc = f"{task}/{split} shots={n_shots}"

                for entry in tqdm(task_queries, desc=desc):
                    question = entry["questions"][0]
                    gt = normalize_answer(question["ground_truth"])
                    image_obj = entry["image"]
                    image = resolve_image(image_obj, prefer_pil=prefer_pil) if image_obj is not None else None

                    n_sets = args.num_demo_sets if n_shots > 0 else 1

                    for demo_idx in range(n_sets):
                        icl_context = None
                        if n_shots > 0:
                            seed_i = (args.seed + demo_idx) if args.seed is not None else None
                            context_samples = select_from_demo_pool(
                                entry, question, demo_pool, n_shots, seed=seed_i,
                            )
                            if context_samples is None:
                                continue
                            shuffle_rng = random.Random(seed_i + 1) if seed_i is not None else random.Random()
                            shuffled_answers = shuffle_icl_answers(
                                context_samples, "random_icd", shuffle_rng,
                            )
                            icl_context = process_icl_context_images(
                                context_samples, shuffled_answers, args.special,
                                format_icl_answer, prefer_pil=prefer_pil,
                            )

                        result = scorer.score_answers(
                            image=image, prompt=question["prompt"],
                            labels=label_order, context=icl_context,
                            **({"logprobs_k": args.vllm_logprobs_k} if isinstance(scorer, VLLMAnswerScorer) else {}),
                        )

                        predicted_label = result["predicted_label"]
                        logp_vec = np.array([result["label_logprobs"].get(l, float("-inf")) for l in label_order], dtype=np.float32)
                        if np.isneginf(logp_vec).all():
                            continue
                        dist = softmax(logp_vec).astype(np.float64)
                        gt_onehot = one_hot(gt, label_order)
                        ce_to_gt = cross_entropy(gt_onehot, dist)
                        ce_to_unif = cross_entropy(uniform_dist, dist)

                        records.append({
                            "image_id": entry["image_id"],
                            "task": task,
                            "split": split,
                            "ground_truth": gt,
                            "n_shots": n_shots,
                            "demo_set_idx": demo_idx,
                            "predicted_label": predicted_label,
                            "correct": predicted_label == gt,
                            "distribution": dist.tolist(),
                            "ce_to_gt": ce_to_gt,
                            "ce_to_unif": ce_to_unif,
                        })

                shot_dir = out_dir / f"shots_{n_shots}"
                shot_dir.mkdir(parents=True, exist_ok=True)

                with open(shot_dir / "results.jsonl", "w") as f:
                    for r in records:
                        f.write(json.dumps(r) + "\n")

                n_samples = len(records)
                if n_samples == 0:
                    continue

                correct_all = [r["correct"] for r in records]
                valid_ce_gt = [r["ce_to_gt"] for r in records]
                valid_ce_unif = [r["ce_to_unif"] for r in records]

                dist_by_gt: Dict[str, List[List[float]]] = {}
                for r in records:
                    dist_by_gt.setdefault(r["ground_truth"], []).append(r["distribution"])
                per_label_mean = {
                    gt_label: np.mean(dists, axis=0).tolist() for gt_label, dists in dist_by_gt.items()
                }

                summary = {
                    "n_shots": n_shots,
                    "n_samples": n_samples,
                    "accuracy": {"overall": sum(correct_all) / len(correct_all) if correct_all else 0},
                    "mean_ce_to_gt": float(np.mean(valid_ce_gt)) if valid_ce_gt else None,
                    "mean_ce_to_unif": float(np.mean(valid_ce_unif)) if valid_ce_unif else None,
                    "per_label_mean_distribution": per_label_mean,
                    "label_order": label_order,
                }

                with open(shot_dir / "summary.json", "w") as f:
                    json.dump(summary, f, indent=2)

                print(f"  {task}/{split} shots={n_shots}: acc={summary['accuracy']['overall']:.3f}")

            print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
