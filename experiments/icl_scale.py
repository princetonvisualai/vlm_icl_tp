"""Formal ICL Scale: unified ICL scaling experiment across simple, noprior, and complex splits.

``--preset stresstest`` runs the same experiment on ``visual_reasoning_stresstest``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
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

ALL_SPLITS = ("simple", "noprior", "complex")

STRESSTEST_DEFAULT_TASKS = ["counting", "color_identification", "shape_identification", "direction", "relation"]

# Each preset supplies defaults for the flags it names; flags given on the command line win.
PRESETS = {
    "stresstest": {
        "--dataset": ["visual_reasoning_stresstest"],
        "--analysis_type": ["formal_icl_stresstest"],
        "--tasks": STRESSTEST_DEFAULT_TASKS,
    },
}


def apply_preset(argv: Sequence[str]) -> List[str]:
    """Append the defaults of the ``--preset`` named in ``argv`` for every flag it leaves unset."""
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--preset", choices=sorted(PRESETS))
    known, _ = pre.parse_known_args(argv)
    argv = list(argv)
    if known.preset is not None:
        for flag, values in PRESETS[known.preset].items():
            if not any(a == flag or a.startswith(flag + "=") for a in argv):
                argv += [flag, *values]
    return argv


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal ICL Scale: unified ICL scaling across all splits")
    add_model_args(parser)
    add_data_args(parser)
    add_run_config_args(parser)

    parser.add_argument("--preset", choices=sorted(PRESETS), default=None,
                        help="stresstest: run on visual_reasoning_stresstest and write under "
                             "formal_icl_stresstest/ (dataset, analysis_type and tasks default accordingly)")

    parser.add_argument("--shot_counts", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32])
    parser.add_argument("--splits", type=str, nargs="+", default=None,
                        help="Splits to run (default: all of simple, noprior, complex)")
    parser.add_argument("--n_per_label", type=int, default=40)
    parser.add_argument("--num_demo_sets", type=int, default=10)
    parser.add_argument("--vllm_logprobs_k", type=int, default=20)
    parser.add_argument("--analysis_type", type=str, default="formal_icl_scale")

    args = parser.parse_args(apply_preset(sys.argv[1:] if argv is None else argv))
    if args.tasks is None:
        args.tasks = ["counting", "shape_identification", "color_identification", "direction", "relation"]
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    all_data = prepare_data(tasks=args.tasks, dataset=args.dataset, seed=args.seed, data_root=args.data_root, data_source=args.data_source)

    splits = tuple(args.splits) if args.splits else ALL_SPLITS

    # Per-split data and resources
    split_data = {s: [e for e in all_data if e["split"] == s] for s in ALL_SPLITS}
    split_queries = {s: sample_per_label(split_data[s], n_per_label=args.n_per_label, seed=args.seed) for s in ALL_SPLITS if split_data[s]}
    split_pools = {}
    split_demo_pools = {}
    for s in ALL_SPLITS:
        if split_data[s]:
            split_pools[s] = build_icl_context_pools(split_data[s])
            split_demo_pools[s] = build_demo_pool(split_pools[s], seed=args.seed)

    # Label vocabs: include_expected_bias=True for complex, False for simple/noprior
    label_vocab_simple = build_label_vocab_by_partition(
        [e for e in all_data if e["split"] in ("simple", "noprior")], include_expected_bias=False
    )
    label_vocab_complex = build_label_vocab_by_partition(split_data["complex"], include_expected_bias=True) if split_data["complex"] else {}

    # Scorer
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
            if split not in split_queries or not split_data[split]:
                continue

            partition_key = (task, split)
            lv = label_vocab_complex if split == "complex" else label_vocab_simple
            if partition_key not in lv:
                print(f"No label vocab for {partition_key}, skipping")
                continue
            label_order = lv[partition_key]
            is_complex = split == "complex"

            leaf = f"npl{args.n_per_label}_{split}"
            out_base = build_output_path(
                base_dir=args.output_dir,
                analysis_type=args.analysis_type,
                dataset=args.dataset,
                model_path=args.model_path,
                task_name=task,
                run_config=run_config,
                stable=True,
            )
            out_dir = out_base.parent / leaf
            out_dir.mkdir(parents=True, exist_ok=True)

            with open(out_dir / "run_config.json", "w") as f:
                json.dump(vars(args), f, indent=2, default=str)

            task_queries = [e for e in split_queries[split] if e["task"] == task]
            demo_pool = split_demo_pools.get(split)

            for n_shots in args.shot_counts:
                records: List[Dict[str, Any]] = []
                desc = f"{task}/{split} shots={n_shots}"

                for entry in tqdm(task_queries, desc=desc):
                    question = entry["questions"][0]
                    gt = normalize_answer(question["ground_truth"])
                    expected_bias = normalize_answer(question.get("expected_bias", "")) if is_complex else ""
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
                            shuffled_answers = shuffle_icl_answers(context_samples, args.special, random.Random(seed_i + 1 if seed_i is not None else None))
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

                        rec: Dict[str, Any] = {
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
                        }

                        if is_complex:
                            rec["expected_bias"] = expected_bias
                            rec["alignment"] = "original" if gt == expected_bias else "perturbed"
                            if expected_bias and expected_bias in label_order:
                                bias_onehot = one_hot(expected_bias, label_order)
                                rec["ce_to_bias"] = cross_entropy(bias_onehot, dist)
                            else:
                                rec["ce_to_bias"] = None

                        records.append(rec)

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

                dist_by_gt: Dict[str, List[List[float]]] = {}
                for r in records:
                    dist_by_gt.setdefault(r["ground_truth"], []).append(r["distribution"])
                per_label_mean = {
                    gt_label: np.mean(dists, axis=0).tolist() for gt_label, dists in dist_by_gt.items()
                }

                summary: Dict[str, Any] = {
                    "n_shots": n_shots,
                    "n_samples": n_samples,
                    "accuracy": {"overall": sum(correct_all) / len(correct_all) if correct_all else 0},
                    "mean_ce_to_gt": float(np.mean(valid_ce_gt)) if valid_ce_gt else None,
                    "per_label_mean_distribution": per_label_mean,
                    "label_order": label_order,
                }

                if is_complex:
                    correct_orig = [r["correct"] for r in records if r["alignment"] == "original"]
                    correct_pert = [r["correct"] for r in records if r["alignment"] == "perturbed"]
                    summary["accuracy"]["original"] = sum(correct_orig) / len(correct_orig) if correct_orig else 0
                    summary["accuracy"]["perturbed"] = sum(correct_pert) / len(correct_pert) if correct_pert else 0
                    valid_ce_bias = [r["ce_to_bias"] for r in records if r.get("ce_to_bias") is not None]
                    summary["mean_ce_to_bias"] = float(np.mean(valid_ce_bias)) if valid_ce_bias else None

                with open(shot_dir / "summary.json", "w") as f:
                    json.dump(summary, f, indent=2)

                acc_str = f"acc={summary['accuracy']['overall']:.3f}"
                if is_complex:
                    acc_str += f" (orig={summary['accuracy']['original']:.3f}, pert={summary['accuracy']['perturbed']:.3f})"
                print(f"  {task}/{split} shots={n_shots}: {acc_str}")

            print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
