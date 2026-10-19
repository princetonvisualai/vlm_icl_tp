"""Formal ICL Reasoning: ICL scaling with thinking/reasoning mode (generative, no logprobs)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
from vlm_icl.metrics import normalize_answer
from vlm_icl.models import load_model
from vlm_icl.args import add_model_args, add_data_args, add_run_config_args

ALL_SPLITS = ("simple", "noprior", "complex")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal ICL Reasoning: ICL scaling with thinking mode")
    add_model_args(parser)
    add_data_args(parser)
    add_run_config_args(parser)

    parser.add_argument("--shot_counts", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32])
    parser.add_argument("--splits", type=str, nargs="+", default=None,
                        help="Splits to run (default: all). Choices: simple, noprior, complex")
    parser.add_argument("--n_per_label", type=int, default=40)
    parser.add_argument("--num_demo_sets", type=int, default=10)

    args = parser.parse_args(argv)
    # Enable thinking by default (this experiment's purpose)
    if not args.enable_thinking:
        args.enable_thinking = True
    if args.tasks is None:
        args.tasks = ["counting", "shape_identification", "color_identification", "direction", "relation"]
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    all_data = prepare_data(tasks=args.tasks, dataset="visual_reasoning", seed=args.seed, data_root=args.data_root, data_source=args.data_source)

    splits = tuple(args.splits) if args.splits else ALL_SPLITS
    split_data = {s: [e for e in all_data if e["split"] == s] for s in splits}
    split_queries = {s: sample_per_label(split_data[s], n_per_label=args.n_per_label, seed=args.seed) for s in ALL_SPLITS if split_data[s]}
    split_pools = {}
    split_demo_pools = {}
    for s in splits:
        if split_data[s]:
            split_pools[s] = build_icl_context_pools(split_data[s])
            split_demo_pools[s] = build_demo_pool(split_pools[s], seed=args.seed)

    # Load generative model
    prefer_pil = args.backend != "vllm"
    model_kwargs = {}
    if args.backend == "vllm":
        model_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
        model_kwargs["max_num_images"] = args.max_num_images
        model_kwargs["max_model_len"] = args.max_model_len
    model = load_model(args.model_path, backend=args.backend, enable_thinking=args.enable_thinking, **model_kwargs)
    print(f"Model loaded: {type(model).__name__} (thinking={args.enable_thinking})")

    run_config = derive_run_config(args.special, args.cot)

    for task in args.tasks:
        for split in splits:
            if split not in split_queries or not split_data[split]:
                continue

            is_complex = split == "complex"
            leaf = f"npl{args.n_per_label}_{split}"
            out_base = build_output_path(
                base_dir=args.output_dir,
                analysis_type="formal_icl_reasoning",
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

                        raw_response = model.generate(
                            image=image,
                            prompt=question["prompt"],
                            context=icl_context,
                            cot=args.cot,
                        )
                        predicted_label = normalize_answer(raw_response)

                        rec: Dict[str, Any] = {
                            "image_id": entry["image_id"],
                            "task": task,
                            "split": split,
                            "ground_truth": gt,
                            "n_shots": n_shots,
                            "demo_set_idx": demo_idx,
                            "predicted_label": predicted_label,
                            "correct": predicted_label == gt,
                            "raw_response": raw_response,
                        }

                        if is_complex:
                            rec["expected_bias"] = expected_bias
                            rec["alignment"] = "original" if gt == expected_bias else "perturbed"

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
                summary: Dict[str, Any] = {
                    "n_shots": n_shots,
                    "n_samples": n_samples,
                    "accuracy": {"overall": sum(correct_all) / len(correct_all) if correct_all else 0},
                }

                if is_complex:
                    correct_orig = [r["correct"] for r in records if r["alignment"] == "original"]
                    correct_pert = [r["correct"] for r in records if r["alignment"] == "perturbed"]
                    summary["accuracy"]["original"] = sum(correct_orig) / len(correct_orig) if correct_orig else 0
                    summary["accuracy"]["perturbed"] = sum(correct_pert) / len(correct_pert) if correct_pert else 0

                with open(shot_dir / "summary.json", "w") as f:
                    json.dump(summary, f, indent=2)

                acc_str = f"acc={summary['accuracy']['overall']:.3f}"
                if is_complex:
                    acc_str += f" (orig={summary['accuracy']['original']:.3f}, pert={summary['accuracy']['perturbed']:.3f})"
                print(f"  {task}/{split} shots={n_shots}: {acc_str}")

            print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()

