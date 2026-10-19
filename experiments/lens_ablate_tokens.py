"""Formal ablation: identify distracting visual tokens via LatentLens, then compare baseline / ablate / 8-shot."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm import tqdm

import latentlens

from vlm_icl.args import add_model_args, add_data_args, add_run_config_args
from vlm_icl.data import prepare_data
from vlm_icl.images import resolve_image
from vlm_icl.icl import (
    build_demo_pool,
    build_icl_context_pools,
    sample_per_label,
    select_from_demo_pool,
    shuffle_icl_answers,
    process_icl_context_images,
)
from vlm_icl.paths import build_analysis_path
from vlm_icl.metrics import normalize_answer
from vlm_icl.models import ISOLATION_MASK, require_capability
from vlm_icl.lens import (
    identify_distracting_positions,
    abs_to_grid_positions,
)
from vlm_icl.scoring import HFAnswerScorer


# ---------------------------------------------------------------------------
# Logprob → probability conversion
# ---------------------------------------------------------------------------

def _logprobs_to_probs(label_logprobs, labels):
    """Convert label log-probabilities to a softmax distribution over *labels*."""
    import math
    lps = [label_logprobs[l] for l in labels]
    max_lp = max(lps)
    if max_lp == float('-inf'):
        n = len(labels)
        return {l: round(1.0 / n, 6) for l in labels}
    exp_vals = {l: math.exp(label_logprobs[l] - max_lp) for l in labels}
    total = sum(exp_vals.values())
    return {l: round(v / total, 6) for l, v in exp_vals.items()}


# ---------------------------------------------------------------------------
# Token ablation mask
# ---------------------------------------------------------------------------

def _make_token_ablation_mask_fn(distract_positions):
    """Return and_mask_function closure that blocks attention TO distract positions (column mask)."""
    pos = torch.tensor(sorted(distract_positions), dtype=torch.int64, device="cuda")
    n = pos.shape[0]

    def mask_fn(b, h, q_idx, kv_idx):
        blocked = torch.tensor(False, device=kv_idx.device)
        for i in range(n):
            blocked = blocked | (kv_idx == pos[i])
        return ~blocked

    return mask_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal ablation: distract token identification + 3-condition eval")
    add_model_args(parser)
    add_data_args(parser)
    add_run_config_args(parser)

    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--latentlens_index", type=str, required=True)
    parser.add_argument("--n_per_label", type=int, default=10)
    parser.add_argument("--distract_cue", type=str, default=None,
                        help="Override distractor keyword. Defaults to per-entry entry['distract'].")
    parser.add_argument("--expected_bias", type=str, default=None,
                        help="Optional filter: only keep questions whose expected_bias matches.")
    parser.add_argument("--num_icl_examples", type=int, default=8)
    parser.add_argument("--num_demo_sets", type=int, default=10,
                        help="Number of demo sets to sample per query for ICL (default: 5)")
    parser.add_argument("--answer_choices", type=str, nargs="+", default=None,
                        help="Answer choices for probability computation (auto-detected if omitted)")

    args = parser.parse_args(argv)
    if args.seed is None:
        args.seed = 42
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    answer_rng = random.Random(args.seed + 1)

    # Load model (via HFAnswerScorer so we can reuse score_answers)
    print(f"Loading model from {args.model_path}...")
    scorer = HFAnswerScorer(model_path=args.model_path, enable_thinking=args.enable_thinking)
    model = scorer.model_wrapper
    require_capability(model, ISOLATION_MASK)

    # Load LatentLens index (use the earliest available layer)
    index_path = Path(args.latentlens_index)
    if index_path.is_dir():
        index = latentlens.ContextualIndex.from_directory(str(index_path))
    else:
        index = latentlens.ContextualIndex.from_pretrained(args.latentlens_index)
    index.to("cuda")
    lens_layer = min(index.available_layers)
    print(f"LatentLens index loaded (using layer {lens_layer}).")

    # Load and filter data
    all_data = prepare_data(dataset="visual_reasoning_lens", tasks=args.tasks, seed=args.seed, data_root=args.data_root, data_source=args.data_source)
    complex_data = [e for e in all_data if e["split"] == "complex"]

    if args.expected_bias:
        subset = [
            e for e in complex_data
            if any(q.get("expected_bias") == args.expected_bias for q in e.get("questions", []))
        ]
        print(f"Filtered to {len(subset)} entries with expected_bias='{args.expected_bias}'")
    else:
        subset = complex_data
        print(f"No expected_bias filter; using {len(subset)} complex entries")

    query_data = sample_per_label(subset, n_per_label=args.n_per_label, seed=args.seed)
    print(f"Sampled {len(query_data)} query entries ({args.n_per_label} per label)")

    # Collect answer choices for probability computation
    if args.answer_choices:
        answer_choices = [normalize_answer(c) for c in args.answer_choices]
    else:
        answer_choices = sorted({
            normalize_answer(q["ground_truth"])
            for e in complex_data for q in e.get("questions", [])
        })
    print(f"Answer choices ({len(answer_choices)}): {answer_choices}")

    # Build ICL demo pool
    context_pools = build_icl_context_pools(complex_data)
    demo_pool = build_demo_pool(context_pools, seed=args.seed)

    conditions = ["baseline", "ablate", "8shot"]
    results: List[Dict[str, Any]] = []

    for entry in tqdm(query_data, desc="Processing entries"):
        question = entry["questions"][0]
        gt = normalize_answer(question["ground_truth"])
        image = resolve_image(entry["image"], prefer_pil=True)

        distract_cue = args.distract_cue or entry.get("distract")
        if not distract_cue:
            tqdm.write(f"No distract cue for {entry['image_id']}, skipping")
            continue
        entry_expected_bias = question.get("expected_bias")

        # Phase 1: identify distracting tokens
        try:
            distract_positions, grid_thw, n_query_tokens, query_run, _ = identify_distracting_positions(
                model, index, image, question["prompt"],
                args.top_k, distract_cue, layer=lens_layer,
            )
        except torch.cuda.OutOfMemoryError:
            tqdm.write(f"OOM (LatentLens): {entry['image_id']}, skipping")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            tqdm.write(f"Error (LatentLens): {e}, skipping")
            continue

        record = {
            "image_id": entry["image_id"],
            "question_type": question["type"],
            "ground_truth": gt,
            "expected_bias": entry_expected_bias,
            "distract_cue": distract_cue,
            "n_distract_tokens": len(distract_positions),
            "n_query_tokens": n_query_tokens,
            "distract_positions": sorted(distract_positions),
            "distract_grid_positions": abs_to_grid_positions(distract_positions, query_run, grid_thw) if grid_thw else [],
            "grid_thw": list(grid_thw) if grid_thw else None,
            "conditions": {},
        }

        # Condition A: baseline (zero-shot)
        try:
            result = scorer.score_answers(
                image=image, prompt=question["prompt"], labels=answer_choices,
            )
            record["conditions"]["baseline"] = {
                "answer": result["generated_text"],
                "correct": result["predicted_label"] == gt,
                "answer_probs": _logprobs_to_probs(result["label_logprobs"], answer_choices),
            }
        except Exception as e:
            tqdm.write(f"Error (baseline): {e}")
            record["conditions"]["baseline"] = {"answer": None, "correct": False, "answer_probs": None}

        # Condition B: ablate (zero-shot + mask distracting tokens)
        if distract_positions:
            mask_fn = _make_token_ablation_mask_fn(distract_positions)
            model.set_isolation_mask(mask_fn, decode_only=True)
            try:
                result = scorer.score_answers(
                    image=image, prompt=question["prompt"], labels=answer_choices,
                )
                record["conditions"]["ablate"] = {
                    "answer": result["generated_text"],
                    "correct": result["predicted_label"] == gt,
                    "answer_probs": _logprobs_to_probs(result["label_logprobs"], answer_choices),
                }
            except Exception as e:
                tqdm.write(f"Error (ablate): {e}")
                record["conditions"]["ablate"] = {"answer": None, "correct": False, "answer_probs": None}
            finally:
                model.clear_isolation_mask()
        else:
            # No distracting tokens found — same as baseline
            record["conditions"]["ablate"] = record["conditions"]["baseline"].copy()

        # Condition C: 8-shot ICL (averaged over multiple demo sets)
        demo_entry = {**entry, "split": "complex"}
        ds_answers = []
        ds_probs = []
        for ds_idx in range(args.num_demo_sets):
            demo_seed = args.seed + ds_idx
            ds_answer_rng = random.Random(demo_seed + 1)
            context = select_from_demo_pool(
                demo_entry, question, demo_pool, args.num_icl_examples, seed=demo_seed,
            )
            if context is None:
                continue
            shuffled_answers = shuffle_icl_answers(context, None, ds_answer_rng)
            icl_context = process_icl_context_images(context, shuffled_answers, None, prefer_pil=True)
            try:
                result = scorer.score_answers(
                    image=image, prompt=question["prompt"], labels=answer_choices,
                    context=icl_context,
                )
                ds_answers.append(result["generated_text"])
                ds_probs.append(_logprobs_to_probs(result["label_logprobs"], answer_choices))
            except Exception as e:
                tqdm.write(f"Error (8shot ds{ds_idx}): {e}")

        if ds_answers:
            # Majority vote for answer
            vote_counts = Counter(normalize_answer(a) for a in ds_answers)
            majority_answer = vote_counts.most_common(1)[0][0]
            record["conditions"]["8shot"] = {
                "answer": majority_answer,
                "correct": majority_answer == gt,
                "all_answers": ds_answers,
                "num_demo_sets": len(ds_answers),
            }
            # Average probs across demo sets
            if ds_probs:
                avg_probs = {}
                for c in answer_choices:
                    avg_probs[c] = round(sum(p.get(c, 0.0) for p in ds_probs) / len(ds_probs), 6)
                record["conditions"]["8shot"]["answer_probs"] = avg_probs
                record["conditions"]["8shot"]["all_answer_probs"] = ds_probs
            else:
                record["conditions"]["8shot"]["answer_probs"] = None
        else:
            tqdm.write(f"Not enough demos for {entry['image_id']}")
            record["conditions"]["8shot"] = {"answer": None, "correct": False}

        results.append(record)

    # Compute summary
    accuracy = {}
    label_dist = {}
    for cond in conditions:
        cond_results = [r for r in results if cond in r["conditions"]]
        correct = [r["conditions"][cond]["correct"] for r in cond_results]
        accuracy[cond] = sum(correct) / len(correct) if correct else 0.0

        answers = [
            normalize_answer(r["conditions"][cond]["answer"])
            for r in cond_results if r["conditions"][cond]["answer"] is not None
        ]
        counts = Counter(answers)
        total = sum(counts.values())
        label_dist[cond] = {k: v / total for k, v in sorted(counts.items())} if total else {}

    summary = {
        "n_samples": len(results),
        "accuracy": accuracy,
        "label_distribution": label_dist,
        "distract_cue": args.distract_cue,
        "distract_cue_source": "cli" if args.distract_cue else "per_entry",
        "expected_bias_filter": args.expected_bias,
        "n_per_label": args.n_per_label,
        "num_icl_examples": args.num_icl_examples,
        "answer_choices": answer_choices,
    }

    # Save
    task_name = args.tasks[0] if args.tasks else "all"
    leaf = f"npl{args.n_per_label}"
    if args.expected_bias:
        leaf += f"_{args.expected_bias}"
    out_dir = build_analysis_path(
        args.output_dir, "formal_lens_ablate_tokens", "visual_reasoning_lens",
        task_name, args.model_path, leaf,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print(f"\nResults: {len(results)} entries")
    for cond in conditions:
        print(f"  {cond}: acc={accuracy[cond]:.3f}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
