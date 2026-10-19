"""Formal lens divert: per-entry ICL − 0-shot attention-density deltas.

For each query image we classify visual tokens into
{prior_cues, grounding_cues, others} via LatentLens top-k decoding at
max(index.available_layers), run a single 0-shot forward plus N independent
N-shot draws, and record per-entry
    delta[g] = mean_over_draws(S_g / N_g)_icl  -  (S_g / N_g)_zeroshot
where S_g / N_g is the per-token attention average within group g for that run.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from tqdm import tqdm

import latentlens

from vlm_icl.args import add_model_args, add_data_args, add_run_config_args
from vlm_icl.data import prepare_data
from vlm_icl.models import load_model
from vlm_icl.models.hf import get_decoder_layers
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
from vlm_icl.lens import (
    classify_query_tokens,
    get_query_token_info,
)


BUCKETS = ("prior_cues", "grounding_cues", "others")


# ---------------------------------------------------------------------------
# Forward pass with last-row-only attention capture
# ---------------------------------------------------------------------------

def _forward_capture_last_row(model, inputs, layer_idx):
    """Run forward; capture only the last-row attention (heads, seq) for one decoder layer.

    Full (heads, seq, seq) capture would be huge; this keeps it tiny.
    """
    decoder_layers = get_decoder_layers(model.model)
    captured: Dict[int, torch.Tensor] = {}

    def hook_fn(module, module_input, output):
        captured[layer_idx] = output[1][0, :, -1, :].detach().cpu()
        return (output[0], None)

    handle = decoder_layers[layer_idx].self_attn.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model.model(**inputs, output_attentions=False, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    return captured


# ---------------------------------------------------------------------------
# Bucket attention aggregation
# ---------------------------------------------------------------------------

def _compute_bucket_attention(last_row_attn, query_positions, bucket_by_position):
    """Average heads, normalize over query-image tokens, sum attn_pct per bucket.

    Returns {bucket: {"sum_attn_pct": float, "n_tokens": int}}.
    """
    if last_row_attn is None or not query_positions:
        return {b: {"sum_attn_pct": 0.0, "n_tokens": 0} for b in BUCKETS}

    last_row_avg = last_row_attn.float().mean(dim=0)  # (seq,)
    query_idx = torch.tensor(query_positions, dtype=torch.long)
    attn_q = last_row_avg[query_idx]
    total = attn_q.sum()
    attn_pct = attn_q / total if total > 0 else torch.zeros_like(attn_q)

    out = {b: {"sum_attn_pct": 0.0, "n_tokens": 0} for b in BUCKETS}
    for i, pos in enumerate(query_positions):
        bucket = bucket_by_position.get(pos, "others")
        out[bucket]["sum_attn_pct"] += float(attn_pct[i].item())
        out[bucket]["n_tokens"] += 1
    return out


def _mean_per_token(buckets) -> Dict[str, Optional[float]]:
    """Return sum_attn_pct / n_tokens per bucket, or None if n_tokens == 0."""
    out: Dict[str, Optional[float]] = {}
    for b in BUCKETS:
        info = buckets.get(b, {"sum_attn_pct": 0.0, "n_tokens": 0})
        n = info["n_tokens"]
        out[b] = (info["sum_attn_pct"] / n) if n > 0 else None
    return out


# ---------------------------------------------------------------------------
# Per-condition run
# ---------------------------------------------------------------------------

def _run_condition(
    model, index, lens_layer, tokenizer, image, prompt,
    prior_cue, grounding_cues, top_k, last_layer_idx, context=None,
) -> Optional[Dict[str, Optional[float]]]:
    """Run LatentLens classification + last-row attention forward for one condition.

    Returns a per-group per-token mean dict `{prior_cues, grounding_cues, others}`
    with `float` values, or `None` if the run is unusable (no query positions, or
    either `prior_cues` or `grounding_cues` bucket is empty, or alignment fails).
    """
    bucket_by_pos_ll, query_positions_ll, n_query_ll = classify_query_tokens(
        model, index, image, prompt, top_k, prior_cue,
        grounding_cues=grounding_cues, context=context, layer=lens_layer,
    )
    if not query_positions_ll:
        return None
    has_prior = any(v == "prior_cues" for v in bucket_by_pos_ll.values())
    has_grounding = any(v == "grounding_cues" for v in bucket_by_pos_ll.values())
    if not (has_prior and has_grounding):
        return None

    messages = model.build_messages(image, prompt, context=context)
    inputs = model.prepare_inputs(messages)
    query_run, _ = get_query_token_info(
        tokenizer, inputs["input_ids"], inputs.get("image_grid_thw"),
    )
    if not query_run or len(query_run) != n_query_ll:
        return None

    captured = _forward_capture_last_row(model, inputs, last_layer_idx)
    last_row_attn = captured.get(last_layer_idx)
    buckets = _compute_bucket_attention(last_row_attn, query_run, bucket_by_pos_ll)
    return _mean_per_token(buckets)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal lens divert: per-entry ICL − 0-shot attention delta")
    add_model_args(parser)
    add_data_args(parser)
    parser.set_defaults(dataset="visual_reasoning_lens")
    add_run_config_args(parser)

    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--latentlens_index", type=str, required=True)
    parser.add_argument("--n_per_label", type=int, default=10)
    parser.add_argument("--prior_cue", type=str, default=None,
                        help="Prior-cue keyword. Falls back to entry['distract'] when unset.")
    parser.add_argument("--grounding_cue", type=str, default=None,
                        help="Optional single grounding-cue override; falls back to per-entry ground_truth.")
    parser.add_argument("--num_icl_examples", type=int, default=8,
                        help="ICL shot count per draw (default: 8). Arbitrary values accepted.")
    parser.add_argument("--n_icl_draws", type=int, default=10,
                        help="Number of independent N-shot draws per entry (default: 10).")
    parser.add_argument("--counting_extra_cues", type=str, nargs="*",
                        default=["leg", "legs"],
                        help="Extra grounding substrings applied only when --tasks == ['counting'].")

    args = parser.parse_args(argv)
    if args.seed is None:
        args.seed = 42
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    answer_rng = random.Random(args.seed + 1)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, enable_thinking=args.enable_thinking,
                       attn_implementation="eager")

    # Load LatentLens index; use the latest layer for classification
    index_path = Path(args.latentlens_index)
    if index_path.is_dir():
        index = latentlens.ContextualIndex.from_directory(str(index_path))
    else:
        index = latentlens.ContextualIndex.from_pretrained(args.latentlens_index)
    index.to("cuda")
    lens_layer = max(index.available_layers)

    # Data: complex split only, sample_per_label balancing
    all_data = prepare_data(dataset=args.dataset, tasks=args.tasks, seed=args.seed, data_root=args.data_root, data_source=args.data_source)
    complex_data = [e for e in all_data if e["split"] == "complex"]

    query_data = sample_per_label(complex_data, n_per_label=args.n_per_label, seed=args.seed)
    print(f"Sampled {len(query_data)} query entries ({args.n_per_label} per label)")

    # Attention layer: last decoder layer
    last_layer_idx = len(get_decoder_layers(model.model)) - 1
    print(f"Attention hook: layer {last_layer_idx}; LatentLens layer: {lens_layer}")
    print(f"Shot count: {args.num_icl_examples}; draws per entry: {args.n_icl_draws}")

    # Demo pool from the same complex-split data
    context_pools = build_icl_context_pools(complex_data)
    demo_pool = build_demo_pool(context_pools, seed=args.seed)

    tokenizer = model.processor.tokenizer if hasattr(model.processor, "tokenizer") else model.processor

    counting_augment = args.tasks == ["counting"]
    extra_grounding = list(args.counting_extra_cues) if counting_augment else []
    shots = args.num_icl_examples

    per_entry_records = []
    aggregates: Dict[str, list] = {g: [] for g in BUCKETS}

    prior_cue_source = "cli" if args.prior_cue else "per_entry_distract"
    grounding_cue_source = "cli" if args.grounding_cue else "per_entry_ground_truth"

    for entry in tqdm(query_data, desc="Processing entries"):
        question = entry["questions"][0]
        image = resolve_image(entry["image"], prefer_pil=True)
        image_id = entry["image_id"]

        prior_cue = entry.get("distract") or args.prior_cue
        grounding_cue = args.grounding_cue or normalize_answer(question.get("ground_truth", ""))
        grounding_cues = ([grounding_cue.lower()] if grounding_cue else []) + extra_grounding

        if not prior_cue or not grounding_cues:
            tqdm.write(f"Missing prior/grounding cue for {image_id}, skipping")
            continue

        # 0-shot: single forward
        try:
            zs = _run_condition(
                model, index, lens_layer, tokenizer,
                image, question["prompt"], prior_cue, grounding_cues,
                args.top_k, last_layer_idx, context=None,
            )
        except torch.cuda.OutOfMemoryError as e:
            tqdm.write(f"OOM (0-shot) on {image_id}: {e}")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            tqdm.write(f"Error (0-shot) on {image_id}: {e}")
            torch.cuda.empty_cache()
            continue
        if zs is None or any(v is None for v in zs.values()):
            torch.cuda.empty_cache()
            continue
        torch.cuda.empty_cache()

        # N-shot: n_icl_draws independent draws
        icl_vals: Dict[str, list] = {g: [] for g in BUCKETS}
        entry_failed = False
        for k in range(args.n_icl_draws):
            context = select_from_demo_pool(
                entry, question, demo_pool, shots, seed=args.seed + k,
            )
            if context is None:
                tqdm.write(f"Not enough demos for {image_id} (draw {k})")
                entry_failed = True
                break
            shuffled_answers = shuffle_icl_answers(context, None, answer_rng)
            icl_context = process_icl_context_images(
                context, shuffled_answers, None, prefer_pil=True,
            )

            try:
                res = _run_condition(
                    model, index, lens_layer, tokenizer,
                    image, question["prompt"], prior_cue, grounding_cues,
                    args.top_k, last_layer_idx, context=icl_context,
                )
            except torch.cuda.OutOfMemoryError as e:
                tqdm.write(f"OOM (draw {k}) on {image_id}: {e}")
                torch.cuda.empty_cache()
                entry_failed = True
                break
            except Exception as e:
                tqdm.write(f"Error (draw {k}) on {image_id}: {e}")
                torch.cuda.empty_cache()
                entry_failed = True
                break

            if res is None or any(v is None for v in res.values()):
                entry_failed = True
                torch.cuda.empty_cache()
                break
            for g in BUCKETS:
                icl_vals[g].append(res[g])
            torch.cuda.empty_cache()

        if entry_failed:
            continue

        icl_mean = {g: sum(icl_vals[g]) / len(icl_vals[g]) for g in BUCKETS}
        delta = {g: icl_mean[g] - zs[g] for g in BUCKETS}
        per_entry_records.append({
            "image_id": image_id,
            "task": entry["task"],
            "split": entry["split"],
            "delta_icl_minus_zeroshot": delta,
        })
        for g in BUCKETS:
            aggregates[g].append(delta[g])

    # ------------------------------------------------------------------
    # Aggregate summary (population variance, ddof=0)
    # ------------------------------------------------------------------
    n_entries = len(per_entry_records)
    aggregate_delta = {}
    for g in BUCKETS:
        vals = aggregates[g]
        n = len(vals)
        if n > 0:
            mean = sum(vals) / n
            variance = sum((x - mean) ** 2 for x in vals) / n
        else:
            mean = 0.0
            variance = 0.0
        aggregate_delta[g] = {
            "mean": float(mean),
            "variance": float(variance),
            "n": int(n),
        }

    summary = {
        "n_entries": n_entries,
        "layer_index": last_layer_idx,
        "lens_layer": lens_layer,
        "shot_count": args.num_icl_examples,
        "n_icl_draws": args.n_icl_draws,
        "counting_augment": bool(counting_augment),
        "extra_grounding_cues": extra_grounding,
        "prior_cue": args.prior_cue,
        "prior_cue_source": prior_cue_source,
        "grounding_cue_source": grounding_cue_source,
        "aggregate_delta_icl_minus_zeroshot": aggregate_delta,
    }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    task_name = args.tasks[0] if args.tasks else "all"
    leaf = f"npt{args.n_per_label}"
    out_dir = build_analysis_path(
        args.output_dir, "formal_lens_divert", args.dataset,
        task_name, args.model_path, leaf,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    with open(out_dir / "results.jsonl", "w") as f:
        for rec in per_entry_records:
            f.write(json.dumps(rec, default=str) + "\n")

    # ------------------------------------------------------------------
    # Console printout
    # ------------------------------------------------------------------
    print(f"\nProcessed {n_entries} entries")
    for g in BUCKETS:
        info = aggregate_delta[g]
        print(f"  [{g}] delta(mean)={info['mean']:+.6f}  variance={info['variance']:.6e}  n={info['n']}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
