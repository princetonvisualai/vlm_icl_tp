"""Formal lens rewrite: compare LatentLens-decoded tokens at distractor positions under baseline vs 8-shot."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm import tqdm

import latentlens

from vlm_icl.args import add_model_args, add_data_args, add_run_config_args
from vlm_icl.data import prepare_data
from vlm_icl.models import load_model
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
    resolve_latentlens_layers,
    detect_num_layers,
    identify_distracting_positions,
    abs_to_grid_positions,
    neighbor_to_dict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spatial_position(abs_pos, query_run_start, grid_w):
    offset = abs_pos - query_run_start
    return (offset // grid_w, offset % grid_w)


def _decode_positions(hidden_states, positions, index, layers, top_k):
    """Return {pos: {layer: [neighbor_dicts]}} for each position at each layer."""
    results = {}
    for layer in layers:
        hs = hidden_states[layer]
        query = hs[0, positions, :]
        query = torch.nn.functional.normalize(query, dim=-1)
        neighbors = index.search(query, top_k=top_k, layers=[layer])
        for i, pos in enumerate(positions):
            results.setdefault(pos, {})[layer] = [
                neighbor_to_dict(n) for n in neighbors[i]
            ]
    return results


def _rank_of_keyword(neighbors, keyword, top_k):
    """1-indexed rank of first neighbor matching *keyword*; top_k+1 if not found."""
    kw = keyword.lower()
    for i, n in enumerate(neighbors):
        if kw in n["token_str"].lower():
            return i + 1
    return top_k + 1


def _compute_similarity_metrics(baseline_neighbors, icl_neighbors, distractor_kw, gt_label, layer, top_k):
    def max_sim_for_keyword(neighbors, keyword):
        matches = [n["similarity"] for n in neighbors if keyword.lower() in n["token_str"].lower()]
        return max(matches) if matches else 0.0

    survival = any(distractor_kw.lower() in n["token_str"].lower() for n in icl_neighbors)
    sim_distract_baseline = max_sim_for_keyword(baseline_neighbors, distractor_kw)
    sim_distract_icl = max_sim_for_keyword(icl_neighbors, distractor_kw)
    sim_gt_baseline = max_sim_for_keyword(baseline_neighbors, gt_label)
    sim_gt_icl = max_sim_for_keyword(icl_neighbors, gt_label)

    return {
        "survival": survival,
        "sim_distract_baseline": sim_distract_baseline,
        "sim_distract_icl": sim_distract_icl,
        "sim_gt_baseline": sim_gt_baseline,
        "sim_gt_icl": sim_gt_icl,
        "rank_distract_baseline": _rank_of_keyword(baseline_neighbors, distractor_kw, top_k),
        "rank_distract_icl": _rank_of_keyword(icl_neighbors, distractor_kw, top_k),
    }


def _compute_drift(hs_baseline, hs_icl, baseline_pos, icl_pos, layer):
    hb = hs_baseline[layer][0, baseline_pos, :]
    hi = hs_icl[layer][0, icl_pos, :]
    hb = torch.nn.functional.normalize(hb, dim=-1)
    hi = torch.nn.functional.normalize(hi, dim=-1)
    return 1.0 - (hb * hi).sum().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal ICL rewrite: distractor token rewrite analysis")
    add_model_args(parser)
    add_data_args(parser)
    add_run_config_args(parser)

    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--latentlens_index", type=str, required=True)
    parser.add_argument("--n_per_label", type=int, default=10)
    parser.add_argument("--distract_cue", type=str, default=None,
                        help="Distractor keyword (e.g. 'stop', 'tiger'). "
                             "Optional when entries have a 'distract' field.")
    parser.add_argument("--expected_bias", type=str, default=None,
                        help="Filter value for expected_bias field")
    parser.add_argument("--num_icl_examples", type=int, default=8)

    args = parser.parse_args(argv)
    if args.seed is None:
        args.seed = 42
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    answer_rng = random.Random(args.seed + 1)
    control_rng = random.Random(args.seed + 2)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, enable_thinking=args.enable_thinking)
    args.layers = resolve_latentlens_layers(args.model_path, args.layers, detect_num_layers(model))

    # Load LatentLens index (loads all available layers; earliest used for distractor ID)
    index_path = Path(args.latentlens_index)
    if index_path.is_dir():
        index = latentlens.ContextualIndex.from_directory(str(index_path))
    else:
        index = latentlens.ContextualIndex.from_pretrained(args.latentlens_index)
    index.to("cuda")
    lens_layer = min(index.available_layers)
    print(f"LatentLens index loaded (using layer {lens_layer} for distractor ID).")

    # Load and filter data
    query_all = prepare_data(dataset=args.dataset, tasks=args.tasks, seed=args.seed, data_root=args.data_root, data_source=args.data_source)
    if args.expected_bias is not None:
        query_all = [
            e for e in query_all
            if e["split"] == "complex"
            and any(q.get("expected_bias") == args.expected_bias for q in e.get("questions", []))
        ]
        print(f"Filtered to {len(query_all)} entries with expected_bias='{args.expected_bias}'")

    query_data = sample_per_label(query_all, n_per_label=args.n_per_label, seed=args.seed)
    print(f"Sampled {len(query_data)} query entries ({args.n_per_label} per label)")

    # Build ICL demo pool (always from visual_reasoning complex split)
    vr_data = prepare_data(dataset="visual_reasoning", tasks=args.tasks, seed=args.seed, data_root=args.data_root, data_source=args.data_source)
    complex_data = [e for e in vr_data if e["split"] == "complex"]
    context_pools = build_icl_context_pools(complex_data)
    demo_pool = build_demo_pool(context_pools, seed=args.seed)

    results: List[Dict[str, Any]] = []

    for entry in tqdm(query_data, desc="Processing entries"):
        question = entry["questions"][0]
        gt = normalize_answer(question["ground_truth"])
        image = resolve_image(entry["image"], prefer_pil=True)

        # Resolve per-entry distractor keyword
        distract_cue = entry.get("distract") or args.distract_cue
        if not distract_cue:
            tqdm.write(f"No distract keyword for {entry['image_id']}, skipping")
            continue

        # --- Baseline condition ---
        try:
            distract_baseline, grid_thw, n_query, query_pos_baseline, hs_baseline = \
                identify_distracting_positions(
                    model, index, image, question["prompt"],
                    args.top_k, distract_cue, layer=lens_layer,
                )
        except torch.cuda.OutOfMemoryError:
            tqdm.write(f"OOM (baseline): {entry['image_id']}, skipping")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            tqdm.write(f"Error (baseline): {e}, skipping")
            continue

        if query_pos_baseline is None or not query_pos_baseline:
            continue

        decoded_baseline = _decode_positions(hs_baseline, query_pos_baseline, index, args.layers, args.top_k)

        # --- 8-shot ICL condition ---
        demo_entry = {**entry, "split": "complex"}
        context = select_from_demo_pool(
            demo_entry, question, demo_pool, args.num_icl_examples, seed=args.seed,
        )
        if context is None:
            tqdm.write(f"Not enough demos for {entry['image_id']}")
            continue

        shuffled_answers = shuffle_icl_answers(context, None, answer_rng)
        icl_context = process_icl_context_images(context, shuffled_answers, None, prefer_pil=True)

        try:
            distract_icl, _, _, query_pos_icl, hs_icl = \
                identify_distracting_positions(
                    model, index, image, question["prompt"],
                    args.top_k, distract_cue,
                    context=icl_context, layer=lens_layer,
                )
        except torch.cuda.OutOfMemoryError:
            tqdm.write(f"OOM (8shot): {entry['image_id']}, skipping")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            tqdm.write(f"Error (8shot): {e}, skipping")
            continue

        if query_pos_icl is None or not query_pos_icl:
            continue

        decoded_icl = _decode_positions(hs_icl, query_pos_icl, index, args.layers, args.top_k)

        # --- Build spatial position mapping ---
        grid_w = grid_thw[2] if grid_thw else 1
        baseline_start = query_pos_baseline[0]
        icl_start = query_pos_icl[0]

        # Map baseline absolute pos -> spatial (row,col) -> ICL absolute pos
        spatial_to_icl = {}
        for pos in query_pos_icl:
            sp = _spatial_position(pos, icl_start, grid_w)
            spatial_to_icl[sp] = pos

        # --- Classify positions ---
        non_distract = [p for p in query_pos_baseline if p not in distract_baseline]
        n_control = min(len(non_distract), len(distract_baseline))
        control_positions = control_rng.sample(non_distract, n_control) if n_control > 0 else []

        # --- Per-position metrics (distractor positions) ---
        per_position = []
        for pos_b in sorted(distract_baseline):
            sp = _spatial_position(pos_b, baseline_start, grid_w)
            pos_i = spatial_to_icl.get(sp)
            if pos_i is None:
                continue

            metrics_per_layer = {}
            for layer in args.layers:
                b_neighbors = decoded_baseline.get(pos_b, {}).get(layer, [])
                i_neighbors = decoded_icl.get(pos_i, {}).get(layer, [])
                metrics_per_layer[layer] = _compute_similarity_metrics(
                    b_neighbors, i_neighbors, distract_cue, gt, layer, args.top_k,
                )

            # Aggregate survival across layers (any layer shows survival)
            survival = any(m["survival"] for m in metrics_per_layer.values())

            # Mean drift across layers
            drifts = [_compute_drift(hs_baseline, hs_icl, pos_b, pos_i, l) for l in args.layers]
            mean_drift = sum(drifts) / len(drifts)

            # Mean similarity metrics across layers
            sim_db = _mean([m["sim_distract_baseline"] for m in metrics_per_layer.values()])
            sim_di = _mean([m["sim_distract_icl"] for m in metrics_per_layer.values()])
            sim_gb = _mean([m["sim_gt_baseline"] for m in metrics_per_layer.values()])
            sim_gi = _mean([m["sim_gt_icl"] for m in metrics_per_layer.values()])

            # Mean rank across layers
            rank_db = _mean([m["rank_distract_baseline"] for m in metrics_per_layer.values()])
            rank_di = _mean([m["rank_distract_icl"] for m in metrics_per_layer.values()])

            per_position.append({
                "grid_pos": list(sp),
                "survival": survival,
                "sim_distract_baseline": sim_db,
                "sim_distract_icl": sim_di,
                "sim_gt_baseline": sim_gb,
                "sim_gt_icl": sim_gi,
                "rank_distract_baseline": rank_db,
                "rank_distract_icl": rank_di,
                "drift": mean_drift,
                "decoded_baseline": {str(l): decoded_baseline.get(pos_b, {}).get(l, []) for l in args.layers},
                "decoded_icl": {str(l): decoded_icl.get(pos_i, {}).get(l, []) for l in args.layers},
            })

        # --- Control position drift ---
        control_drift = []
        for pos_b in control_positions:
            sp = _spatial_position(pos_b, baseline_start, grid_w)
            pos_i = spatial_to_icl.get(sp)
            if pos_i is None:
                continue
            drifts = [_compute_drift(hs_baseline, hs_icl, pos_b, pos_i, l) for l in args.layers]
            control_drift.append(sum(drifts) / len(drifts))

        # --- Aggregate ---
        n_surviving = sum(1 for p in per_position if p["survival"])
        n_distract = len(per_position)

        new_distractors_icl = distract_icl - distract_baseline

        record = {
            "image_id": entry["image_id"],
            "question_type": question["type"],
            "ground_truth": gt,
            "distract_cue": distract_cue,
            "expected_bias": question.get("expected_bias"),
            "n_distract_tokens_baseline": len(distract_baseline),
            "n_distract_tokens_icl": len(distract_icl),
            "n_query_tokens": len(query_pos_baseline),
            "grid_thw": list(grid_thw) if grid_thw else None,
            "distract_grid_positions_baseline": abs_to_grid_positions(distract_baseline, query_pos_baseline, grid_thw) if grid_thw else [],
            "distract_grid_positions_icl": abs_to_grid_positions(distract_icl, query_pos_icl, grid_thw) if grid_thw else [],
            "per_position": per_position,
            "control_drift": control_drift,
            "aggregate": {
                "survival_rate": n_surviving / n_distract if n_distract else 0.0,
                "mean_distract_sim_delta": _mean([p["sim_distract_baseline"] - p["sim_distract_icl"] for p in per_position]),
                "mean_gt_sim_delta": _mean([p["sim_gt_icl"] - p["sim_gt_baseline"] for p in per_position]),
                "mean_drift_distractor": _mean([p["drift"] for p in per_position]),
                "mean_drift_control": _mean(control_drift),
                "mean_rank_distract_baseline": _mean([p["rank_distract_baseline"] for p in per_position]),
                "mean_rank_distract_icl": _mean([p["rank_distract_icl"] for p in per_position]),
                "mean_rank_drop": _mean([p["rank_distract_icl"] - p["rank_distract_baseline"] for p in per_position]),
                "n_new_distractors_icl": len(new_distractors_icl),
            },
        }
        results.append(record)
        torch.cuda.empty_cache()

    # --- Summary ---
    summary = {
        "n_samples": len(results),
        "mean_survival_rate": _mean_std([r["aggregate"]["survival_rate"] for r in results]),
        "mean_distract_sim_delta": _mean_std([r["aggregate"]["mean_distract_sim_delta"] for r in results]),
        "mean_gt_sim_delta": _mean_std([r["aggregate"]["mean_gt_sim_delta"] for r in results]),
        "mean_drift_distractor": _mean_std([r["aggregate"]["mean_drift_distractor"] for r in results]),
        "mean_drift_control": _mean_std([r["aggregate"]["mean_drift_control"] for r in results]),
        "mean_rank_distract_baseline": _mean_std([r["aggregate"]["mean_rank_distract_baseline"] for r in results]),
        "mean_rank_distract_icl": _mean_std([r["aggregate"]["mean_rank_distract_icl"] for r in results]),
        "mean_rank_drop": _mean_std([r["aggregate"]["mean_rank_drop"] for r in results]),
        "total_new_distractors_icl": sum(r["aggregate"]["n_new_distractors_icl"] for r in results),
        "metadata": {
            "distract_cue": args.distract_cue,
            "expected_bias": args.expected_bias,
            "distract_cue_source": "per_entry" if args.distract_cue is None else "cli",
            "layers": args.layers,
            "n_per_label": args.n_per_label,
            "num_icl_examples": args.num_icl_examples,
            "top_k": args.top_k,
        },
    }

    # Per-layer breakdown
    per_layer = {}
    for layer in args.layers:
        layer_survivals = []
        layer_distract_deltas = []
        layer_gt_deltas = []
        layer_rank_baseline = []
        layer_rank_icl = []
        for r in results:
            kw = r["distract_cue"]
            for p in r["per_position"]:
                b_n = p["decoded_baseline"].get(str(layer), [])
                i_n = p["decoded_icl"].get(str(layer), [])
                surv = any(kw.lower() in n["token_str"].lower() for n in i_n)
                layer_survivals.append(surv)
                db = max((n["similarity"] for n in b_n if kw.lower() in n["token_str"].lower()), default=0.0)
                di = max((n["similarity"] for n in i_n if kw.lower() in n["token_str"].lower()), default=0.0)
                layer_distract_deltas.append(db - di)
                gb = max((n["similarity"] for n in b_n if r["ground_truth"].lower() in n["token_str"].lower()), default=0.0)
                gi = max((n["similarity"] for n in i_n if r["ground_truth"].lower() in n["token_str"].lower()), default=0.0)
                layer_gt_deltas.append(gi - gb)
                layer_rank_baseline.append(_rank_of_keyword(b_n, kw, args.top_k))
                layer_rank_icl.append(_rank_of_keyword(i_n, kw, args.top_k))
        per_layer[str(layer)] = {
            "survival_rate": sum(layer_survivals) / len(layer_survivals) if layer_survivals else 0.0,
            "mean_distract_sim_delta": _mean(layer_distract_deltas),
            "mean_gt_sim_delta": _mean(layer_gt_deltas),
            "mean_rank_distract_baseline": _mean(layer_rank_baseline),
            "mean_rank_distract_icl": _mean(layer_rank_icl),
            "mean_rank_drop": _mean([i - b for b, i in zip(layer_rank_baseline, layer_rank_icl)]),
        }
    summary["per_layer"] = per_layer

    # Save
    task_name = args.tasks[0] if args.tasks else "all"
    leaf_parts = [f"npl{args.n_per_label}"]
    if args.expected_bias is not None:
        leaf_parts.append(args.expected_bias)
    leaf = "_".join(leaf_parts)
    out_dir = build_analysis_path(
        args.output_dir, "formal_lens_rewrite", args.dataset,
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
    if results:
        print(f"  Survival rate: {summary['mean_survival_rate']['mean']:.3f} +/- {summary['mean_survival_rate']['std']:.3f}")
        print(f"  Distract sim delta: {summary['mean_distract_sim_delta']['mean']:.4f}")
        print(f"  GT sim delta: {summary['mean_gt_sim_delta']['mean']:.4f}")
        print(f"  Drift (distract): {summary['mean_drift_distractor']['mean']:.4f}")
        print(f"  Drift (control): {summary['mean_drift_control']['mean']:.4f}")
        print(f"  Rank baseline: {summary['mean_rank_distract_baseline']['mean']:.2f} +/- {summary['mean_rank_distract_baseline']['std']:.2f}")
        print(f"  Rank ICL: {summary['mean_rank_distract_icl']['mean']:.2f} +/- {summary['mean_rank_distract_icl']['std']:.2f}")
        print(f"  Rank drop: {summary['mean_rank_drop']['mean']:.2f} +/- {summary['mean_rank_drop']['std']:.2f}")
    print(f"Saved to: {out_dir}")


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------

def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _mean_std(vals):
    m = _mean(vals)
    if len(vals) < 2:
        return {"mean": m, "std": 0.0}
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return {"mean": m, "std": var ** 0.5}


if __name__ == "__main__":
    main()
