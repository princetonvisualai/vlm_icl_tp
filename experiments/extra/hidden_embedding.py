"""Extract last-token hidden embeddings from every transformer layer and compare zero-shot vs ICL."""

import argparse
import json
import random
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from vlm_icl.data import prepare_data
from vlm_icl.models import HIDDEN_STATES, load_model, require_capability
from vlm_icl.images import resolve_image
from vlm_icl.icl import (
    build_demo_pool,
    build_icl_context_pools,
    select_from_demo_pool,
    shuffle_icl_answers,
    process_icl_context_images,
)
from vlm_icl.paths import build_analysis_path
from vlm_icl.eb_distance import cosine_divergence
from vlm_icl.evaluation import save_results


def _aggregate_single_icl(averaged, split_lookup, num_layers):
    """Aggregate per-datapoint distances into overall and by-split stats."""
    # Overall stats per layer
    overall = {}
    for layer_idx in range(num_layers):
        vals = [v[layer_idx] for v in averaged.values() if layer_idx in v]
        if vals:
            overall[f"layer_{layer_idx}"] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

    # By split
    by_split = defaultdict(lambda: defaultdict(list))
    for key, layer_dists in averaged.items():
        split = split_lookup.get(key, "unknown")
        for layer_idx, val in layer_dists.items():
            by_split[split][layer_idx].append(val)

    by_split_stats = {}
    for split, layer_vals in sorted(by_split.items()):
        split_stats = {}
        for layer_idx in range(num_layers):
            vals = layer_vals.get(layer_idx, [])
            if vals:
                split_stats[f"layer_{layer_idx}"] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }
        by_split_stats[split] = split_stats

    return {
        "overall": overall,
        "by_split": by_split_stats,
    }


def main():
    from vlm_icl.args import add_model_args, add_data_args, add_icl_args, add_run_config_args

    parser = argparse.ArgumentParser(description="Hidden embedding analysis: zero-shot vs ICL")
    add_model_args(parser)
    add_data_args(parser)
    add_icl_args(parser)
    add_run_config_args(parser)
    parser.set_defaults(sample_per_split=100)
    parser.add_argument("--icl_counts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    args = parser.parse_args()

    random.seed(args.seed)

    # Phase 0: Setup
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, enable_thinking=args.enable_thinking)
    require_capability(model, HIDDEN_STATES)
    print(f"Model loaded: {type(model).__name__}")

    print(f"\nPreparing sampled dataset ({args.dataset})...")
    data_kwargs = {"tasks": args.tasks, "sample_per_split": args.sample_per_split}
    data = prepare_data(dataset=args.dataset, seed=args.seed, **data_kwargs, data_root=args.data_root, data_source=args.data_source)
    print(f"Sampled data: {len(data)} entries")

    if not data:
        print("No data entries found!")
        return

    # Load full dataset for ICL context pools
    full_data_kwargs = {"tasks": args.tasks}
    icl_pool_data = prepare_data(dataset=args.dataset, **full_data_kwargs, data_root=args.data_root, data_source=args.data_source)
    print(f"ICL context pool: {len(icl_pool_data)} entries")

    context_pools = build_icl_context_pools(icl_pool_data)
    demo_pool = build_demo_pool(context_pools, seed=args.seed)

    # Group data by task
    task_groups = defaultdict(list)
    for entry in data:
        task_groups[entry["task"]].append(entry)

    # Prefer PIL images for HF backend
    prefer_pil = True
    answer_rng = random.Random(args.seed + 1)

    for task_name, task_entries in sorted(task_groups.items()):
        print(f"\n{'='*60}")
        print(f"Task: {task_name} ({len(task_entries)} entries)")
        print(f"{'='*60}")

        # Phase 1: Zero-shot embeddings
        print("\nPhase 1: Extracting zero-shot embeddings...")
        zeroshot_embeddings = {}  # (image_id, question_type) -> Dict[int, Tensor]
        skipped_keys = set()

        for entry in tqdm(task_entries, desc="Zero-shot", unit="entry"):
            image_obj = entry.get("image")
            image = resolve_image(image_obj, prefer_pil=prefer_pil) if image_obj is not None else None
            extra_imgs = {}
            for extra_key in ("image2", "image3", "image4"):
                extra_obj = entry.get(extra_key)
                if extra_obj is not None:
                    extra_imgs[extra_key] = resolve_image(extra_obj, prefer_pil=prefer_pil)

            for question in entry.get("questions", []):
                key = (entry["image_id"], question["type"])
                try:
                    hs = model.forward_hidden_states(
                        image=image,
                        prompt=question["prompt"],
                        **extra_imgs,
                    )
                    zeroshot_embeddings[key] = hs
                except torch.cuda.OutOfMemoryError:
                    tqdm.write(f"  CUDA OOM (zero-shot): image_id={entry['image_id']}, skipping")
                    torch.cuda.empty_cache()
                    skipped_keys.add(key)
                except Exception as e:
                    tqdm.write(f"  Error (zero-shot): {e}, skipping")
                    skipped_keys.add(key)

        print(f"Zero-shot: {len(zeroshot_embeddings)} embeddings, {len(skipped_keys)} skipped")

        # Pre-compute lookups and output path before ICL loop
        split_lookup = {}
        for entry in task_entries:
            for question in entry.get("questions", []):
                key = (entry["image_id"], question["type"])
                split_lookup[key] = entry.get("split", "unknown")

        num_layers = max(
            (max(zs_emb.keys()) + 1 for zs_emb in zeroshot_embeddings.values()),
            default=0,
        )

        output_path = build_analysis_path(
            args.output_dir, "hidden_embedding", args.dataset, task_name, args.model_path,
            f"sps{args.sample_per_split}_k{args.num_icl_sample_sets}",
        )
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / "hidden_embedding.json"
        results = {}

        # Phase 2: ICL embeddings + distance computation
        print("\nPhase 2: ICL embeddings + cosine divergence...")

        for num_icl in args.icl_counts:
            print(f"\n  ICL count: {num_icl}")
            # per_datapoint[key] = list of k dicts (layer -> distance scalar)
            per_datapoint_distances = defaultdict(list)

            for entry in tqdm(task_entries, desc=f"ICL-{num_icl}", unit="entry"):
                image_obj = entry.get("image")
                image = resolve_image(image_obj, prefer_pil=prefer_pil) if image_obj is not None else None
                extra_imgs = {}
                for extra_key in ("image2", "image3", "image4"):
                    extra_obj = entry.get(extra_key)
                    if extra_obj is not None:
                        extra_imgs[extra_key] = resolve_image(extra_obj, prefer_pil=prefer_pil)

                for question in entry.get("questions", []):
                    key = (entry["image_id"], question["type"])

                    # Skip if zero-shot was skipped
                    if key in skipped_keys or key not in zeroshot_embeddings:
                        continue

                    zs_emb = zeroshot_embeddings[key]

                    for j in range(args.num_icl_sample_sets):
                        set_seed = args.seed + j if args.seed is not None else None
                        context = select_from_demo_pool(
                            entry, question, demo_pool, num_icl, seed=set_seed
                        )
                        if context is None:
                            tqdm.write(
                                f"  Warning: Insufficient ICL context for "
                                f"image_id={entry['image_id']}, num_icl={num_icl}, skipping"
                            )
                            continue

                        shuffled_answers = shuffle_icl_answers(context, None, answer_rng)
                        icl_context = process_icl_context_images(
                            context, shuffled_answers, None, prefer_pil=prefer_pil
                        )

                        try:
                            icl_emb = model.forward_hidden_states(
                                image=image,
                                prompt=question["prompt"],
                                context=icl_context,
                                **extra_imgs,
                            )

                            layer_dists = {}
                            for layer_idx in zs_emb:
                                if layer_idx in icl_emb:
                                    d = cosine_divergence(
                                        zs_emb[layer_idx].unsqueeze(0),
                                        icl_emb[layer_idx].unsqueeze(0),
                                    )
                                    layer_dists[layer_idx] = d.item()
                            per_datapoint_distances[key].append(layer_dists)

                        except torch.cuda.OutOfMemoryError:
                            tqdm.write(
                                f"  CUDA OOM (ICL-{num_icl}): image_id={entry['image_id']}, skipping"
                            )
                            torch.cuda.empty_cache()
                        except Exception as e:
                            tqdm.write(f"  Error (ICL-{num_icl}): {e}, skipping")

            # Average over k sets per datapoint
            averaged = {}
            for key, dist_list in per_datapoint_distances.items():
                if not dist_list:
                    continue
                all_layers = set()
                for d in dist_list:
                    all_layers.update(d.keys())
                avg = {}
                for layer_idx in sorted(all_layers):
                    vals = [d[layer_idx] for d in dist_list if layer_idx in d]
                    avg[layer_idx] = np.mean(vals)
                averaged[key] = avg

            print(f"  ICL-{num_icl}: {len(averaged)} datapoints with distances")

            # Aggregate and save incrementally
            icl_result = _aggregate_single_icl(averaged, split_lookup, num_layers)
            results[f"icl_{num_icl}"] = icl_result
            save_results(
                results, output_file, model_path=args.model_path,
                extra_metadata={"task": task_name, "dataset": args.dataset, "sample_per_split": args.sample_per_split, "num_icl_sample_sets": args.num_icl_sample_sets, "icl_counts": args.icl_counts, "seed": args.seed, "num_layers": num_layers, "num_datapoints": len(zeroshot_embeddings)},
                format="json",
            )
            print(f"  ICL-{num_icl}: results saved to {output_file}")

        # Summary table
        print(f"\nSummary for {task_name}:")
        print(f"  {'ICL Count':<12} {'Last Layer Mean Dist':<22} {'Last Layer Std'}")
        print(f"  {'-'*50}")
        for num_icl in args.icl_counts:
            key = f"icl_{num_icl}"
            if key in results:
                last_layer_key = f"layer_{num_layers - 1}"
                stats = results[key]["overall"].get(last_layer_key, {})
                mean = stats.get("mean", float("nan"))
                std = stats.get("std", float("nan"))
                print(f"  {num_icl:<12} {mean:<22.6f} {std:.6f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
