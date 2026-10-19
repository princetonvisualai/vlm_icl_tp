"""Control hidden embedding analysis — fixed demo set, clean vs corrupt, layer-wise cosine divergence."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from vlm_icl.data import prepare_data
from vlm_icl.models import HIDDEN_STATES, load_model, require_capability
from vlm_icl.images import resolve_image
from vlm_icl.prompts import format_icl_answer
from vlm_icl.icl import (
    build_control_demo_set,
    build_icl_context_pools,
    process_icl_context_images,
    sample_per_label,
)
from vlm_icl.paths import build_analysis_path
from vlm_icl.eb_distance import cosine_divergence
from vlm_icl.evaluation import save_results

PartitionKey = Tuple[str, str]


@dataclass(frozen=True)
class ControlHiddenEmbConfig:
    model_path: str
    backend: str = "default"
    tensor_parallel_size: int = 1
    max_num_images: int = 64
    max_model_len: int = 32768
    seed: int = 42
    output_dir: Optional[str] = None
    tasks: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    sample_per_label_n: int = 10
    max_demos: int = 32
    k_values: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    enable_thinking: bool = False
    cot: bool = False
    data_root: Optional[str] = None
    data_source: str = "auto"


# ---------------------------------------------------------------------------
# Aggregation (reuses pattern from hidden_embedding.py)
# ---------------------------------------------------------------------------


def _aggregate_distances(per_query_distances, split_lookup, num_layers):
    """Aggregate per-datapoint distances into overall and by-split stats."""
    overall = {}
    for layer_idx in range(num_layers):
        vals = [v[layer_idx] for v in per_query_distances.values() if layer_idx in v]
        if vals:
            overall[f"layer_{layer_idx}"] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

    by_split = defaultdict(lambda: defaultdict(list))
    for key, layer_dists in per_query_distances.items():
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

    return {"overall": overall, "by_split": by_split_stats}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_control_hidden_emb(config: ControlHiddenEmbConfig) -> Dict[str, Any]:
    # 1. Load model (HF backend required for forward_hidden_states)
    print(f"Loading model from {config.model_path}...")
    model = load_model(config.model_path, enable_thinking=config.enable_thinking)
    require_capability(model, HIDDEN_STATES)
    print(f"Model loaded: {type(model).__name__}")

    # 2. Load data
    full_data = prepare_data(tasks=config.tasks, topics=config.topics, seed=config.seed, data_root=config.data_root, data_source=config.data_source)
    test_data = sample_per_label(full_data, config.sample_per_label_n, seed=config.seed)
    test_ids = {e["image_id"] for e in test_data}
    print(f"Test queries: {len(test_data)} entries ({config.sample_per_label_n} per label)")

    # 3. Build context pools (excluding test queries) + control demos
    pool_data = [e for e in full_data if e["image_id"] not in test_ids]
    context_pools = build_icl_context_pools(pool_data)
    control_demos, corrupted_answers = build_control_demo_set(
        context_pools, max_demos=config.max_demos, seed=config.seed,
    )

    prefer_pil = True  # HF backend

    # 4. Group by task
    task_groups: Dict[str, List[Dict]] = defaultdict(list)
    for entry in test_data:
        task_groups[entry["task"]].append(entry)

    leaf = f"spl{config.sample_per_label_n}_standard"
    output_dirs = []

    for task_name, task_entries in sorted(task_groups.items()):
        print(f"\n{'='*60}")
        print(f"Task: {task_name} ({len(task_entries)} entries)")
        print(f"{'='*60}")

        output_path = build_analysis_path(
            config.output_dir or "./output",
            "control_hidden_emb", "visual_reasoning", task_name,
            config.model_path, leaf,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        output_dirs.append(str(output_path))

        with open(output_path / "run_config.json", "w") as f:
            json.dump(asdict(config), f, indent=2, default=str)

        for key, demos in control_demos.items():
            if key[0] == task_name:
                with open(output_path / "control_demos.json", "w") as f:
                    json.dump([
                        {"image_id": d["entry"]["image_id"], "gt": d["question"]["ground_truth"]}
                        for d in demos
                    ], f, indent=2)

        # Phase A: Zero-shot embeddings
        print("\nPhase A: Extracting zero-shot embeddings...")
        zeroshot_embeddings = {}  # (image_id, question_type) -> Dict[int, Tensor]
        skipped_keys = set()
        split_lookup = {}

        for entry in tqdm(task_entries, desc="Zero-shot", unit="entry"):
            image_obj = entry.get("image")
            image = resolve_image(image_obj, prefer_pil=prefer_pil) if image_obj is not None else None
            extra_imgs = {}
            for extra_key in ("image2", "image3", "image4"):
                extra_obj = entry.get(extra_key)
                if extra_obj is not None:
                    extra_imgs[extra_key] = resolve_image(extra_obj, prefer_pil=prefer_pil)

            for question in entry.get("questions", []):
                qkey = (entry["image_id"], question["type"])
                split_lookup[qkey] = entry.get("split", "unknown")
                try:
                    hs = model.forward_hidden_states(
                        image=image, prompt=question["prompt"], **extra_imgs,
                    )
                    zeroshot_embeddings[qkey] = hs
                except torch.cuda.OutOfMemoryError:
                    tqdm.write(f"  CUDA OOM (zero-shot): {entry['image_id']}, skipping")
                    torch.cuda.empty_cache()
                    skipped_keys.add(qkey)
                except Exception as e:
                    tqdm.write(f"  Error (zero-shot): {e}, skipping")
                    skipped_keys.add(qkey)

        print(f"Zero-shot: {len(zeroshot_embeddings)} embeddings, {len(skipped_keys)} skipped")

        num_layers = max(
            (max(emb.keys()) + 1 for emb in zeroshot_embeddings.values()),
            default=0,
        )

        output_file = output_path / "hidden_embedding.json"
        results = {}

        # Phase B: ICL conditions
        print("\nPhase B: ICL embeddings + cosine divergence...")

        for condition in ("clean", "corrupt"):
            for k in config.k_values:
                if k > config.max_demos:
                    continue
                result_key = f"{condition}_k{k}"
                print(f"\n  {result_key}")

                # Build fixed ICL context for this condition/k
                fixed_contexts: Dict[PartitionKey, List[Dict[str, Any]]] = {}
                for pk, demos in control_demos.items():
                    subset = demos[:k]
                    if condition == "clean":
                        answers = [d["question"]["ground_truth"] for d in subset]
                    else:
                        answers = corrupted_answers[pk][:k]
                    fixed_contexts[pk] = process_icl_context_images(
                        subset, answers, None, format_icl_answer,
                        prefer_pil=prefer_pil,
                    )

                per_query_distances = {}

                for entry in tqdm(task_entries, desc=result_key, unit="entry"):
                    image_obj = entry.get("image")
                    image = resolve_image(image_obj, prefer_pil=prefer_pil) if image_obj is not None else None
                    extra_imgs = {}
                    for extra_key in ("image2", "image3", "image4"):
                        extra_obj = entry.get(extra_key)
                        if extra_obj is not None:
                            extra_imgs[extra_key] = resolve_image(extra_obj, prefer_pil=prefer_pil)

                    pk = (entry["task"], entry["split"])
                    context = fixed_contexts.get(pk)

                    for question in entry.get("questions", []):
                        qkey = (entry["image_id"], question["type"])
                        if qkey in skipped_keys or qkey not in zeroshot_embeddings:
                            continue

                        zs_emb = zeroshot_embeddings[qkey]

                        try:
                            icl_emb = model.forward_hidden_states(
                                image=image, prompt=question["prompt"],
                                context=context, **extra_imgs,
                            )
                            layer_dists = {}
                            for layer_idx in zs_emb:
                                if layer_idx in icl_emb:
                                    d = cosine_divergence(
                                        zs_emb[layer_idx].unsqueeze(0),
                                        icl_emb[layer_idx].unsqueeze(0),
                                    )
                                    layer_dists[layer_idx] = d.item()
                            per_query_distances[qkey] = layer_dists
                        except torch.cuda.OutOfMemoryError:
                            tqdm.write(f"  CUDA OOM ({result_key}): {entry['image_id']}, skipping")
                            torch.cuda.empty_cache()
                        except Exception as e:
                            tqdm.write(f"  Error ({result_key}): {e}, skipping")

                print(f"  {result_key}: {len(per_query_distances)} datapoints")

                # Aggregate and save incrementally
                icl_result = _aggregate_distances(per_query_distances, split_lookup, num_layers)
                results[result_key] = icl_result
                save_results(
                    results, output_file, model_path=config.model_path,
                    extra_metadata={
                        "task": task_name, "dataset": "visual_reasoning",
                        "sample_per_label": config.sample_per_label_n,
                        "k_values": list(config.k_values),
                        "seed": config.seed, "num_layers": num_layers,
                        "num_datapoints": len(zeroshot_embeddings),
                    },
                    format="json",
                )
                print(f"  Saved to {output_file}")

        # Summary table
        print(f"\nSummary for {task_name}:")
        print(f"  {'Condition':<15} {'Last Layer Mean':<20} {'Last Layer Std'}")
        print(f"  {'-'*50}")
        for rkey in sorted(results):
            last_layer_key = f"layer_{num_layers - 1}"
            stats = results[rkey]["overall"].get(last_layer_key, {})
            mean = stats.get("mean", float("nan"))
            std = stats.get("std", float("nan"))
            print(f"  {rkey:<15} {mean:<20.6f} {std:.6f}")

    print("\nDone!")
    return {"output_dirs": output_dirs}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> ControlHiddenEmbConfig:
    from vlm_icl.args import add_model_args, add_data_args, add_run_config_args

    parser = argparse.ArgumentParser(description="Control hidden embedding analysis.")
    add_model_args(parser)
    add_data_args(parser)
    add_run_config_args(parser)

    parser.add_argument("--sample_per_label", type=int, default=10)
    parser.add_argument("--max_demos", type=int, default=32)
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])

    args = parser.parse_args(argv)
    return ControlHiddenEmbConfig(
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
        data_root=args.data_root,
        data_source=args.data_source,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    result = run_control_hidden_emb(config)
    for d in result["output_dirs"]:
        print(f"Saved to: {d}")


if __name__ == "__main__":
    main()
