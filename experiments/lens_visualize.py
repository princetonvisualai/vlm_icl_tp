"""LatentLens / Logit Lens visual token decoding — decode what text each query-image token encodes at each layer."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

import latentlens

from vlm_icl.args import add_model_args, add_data_args, add_icl_args, add_run_config_args
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
from vlm_icl.evaluation import save_results
from vlm_icl.lens import (
    detect_num_layers,
    flatten_ids,
    get_lm_head_and_norm,
    logitlens_project,
    neighbor_to_dict,
    resolve_grid_thw,
    resolve_image_token_id,
    resolve_latentlens_layers,
    split_runs,
)


def _get_query_token_info(tokenizer, input_ids, image_grid_thw):
    """Find query image token positions and grid shape.

    Returns (positions, grid_thw) where positions is a list of ints
    and grid_thw is a (T, H, W) tuple.
    """
    flat_ids = flatten_ids(input_ids)
    image_token_id = resolve_image_token_id(tokenizer, flat_ids)
    if image_token_id is None:
        return None, None

    positions = torch.where(flat_ids == image_token_id)[0].tolist()
    runs = split_runs(positions)
    if not runs:
        return None, None

    query_run = runs[-1]  # last image = query
    query_image_index = len(runs) - 1
    grid_thw = resolve_grid_thw(image_grid_thw, query_image_index, len(query_run))
    return query_run, grid_thw


def _search_layer(index, hidden_states, layer, positions, top_k):
    """Search the index for one layer, returning list-of-lists of neighbor dicts."""
    # hidden_states[0] = embeddings, hidden_states[L] = layer L output
    hs = hidden_states[layer]  # (batch, seq_len, hidden_dim)
    query = hs[0, positions, :]  # (num_tokens, hidden_dim)
    query = torch.nn.functional.normalize(query, dim=-1)
    neighbors = index.search(query, top_k=top_k, layers=[layer])
    return [[neighbor_to_dict(n) for n in token_neighbors] for token_neighbors in neighbors]


def _run_condition(model, index, image, prompt, layers, top_k, context=None, extra_imgs=None):
    """Run one forward pass and return latentlens search results + grid_thw.

    Returns (layer_results, grid_thw) or (None, None) on failure.
    """
    extra_imgs = extra_imgs or {}
    messages = model.build_messages(image, prompt, context=context, **extra_imgs)
    inputs = model.prepare_inputs(messages)

    with torch.no_grad():
        hidden_states = latentlens.get_hidden_states(model.model, **inputs)

    tokenizer = model.processor.tokenizer if hasattr(model.processor, "tokenizer") else model.processor
    query_positions, grid_thw = _get_query_token_info(
        tokenizer, inputs["input_ids"], inputs.get("image_grid_thw")
    )
    if query_positions is None:
        return None, None

    layer_results = {}
    for layer in layers:
        layer_results[f"layer_{layer}"] = _search_layer(index, hidden_states, layer, query_positions, top_k)

    return layer_results, list(grid_thw)


def _run_condition_logitlens(model, lm_head, final_norm, image, prompt, layers, top_k,
                             context=None, extra_imgs=None):
    """Run one forward pass and return logit lens results + grid_thw."""
    extra_imgs = extra_imgs or {}
    messages = model.build_messages(image, prompt, context=context, **extra_imgs)
    inputs = model.prepare_inputs(messages)

    with torch.no_grad():
        hidden_states = latentlens.get_hidden_states(model.model, **inputs)

    tokenizer = model.processor.tokenizer if hasattr(model.processor, "tokenizer") else model.processor
    query_positions, grid_thw = _get_query_token_info(
        tokenizer, inputs["input_ids"], inputs.get("image_grid_thw")
    )
    if query_positions is None:
        return None, None

    layer_results = {}
    for layer in layers:
        layer_results[f"layer_{layer}"] = logitlens_project(
            hidden_states, layer, query_positions, lm_head, final_norm, tokenizer, top_k,
        )

    return layer_results, list(grid_thw)


def main():
    parser = argparse.ArgumentParser(description="LatentLens visual token decoding")
    add_model_args(parser)
    add_data_args(parser)
    add_icl_args(parser)
    add_run_config_args(parser)
    parser.add_argument("--icl_counts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--n_per_label", type=int, default=10,
                        help="Sample N per (task, split, ground_truth). Applied in both --expected_bias and default branches.")
    parser.add_argument("--expected_bias", type=str, default=None,
                        help="Filter value for expected_bias field (e.g. 'octagon')")
    parser.add_argument("--num_demo_sets", type=int, default=10,
                        help="Number of demo sets to sample per query (default: 5)")
    parser.add_argument("--latentlens_index", type=str, default=None,
                        help="LatentLens index: a local directory or a Hub name (required for --lens_type latentlens)")
    parser.add_argument("--lens_type", type=str, choices=["latentlens", "logitlens"],
                        default="latentlens", help="Lens type: latentlens (index search) or logitlens (norm+lm_head projection)")
    args = parser.parse_args()
    if args.lens_type == "latentlens" and args.latentlens_index is None:
        parser.error("--latentlens_index is required for --lens_type latentlens")

    if args.seed is None:
        args.seed = 42
    random.seed(args.seed)
    answer_rng = random.Random(args.seed + 1)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, enable_thinking=args.enable_thinking)
    args.layers = resolve_latentlens_layers(args.model_path, args.layers, detect_num_layers(model))

    # Load lens resources
    index, lm_head, final_norm = None, None, None
    if args.lens_type == "latentlens":
        index_path = Path(args.latentlens_index)
        if index_path.is_dir():
            print(f"Loading latentlens index from local directory: {index_path}")
            index = latentlens.ContextualIndex.from_directory(str(index_path), layers=args.layers)
        else:
            print(f"Loading latentlens index from HuggingFace: {args.latentlens_index}")
            index = latentlens.ContextualIndex.from_pretrained(args.latentlens_index, layers=args.layers)
        index.to("cuda")
        print("Index loaded and moved to GPU.")
    else:
        lm_head, final_norm = get_lm_head_and_norm(model)
        print("Logit lens: extracted lm_head and final_norm.")

    # Prepare data
    if args.expected_bias is not None:
        # Targeted query selection: complex split, expected_bias filter, label-balanced sampling
        all_data = prepare_data(dataset="visual_reasoning", tasks=args.tasks, seed=args.seed, data_root=args.data_root, data_source=args.data_source)
        complex_data = [e for e in all_data if e["split"] == "complex"]

        subset = [
            e for e in complex_data
            if any(q.get("expected_bias") == args.expected_bias for q in e.get("questions", []))
        ]
        print(f"Filtered to {len(subset)} entries with expected_bias='{args.expected_bias}'")

        data = sample_per_label(subset, n_per_label=args.n_per_label, seed=args.seed)
        print(f"Sampled {len(data)} query entries ({args.n_per_label} per label)")
    else:
        # Original selection: stratified by (task, split)
        data = prepare_data(dataset=args.dataset, tasks=args.tasks,
                            sample_per_split=args.sample_per_split, seed=args.seed, data_root=args.data_root, data_source=args.data_source)
        data = sample_per_label(data, n_per_label=args.n_per_label, seed=args.seed)
        print(f"Sampled data: {len(data)} entries ({args.n_per_label} per label)")

        all_data = prepare_data(dataset="visual_reasoning", tasks=args.tasks, data_root=args.data_root, data_source=args.data_source)
        complex_data = [e for e in all_data if e["split"] == "complex"]

    if not data:
        print("No data entries found!")
        return

    context_pools = build_icl_context_pools(complex_data)
    demo_pool = build_demo_pool(context_pools, seed=args.seed)

    # Group by task
    task_groups = defaultdict(list)
    for entry in data:
        task_groups[entry["task"]].append(entry)

    for task_name, task_entries in sorted(task_groups.items()):
        print(f"\n{'='*60}\nTask: {task_name} ({len(task_entries)} entries)\n{'='*60}")

        if args.expected_bias is not None:
            leaf = f"npl{args.n_per_label}_{args.expected_bias}"
        else:
            leaf_parts = []
            if args.sample_per_split is not None:
                leaf_parts.append(f"sps{args.sample_per_split}")
            leaf_parts.append(f"npl{args.n_per_label}")
            leaf = "_".join(leaf_parts)
        if args.lens_type != "latentlens":
            leaf = f"{leaf}_{args.lens_type}"

        output_path = build_analysis_path(
            args.output_dir, "formal_lens_visualize", args.dataset,
            task_name, args.model_path, leaf,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / "formal_lens_visualize.json"

        results = {}

        for entry in tqdm(task_entries, desc=f"{task_name}", unit="entry"):
            image_obj = entry.get("image")
            image = resolve_image(image_obj, prefer_pil=True) if image_obj is not None else None
            extra_imgs = {}
            for extra_key in ("image2", "image3", "image4"):
                extra_obj = entry.get(extra_key)
                if extra_obj is not None:
                    extra_imgs[extra_key] = resolve_image(extra_obj, prefer_pil=True)

            for question in entry.get("questions", []):
                key = f"{entry['image_id']}__{question['type']}"
                entry_result = {
                    "image_id": entry["image_id"],
                    "question_type": question["type"],
                    "ground_truth": question["ground_truth"],
                    "expected_bias": question.get("expected_bias"),
                    "distract": entry.get("distract"),
                    "grid_thw": None,
                    "conditions": {},
                }

                # Zero-shot
                try:
                    if args.lens_type == "latentlens":
                        layer_res, grid_thw = _run_condition(
                            model, index, image, question["prompt"],
                            args.layers, args.top_k, extra_imgs=extra_imgs,
                        )
                    else:
                        layer_res, grid_thw = _run_condition_logitlens(
                            model, lm_head, final_norm, image, question["prompt"],
                            args.layers, args.top_k, extra_imgs=extra_imgs,
                        )
                    if layer_res is not None:
                        entry_result["grid_thw"] = grid_thw
                        entry_result["conditions"]["zero_shot"] = layer_res
                except torch.cuda.OutOfMemoryError:
                    tqdm.write(f"  OOM (zero-shot): {entry['image_id']}, skipping")
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    tqdm.write(f"  Error (zero-shot): {e}, skipping")
                    continue

                # ICL conditions — use "complex" split key to match demo pool
                demo_entry = {**entry, "split": "complex"}
                for num_icl in args.icl_counts:
                    demo_set_results = []
                    for ds_idx in range(args.num_demo_sets):
                        demo_seed = args.seed + ds_idx
                        ds_answer_rng = random.Random(demo_seed + 1)
                        context = select_from_demo_pool(
                            demo_entry, question, demo_pool, num_icl, seed=demo_seed,
                        )
                        if context is None:
                            continue

                        shuffled_answers = shuffle_icl_answers(context, None, ds_answer_rng)
                        icl_context = process_icl_context_images(
                            context, shuffled_answers, None, prefer_pil=True,
                        )

                        try:
                            if args.lens_type == "latentlens":
                                layer_res, grid_thw = _run_condition(
                                    model, index, image, question["prompt"],
                                    args.layers, args.top_k,
                                    context=icl_context, extra_imgs=extra_imgs,
                                )
                            else:
                                layer_res, grid_thw = _run_condition_logitlens(
                                    model, lm_head, final_norm, image, question["prompt"],
                                    args.layers, args.top_k,
                                    context=icl_context, extra_imgs=extra_imgs,
                                )
                            if layer_res is not None:
                                demo_set_results.append(layer_res)
                                if entry_result["grid_thw"] is None:
                                    entry_result["grid_thw"] = grid_thw
                        except torch.cuda.OutOfMemoryError:
                            tqdm.write(f"  OOM (icl_{num_icl}, ds{ds_idx}): {entry['image_id']}, skipping")
                            torch.cuda.empty_cache()
                        except Exception as e:
                            tqdm.write(f"  Error (icl_{num_icl}, ds{ds_idx}): {e}, skipping")

                    if demo_set_results:
                        entry_result["conditions"][f"icl_{num_icl}"] = demo_set_results

                if entry_result["conditions"]:
                    results[key] = entry_result

        # Save results for this task
        save_results(
            results, output_file, model_path=args.model_path,
            extra_metadata={
                "task": task_name, "dataset": args.dataset,
                "lens_type": args.lens_type,
                "layers": args.layers, "top_k": args.top_k,
                "icl_counts": args.icl_counts,
                "sample_per_split": args.sample_per_split,
                "n_per_label": args.n_per_label,
                "seed": args.seed,
                "num_demo_sets": args.num_demo_sets,
            },
            format="json",
        )
        print(f"Saved {len(results)} entries to {output_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
