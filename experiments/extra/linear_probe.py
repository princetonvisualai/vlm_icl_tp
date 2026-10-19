"""Per-layer linear probe on frozen hidden embeddings: zero-shot vs ICL."""

import argparse
import random
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from vlm_icl.images import resolve_image
from vlm_icl.icl import (
    build_demo_pool,
    build_icl_context_pools,
    process_icl_context_images,
    select_from_demo_pool,
    shuffle_icl_answers,
)
from vlm_icl.paths import build_analysis_path
from vlm_icl.data import prepare_data
from vlm_icl.evaluation import save_results
from vlm_icl.models import HIDDEN_STATES, load_model, require_capability


def _extract_embeddings(model, task_entries, prefer_pil, context_fn=None):
    """Extract last-token hidden states for each entry x question.

    Args:
        context_fn: If provided, called with (entry, question) and should return
            kwargs dict with 'context' key (or None to skip).

    Returns:
        List of (key, layer_dict, ground_truth) tuples.
    """
    samples = []
    for entry in tqdm(task_entries, desc="Extracting", unit="entry"):
        image_obj = entry.get("image")
        image = resolve_image(image_obj, prefer_pil=prefer_pil) if image_obj is not None else None
        extra_imgs = {}
        for extra_key in ("image2", "image3", "image4"):
            extra_obj = entry.get(extra_key)
            if extra_obj is not None:
                extra_imgs[extra_key] = resolve_image(extra_obj, prefer_pil=prefer_pil)

        for question in entry.get("questions", []):
            key = (entry["image_id"], question["type"])
            gt = question["ground_truth"]

            extra_kwargs = {}
            if context_fn is not None:
                ctx_result = context_fn(entry, question)
                if ctx_result is None:
                    continue
                extra_kwargs = ctx_result

            try:
                hs = model.forward_hidden_states(
                    image=image,
                    prompt=question["prompt"],
                    **extra_kwargs,
                    **extra_imgs,
                )
                samples.append((key, hs, gt))
            except torch.cuda.OutOfMemoryError:
                tqdm.write(f"  CUDA OOM: image_id={entry['image_id']}, skipping")
                torch.cuda.empty_cache()
            except Exception as e:
                tqdm.write(f"  Error: {e}, skipping")

    return samples


def _train_probes(samples, num_layers, test_fraction, reg_strength):
    """Train logistic regression probes per layer and return accuracy dict."""
    # Collect labels
    labels = [gt for _, _, gt in samples]
    le = LabelEncoder()
    le.fit(labels)
    y_all = le.transform(labels)
    n_classes = len(le.classes_)

    if n_classes < 2:
        print(f"  Skipping: only {n_classes} class(es)")
        return None

    per_layer = {}
    for layer_idx in range(num_layers):
        X = np.stack([s[1][layer_idx].float().numpy() for s in samples])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_all, test_size=test_fraction, stratify=y_all, random_state=42,
        )
        clf = LogisticRegression(
            C=1.0 / reg_strength, max_iter=5000, solver="lbfgs",
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        per_layer[f"layer_{layer_idx}"] = {
            "accuracy": float(acc),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_classes": n_classes,
        }

    return per_layer


def main():
    from vlm_icl.args import add_data_args, add_icl_args, add_model_args, add_run_config_args

    parser = argparse.ArgumentParser(description="Linear probe analysis: zero-shot vs ICL")
    add_model_args(parser)
    add_data_args(parser)
    add_icl_args(parser)
    add_run_config_args(parser)
    parser.set_defaults(sample_per_split=100, dataset="visual_reasoning_many")
    parser.add_argument("--icl_counts", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--reg_strength", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(args.seed)

    # Phase 0: Setup
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, enable_thinking=args.enable_thinking)
    require_capability(model, HIDDEN_STATES)
    print(f"Model loaded: {type(model).__name__}")

    print(f"\nPreparing sampled dataset ({args.dataset})...")
    data = prepare_data(
        dataset=args.dataset, tasks=args.tasks,
        sample_per_split=args.sample_per_split, seed=args.seed,
        data_root=args.data_root,
        data_source=args.data_source,
    )
    print(f"Sampled data: {len(data)} entries")

    # Expanded dataset for zero-shot (match total ICL sample count)
    if 0 in args.icl_counts and args.num_icl_sample_sets > 1:
        zs_sps = args.sample_per_split * args.num_icl_sample_sets
        zs_data = prepare_data(
            dataset=args.dataset, tasks=args.tasks,
            sample_per_split=zs_sps, seed=args.seed,
            data_root=args.data_root,
            data_source=args.data_source,
        )
        zs_task_groups = defaultdict(list)
        for entry in zs_data:
            zs_task_groups[entry["task"]].append(entry)
        print(f"Zero-shot expanded data: {len(zs_data)} entries (sps={zs_sps})")
    else:
        zs_task_groups = None

    if not data:
        print("No data entries found!")
        return

    # Load full dataset for ICL context pools
    icl_pool_data = prepare_data(dataset=args.dataset, tasks=args.tasks, data_root=args.data_root, data_source=args.data_source)
    print(f"ICL context pool: {len(icl_pool_data)} entries")

    context_pools = build_icl_context_pools(icl_pool_data)
    demo_pool = build_demo_pool(context_pools, seed=args.seed)

    # Group data by task
    task_groups = defaultdict(list)
    for entry in data:
        task_groups[entry["task"]].append(entry)

    prefer_pil = True
    answer_rng = random.Random(args.seed + 1)

    for task_name, task_entries in sorted(task_groups.items()):
        print(f"\n{'='*60}")
        print(f"Task: {task_name} ({len(task_entries)} entries)")
        print(f"{'='*60}")

        output_path = build_analysis_path(
            args.output_dir, "linear_probe", args.dataset, task_name, args.model_path,
            f"sps{args.sample_per_split}_k{args.num_icl_sample_sets}",
        )
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / "linear_probe.json"
        results = {}
        num_layers = None

        for num_icl in args.icl_counts:
            print(f"\n  Condition: {num_icl}-shot")

            if num_icl == 0:
                # Zero-shot embeddings (use expanded dataset if available)
                zs_entries = zs_task_groups[task_name] if zs_task_groups is not None else task_entries
                samples = _extract_embeddings(model, zs_entries, prefer_pil)
            else:
                # ICL embeddings — each sample set is a separate training sample
                samples = []
                for j in range(args.num_icl_sample_sets):
                    set_seed = args.seed + j if args.seed is not None else None

                    def make_context_fn(seed_val):
                        def context_fn(entry, question):
                            context = select_from_demo_pool(
                                entry, question, demo_pool, num_icl, seed=seed_val,
                            )
                            if context is None:
                                return None
                            shuffled = shuffle_icl_answers(context, None, answer_rng)
                            icl_ctx = process_icl_context_images(
                                context, shuffled, None, prefer_pil=prefer_pil,
                            )
                            return {"context": icl_ctx}
                        return context_fn

                    batch = _extract_embeddings(
                        model, task_entries, prefer_pil,
                        context_fn=make_context_fn(set_seed),
                    )
                    samples.extend(batch)

            if not samples:
                print(f"  No samples for {num_icl}-shot, skipping")
                continue

            # Determine num_layers from first sample
            if num_layers is None:
                num_layers = max(samples[0][1].keys()) + 1
                print(f"  Detected {num_layers} layers")

            print(f"  Training probes on {len(samples)} samples...")
            per_layer = _train_probes(
                samples, num_layers, args.test_fraction, args.reg_strength,
            )
            if per_layer is None:
                continue

            results[f"icl_{num_icl}"] = {"per_layer": per_layer}

            # Save incrementally
            save_results(
                results, output_file, model_path=args.model_path,
                extra_metadata={
                    "task": task_name, "dataset": args.dataset,
                    "sample_per_split": args.sample_per_split,
                    "num_icl_sample_sets": args.num_icl_sample_sets,
                    "icl_counts": args.icl_counts, "seed": args.seed,
                    "num_layers": num_layers, "reg_strength": args.reg_strength,
                    "test_fraction": args.test_fraction,
                },
                format="json",
            )
            print(f"  Saved to {output_file}")

        # Summary
        if results and num_layers:
            print(f"\nSummary for {task_name}:")
            print(f"  {'Condition':<12} {'Last Layer Acc':<16} {'N_train':<10} {'N_test'}")
            print(f"  {'-'*50}")
            for num_icl in args.icl_counts:
                key = f"icl_{num_icl}"
                if key in results:
                    last = results[key]["per_layer"].get(f"layer_{num_layers - 1}", {})
                    print(f"  {num_icl}-shot{'':<6} {last.get('accuracy', float('nan')):<16.4f} "
                          f"{last.get('n_train', '?'):<10} {last.get('n_test', '?')}")

    print("\nDone!")


if __name__ == "__main__":
    main()
