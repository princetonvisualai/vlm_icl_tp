"""VLM inference on visual_reasoning dataset with optional ICL and attention visualization."""

import argparse
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from tqdm import tqdm

from vlm_icl.data import prepare_data
from vlm_icl.models import ATTENTIONS, load_model
from vlm_icl.images import resolve_image
from vlm_icl.prompts import format_icl_answer
from vlm_icl.icl import (
    apply_special_mode,
    build_icl_context_pools,
    build_demo_pool,
    build_label_space,
    corrupt_icl_answers,
    ensure_gt_demo,
    reposition_gt_demo,
    select_from_demo_pool,
    validate_special_mode,
    shuffle_icl_answers,
    process_icl_context_images,
)
from vlm_icl.paths import derive_run_config
from vlm_icl.attention import (
    build_heatmap_output_path,
    save_heatmap_sample,
    write_heatmap_metadata,
)


def cleanup_model(model) -> None:
    """Best-effort model offload and GPU cache cleanup."""
    if model is None:
        return
    try:
        if hasattr(model, "to"):
            model.to("cpu")
    except Exception:
        pass
    try:
        del model
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def inference(
    model_path: str,
    backend: str = "default",
    tensor_parallel_size: int = 1,
    max_num_images: int = 64,
    max_model_len: int = 32768,
    allowed_local_media_path: Optional[str] = None,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    icl: bool = False,
    num_icl_examples: int = 2,
    num_icl_sample_sets: int = 1,
    max_samples: Optional[int] = None,
    image_attentionmap: bool = False,
    k: int = 1,
    special: Optional[str] = None,
    cot: bool = False,
    query_gt_demo: Optional[str] = None,
    demo_pool: Optional[Dict] = None,
    dataset: str = "visual_reasoning",
    enable_thinking: bool = False,
    data_root: Optional[str] = None,
    data_source: str = "auto",
    model=None,
    **data_kwargs
) -> List[Dict[str, Any]]:
    """Run inference on dataset with optional ICL/attention support. Returns results list."""
    # Set random seed for reproducibility (None = non-deterministic)
    if seed is not None:
        random.seed(seed)
    validate_special_mode(special, icl, num_icl_examples)

    owns_model = model is None
    try:
        if owns_model:
            print(f"Loading model from {model_path}...")

            # Use eager attention implementation if visualization is requested
            # (required for Qwen3VL to output attention weights)
            model_kwargs = {}
            if image_attentionmap:
                if backend == "vllm":
                    print("Warning: Attention visualization not supported with vllm backend")
                    image_attentionmap = False
                else:
                    model_kwargs['attn_implementation'] = 'eager'
                    print("Using 'eager' attention implementation for attention visualization")

            # Add engine-backend-specific kwargs (vLLM)
            if backend == "vllm":
                model_kwargs['tensor_parallel_size'] = tensor_parallel_size
                model_kwargs['max_num_images'] = max_num_images
                model_kwargs['max_model_len'] = max_model_len
                if allowed_local_media_path is not None:
                    model_kwargs["allowed_local_media_path"] = allowed_local_media_path

            model = load_model(model_path, backend=backend, enable_thinking=enable_thinking, **model_kwargs)
            print(f"Model loaded successfully! (Type: {type(model).__name__})")

        # Prepare data
        print(f"\nPreparing dataset: {dataset}...")
        data = prepare_data(dataset=dataset, seed=seed, **data_kwargs, data_root=data_root, data_source=data_source)
        print(f"Dataset prepared with {len(data)} entries")

        if not data:
            raise ValueError("No data entries found!")

        # When sample_per_split is used with ICL, load the full dataset for
        # ICL context pools so demos are not restricted to the sampled subset.
        icl_pool_data = None
        if icl and data_kwargs.get("sample_per_split") is not None:
            full_data_kwargs = {k: v for k, v in data_kwargs.items() if k != "sample_per_split"}
            icl_pool_data = prepare_data(dataset=dataset, **full_data_kwargs, data_root=data_root, data_source=data_source)
            print(f"ICL context pool: {len(icl_pool_data)} entries (full dataset)")

        # Build demo pool for ICL (shared across all queries)
        if icl and demo_pool is None:
            pool_source = icl_pool_data if icl_pool_data is not None else data
            context_pools = build_icl_context_pools(pool_source)
            demo_pool = build_demo_pool(context_pools, seed=seed)

        # Attention visualization bookkeeping
        if image_attentionmap:
            if backend == "vllm":
                # Should already be handled above, but keep this for robustness.
                print("Warning: Attention visualization not supported with vllm backend; disabling.")
                image_attentionmap = False
            elif ATTENTIONS not in model.capabilities:
                print("Warning: Model does not support attention extraction; disabling --image_attentionmap.")
                image_attentionmap = False
            else:
                if output_dir is None:
                    output_dir = "./output"
                    print("No --output_dir provided; defaulting to ./output for attention visualizations.")
                print("Attention visualization enabled.")

        # HF processors are most reliable with PIL.Image inputs; engine backends use data URLs.
        prefer_pil = backend != "vllm"

        return _run_batch(
            model=model,
            data=data,
            model_path=model_path,
            seed=seed,
            output_dir=output_dir,
            image_attentionmap=image_attentionmap,
            data_kwargs=data_kwargs,
            icl=icl,
            num_icl_examples=num_icl_examples,
            num_icl_sample_sets=num_icl_sample_sets,
            max_samples=max_samples,
            k=k,
            special=special,
            cot=cot,
            prefer_pil=prefer_pil,
            query_gt_demo=query_gt_demo,
            demo_pool=demo_pool,
        )
    finally:
        if owns_model:
            cleanup_model(model)


def _safe_path_component(value: Any) -> str:
    s = str(value) if value is not None else "unknown"
    s = s.replace("/", "_").replace("\\", "_").strip()
    return s if s else "unknown"


def _classify_inference_exception(exc: Exception) -> str:
    msg = str(exc)
    if "Cannot load local files without `--allowed-local-media-path`" in msg:
        return "local_media_blocked"
    if "max_model_len" in msg or "maximum model length" in msg or "context length" in msg:
        return "context_too_long"
    if "limit_mm_per_prompt" in msg or "max_num_images" in msg:
        return "too_many_images"
    return exc.__class__.__name__


def _icl_context_builder(
    demo_pool: Dict,
    num_icl_examples: int,
    special: Optional[str],
    query_gt_demo: Optional[str],
    prefer_pil: bool,
    seed: Optional[int],
):
    """Return ``build(entry, question, set_seed)``, which selects one ICL demo set for a query.

    ``build`` returns ``(context, shuffled_answers, icl_context)`` or None when the pool cannot
    fill the demo set. Demo answers are shuffled or corrupted with one answer RNG shared by
    every query, seeded with ``seed + 1``.
    """
    answer_rng = random.Random(seed + 1 if seed is not None else None)
    if special and special.startswith("corrupt_icd"):
        label_spaces = build_label_space(demo_pool)
    else:
        label_spaces = None

    def build(entry, question, set_seed):
        context = select_from_demo_pool(
            entry, question, demo_pool, num_icl_examples, set_seed,
        )
        if context is None:
            return None
        if query_gt_demo is not None:
            key = (entry["task"], entry["split"])
            pool_candidates = [c for c in demo_pool[key] if c['entry']['image_id'] != entry['image_id']]
            gt_rng = random.Random(set_seed)  # None → OS entropy
            context = ensure_gt_demo(context, question["ground_truth"], pool_candidates, gt_rng)
            context = reposition_gt_demo(context, question["ground_truth"], query_gt_demo)

        if label_spaces is not None:
            key = (entry["task"], entry["split"])
            shuffled_answers = corrupt_icl_answers(context, special, answer_rng, label_spaces[key])
        else:
            shuffled_answers = shuffle_icl_answers(context, special, answer_rng)
        icl_context = process_icl_context_images(
            context, shuffled_answers, special, format_icl_answer, prefer_pil=prefer_pil
        )
        return context, shuffled_answers, icl_context

    return build


def _run_batch(
    model,
    data,
    model_path: str,
    seed: Optional[int],
    output_dir: Optional[str],
    image_attentionmap: bool,
    data_kwargs: Dict[str, Any],
    icl: bool = False,
    num_icl_examples: int = 0,
    num_icl_sample_sets: int = 1,
    max_samples: Optional[int] = None,
    k: int = 1,
    special: Optional[str] = None,
    cot: bool = False,
    prefer_pil: bool = False,
    query_gt_demo: Optional[str] = None,
    demo_pool: Optional[Dict] = None,
):
    """Run batch inference, zero-shot or with ICL demos. Generates k answers per query.

    With ICL, each query is answered with ``num_icl_sample_sets`` demo sets; set ``i`` selects
    its demos with seed ``seed + i``.
    """
    if icl:
        build_context = _icl_context_builder(
            demo_pool, num_icl_examples, special, query_gt_demo, prefer_pil, seed,
        )
    num_sets = num_icl_sample_sets if icl else 1
    results = []
    total_pairs = sum(len(entry['questions']) for entry in data)
    total_work = total_pairs * num_sets
    skipped = 0
    skip_reasons: Counter[str] = Counter()
    skip_by_task_split: Counter[tuple] = Counter()

    # Cap by max_samples for cleaner UX
    if max_samples is not None:
        total_work = min(total_work, max_samples)

    limit_note = f" (limited to {max_samples})" if max_samples is not None else ""
    samples_note = f" x{k}" if k > 1 else ""
    sets_note = f" x{num_sets} sets" if num_sets > 1 else ""
    label = "ICL inference" if icl else "inference"
    print(f"\nRunning {label} on all {total_pairs} samples{samples_note}{sets_note}{limit_note}...")

    # Lazy per-task heatmap output paths (populated on first sample per task)
    heatmap_paths: Dict[str, Path] = {}

    limit_reached = False
    with tqdm(total=total_work, desc="ICL inference" if icl else "Standard inference", unit="sample") as pbar:
        for entry in data:
            if limit_reached:
                break

            image_obj = apply_special_mode(entry['image'], special, is_icd=False)
            image = resolve_image(image_obj, prefer_pil=prefer_pil) if image_obj is not None else None
            # Resolve extra images if present (multi-image datasets)
            extra_imgs = {}
            for extra_key in ('image2', 'image3', 'image4'):
                extra_obj = entry.get(extra_key)
                if extra_obj is not None:
                    extra_obj = apply_special_mode(extra_obj, special, is_icd=False)
                    extra_imgs[extra_key] = resolve_image(extra_obj, prefer_pil=prefer_pil) if extra_obj is not None else None

            for q_idx, question in enumerate(entry['questions'], 1):
                if limit_reached:
                    break

                context = shuffled_answers = icl_context = None
                if icl:
                    # The first demo set is checked once; a query the pool cannot serve skips all sets.
                    built = build_context(entry, question, seed)
                    if built is None:
                        tqdm.write("  Warning: Insufficient ICL context, skipping query")
                        skipped += 1
                        skip_reasons["insufficient_icl_context"] += 1
                        skip_by_task_split[(entry.get("task"), entry.get("split"), "insufficient_icl_context")] += 1
                        pbar.update(num_sets)
                        continue
                    context, shuffled_answers, icl_context = built

                for icl_set_idx in range(num_sets):
                    if max_samples is not None and len(results) >= max_samples:
                        pbar.update(num_sets - icl_set_idx)  # Update remaining
                        limit_reached = True
                        break

                    try:
                        if icl_set_idx > 0:
                            built = build_context(entry, question, seed + icl_set_idx if seed is not None else None)
                            if built is None:
                                skipped += 1
                                continue
                            context, shuffled_answers, icl_context = built

                        attention_data = None
                        answers: List[str] = []

                        if (
                            image_attentionmap
                            and ATTENTIONS in model.capabilities
                            and image is not None
                        ):
                            try:
                                answer, attention_data = model.generate_with_attentions(
                                    image=image,
                                    prompt=question["prompt"],
                                    context=icl_context,
                                    cot=cot,
                                    **extra_imgs,
                                )
                                answers.append(answer)
                            except (NotImplementedError, Exception) as exc:
                                tqdm.write(f"  Attention failed, using standard: {exc}")
                                attention_data = None

                        if not answers:
                            answers.append(model.generate(image=image, prompt=question["prompt"], context=icl_context, cot=cot, **extra_imgs))

                        # Generate remaining samples (pass@k) without attentions
                        for _ in range(max(0, k - len(answers))):
                            answers.append(model.generate(image=image, prompt=question["prompt"], context=icl_context, cot=cot, **extra_imgs))

                        demo_records = None
                        if icl:
                            demo_records = [
                                {
                                    'image_id': ctx['entry']['image_id'],
                                    'task': ctx['entry'].get('task'),
                                    'topic': ctx['entry'].get('topic'),
                                    'prompt': ctx['question']['prompt'],
                                    'ground_truth': shuffled_answers[i]
                                }
                                for i, ctx in enumerate(context)
                            ]
                        result = {
                            'image_id': entry['image_id'],
                            'task': entry.get('task'),
                            'topic': entry.get('topic'),
                            'split': entry.get('split'),
                            'question_type': question['type'],
                            'prompt': question['prompt'],
                            'ground_truth': question['ground_truth'],
                            'expected_bias': question.get('expected_bias'),
                            'model_answers': answers,
                            'metadata': question.get('metadata', {}),
                        }
                        if icl:
                            result['icl_context'] = demo_records
                        results.append(result)

                        if image_attentionmap and attention_data and image is not None:
                            task_key = entry.get("task", "unknown")
                            if task_key not in heatmap_paths:
                                heatmap_paths[task_key] = build_heatmap_output_path(
                                    base_dir=output_dir or "./output",
                                    model_path=model_path,
                                    task_name=task_key,
                                    icl=icl,
                                    num_icl_examples=num_icl_examples if icl else 0,
                                    run_config=derive_run_config(special, cot),
                                )
                                icl_metadata = (
                                    {"num_icl_examples": num_icl_examples, "num_icl_sample_sets": num_icl_sample_sets}
                                    if icl else {}
                                )
                                write_heatmap_metadata(
                                    heatmap_paths[task_key],
                                    mode="batch_icl" if icl else "batch_standard",
                                    model_path=model_path,
                                    seed=seed,
                                    task_name=task_key,
                                    tasks=data_kwargs.get("tasks"),
                                    topics=data_kwargs.get("topics"),
                                    sample_per_split=data_kwargs.get("sample_per_split"),
                                    max_samples=max_samples,
                                    k=k,
                                    **icl_metadata,
                                    special=special,
                                )

                            sample_dir = (
                                heatmap_paths[task_key]
                                / _safe_path_component(entry.get("split"))
                                / _safe_path_component(entry.get("image_id"))
                                / f"q{q_idx}"
                            )
                            if icl:
                                sample_dir = sample_dir / f"set{icl_set_idx + 1}"

                            query_imgs = [image] + [
                                extra_imgs[k] for k in ('image2', 'image3', 'image4')
                                if k in extra_imgs and extra_imgs[k] is not None
                            ]
                            result_data = {
                                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                                "image_id": entry.get("image_id"),
                                "task": entry.get("task"),
                                "topic": entry.get("topic"),
                                "split": entry.get("split"),
                                "question_type": question.get("type"),
                                "query": question.get("prompt"),
                                "ground_truth": question.get("ground_truth"),
                                "expected_bias": question.get("expected_bias"),
                                "model_answers": answers,
                                "model_path": model_path,
                                "seed": seed,
                            }
                            demo_kwargs = {}
                            if icl:
                                result_data["num_icl_examples"] = num_icl_examples
                                result_data["icl_set_index"] = icl_set_idx
                                result_data["context_examples"] = demo_records
                                ctx_images_flat = []
                                ctx_demo_sizes = []
                                for c in icl_context:
                                    demo = [c[k] for k in ("image", "image2", "image3", "image4") if c.get(k) is not None]
                                    ctx_images_flat.extend(demo)
                                    ctx_demo_sizes.append(len(demo))
                                demo_kwargs = {
                                    "icl_context_images": ctx_images_flat,
                                    "icl_context_demo_sizes": ctx_demo_sizes,
                                }

                            save_heatmap_sample(
                                attention_data=attention_data,
                                image=image,
                                sample_dir=sample_dir,
                                answer_text=answers[0] if answers else None,
                                result_data=result_data,
                                **demo_kwargs,
                                query_images=query_imgs,
                            )
                    except torch.cuda.OutOfMemoryError:
                        tqdm.write("  CUDA OOM error, skipping")
                        torch.cuda.empty_cache()
                        skipped += 1
                        skip_reasons["cuda_oom"] += 1
                        skip_by_task_split[(entry.get("task"), entry.get("split"), "cuda_oom")] += 1
                    except Exception as e:
                        reason = _classify_inference_exception(e)
                        set_note = f" icl_set={icl_set_idx}" if icl else ""
                        tqdm.write(
                            f"  Error[{reason}] task={entry.get('task')} split={entry.get('split')} "
                            f"image_id={entry.get('image_id')} q={q_idx}{set_note}: {e}"
                        )
                        skipped += 1
                        skip_reasons[reason] += 1
                        skip_by_task_split[(entry.get("task"), entry.get("split"), reason)] += 1
                    finally:
                        pbar.update(1)

    print(f"\nProcessing complete: {len(results)} successful, {skipped} skipped")
    if skip_reasons:
        print("Skip breakdown:")
        for reason, n in skip_reasons.most_common():
            print(f"  - {reason}: {n}")
        top_task_split = skip_by_task_split.most_common(6)
        if top_task_split:
            print("Top skipped (task, split, reason):")
            for (task, split, reason), n in top_task_split:
                print(f"  - {task} / {split} / {reason}: {n}")
    if heatmap_paths:
        for task_key, hp in sorted(heatmap_paths.items()):
            print(f"Attention visualizations ({task_key}): {hp}")
    return results


def main():
    """CLI entry point."""
    from vlm_icl.args import add_model_args, add_data_args, add_icl_args, add_run_config_args

    parser = argparse.ArgumentParser(
        description="Run VLM inference with a model and dataset"
    )
    add_model_args(parser)
    add_data_args(parser)
    add_icl_args(parser)
    add_run_config_args(parser)
    parser.set_defaults(max_model_len=65536)

    # Script-specific args
    parser.add_argument(
        "--allowed_local_media_path",
        type=str,
        default=None,
        help=(
            "vLLM only: allow loading file:// images under this directory. "
            "If unset, the vLLM wrapper falls back to embedding local images as data: URLs."
        ),
    )
    parser.add_argument(
        "--image_attentionmap",
        action="store_true",
        help="Save attention heatmap overlays (Qwen3-VL HF backend only; saves per-sample in batch mode)"
    )

    args = parser.parse_args()

    # Prepare data kwargs
    data_kwargs = {}
    if args.topics:
        data_kwargs['topics'] = args.topics
    if args.tasks:
        data_kwargs['tasks'] = args.tasks
    if args.sample_per_split:
        data_kwargs['sample_per_split'] = args.sample_per_split

    # Run inference
    inference(
        model_path=args.model_path,
        backend=args.backend,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_images=args.max_num_images,
        max_model_len=args.max_model_len,
        allowed_local_media_path=args.allowed_local_media_path,
        seed=args.seed,
        output_dir=args.output_dir,
        icl=args.icl,
        num_icl_examples=args.num_icl_examples,
        num_icl_sample_sets=args.num_icl_sample_sets,
        image_attentionmap=args.image_attentionmap,
        k=args.k,
        special=args.special,
        cot=args.cot,
        query_gt_demo=args.query_gt_demo,
        dataset=args.dataset,
        enable_thinking=args.enable_thinking,
        data_root=args.data_root,
        data_source=args.data_source,
        **data_kwargs
    )

    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()
