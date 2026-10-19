"""Visualize attention distribution across image/text sections of the input prompt. HuggingFace inference only."""

import argparse
import json
import random
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from vlm_icl.lens import resolve_image_token_id
from vlm_icl.data import TASK_PARTITIONS, prepare_data
from vlm_icl.images import resolve_image
from vlm_icl.prompts import format_icl_answer
from vlm_icl.icl import (
    apply_special_mode,
    build_demo_pool,
    build_icl_context_pools,
    process_icl_context_images,
    select_from_demo_pool,
    shuffle_icl_answers,
    validate_special_mode,
)
from vlm_icl.paths import build_output_path, derive_run_config
from vlm_icl.models import ATTENTIONS, load_model, require_capability
from vlm_icl.models.hf import get_decoder_layers


# ---------------------------------------------------------------------------
# Step 4: Segment the token sequence into sections
# ---------------------------------------------------------------------------

Section = namedtuple("Section", ["label", "start", "end", "kind"])


def segment_token_sequence(
    full_input_ids: torch.Tensor,
    tokenizer,
) -> List[Section]:
    """Segment token IDs into consecutive image/text sections."""
    ids = full_input_ids[0] if full_input_ids.dim() == 2 else full_input_ids
    image_token_id = resolve_image_token_id(tokenizer, ids)

    sections: List[Section] = []
    img_counter = 0
    txt_counter = 0

    i = 0
    seq_len = len(ids)
    while i < seq_len:
        is_image = image_token_id is not None and int(ids[i]) == image_token_id
        start = i
        if is_image:
            while i < seq_len and int(ids[i]) == image_token_id:
                i += 1
            sections.append(Section(f"img_{img_counter}", start, i, "image"))
            img_counter += 1
        else:
            while i < seq_len and (image_token_id is None or int(ids[i]) != image_token_id):
                i += 1
            sections.append(Section(f"txt_{txt_counter}", start, i, "text"))
            txt_counter += 1

    return sections


# ---------------------------------------------------------------------------
# Step 5: Compute section-level attention matrix
# ---------------------------------------------------------------------------


def compute_section_attention(
    attentions: tuple,
    sections: List[Section],
    layer_index: int,
) -> np.ndarray:
    """Compute N x N section-level attention matrix for a single layer, averaged over all heads."""
    layer_attn = attentions[layer_index][0]  # (num_heads, seq_len, seq_len)
    attn = layer_attn.float().mean(dim=0)    # (seq_len, seq_len)

    n = len(sections)
    matrix = np.zeros((n, n), dtype=np.float64)

    for si, src in enumerate(sections):
        src_idx = torch.arange(src.start, src.end, device=attn.device)
        for sj, dst in enumerate(sections):
            dst_idx = torch.arange(dst.start, dst.end, device=attn.device)
            sub = attn.index_select(0, src_idx).index_select(1, dst_idx)
            matrix[si, sj] = sub.mean().item()

    return matrix


def compute_section_attention_aggregate(
    attentions: tuple,
    sections: List[Section],
) -> np.ndarray:
    """Compute section-level attention averaged across all layers."""
    num_layers = len(attentions)
    matrices = [
        compute_section_attention(attentions, sections, li)
        for li in range(num_layers)
    ]
    return np.mean(matrices, axis=0)


# ---------------------------------------------------------------------------
# Step 6: Compute token-sink summary
# ---------------------------------------------------------------------------


def compute_token_sink(
    attentions: tuple,
    layer_index: int,
) -> np.ndarray:
    """Compute per-token received attention (column sums) for one layer."""
    layer_attn = attentions[layer_index][0]  # (num_heads, seq_len, seq_len)
    attn = layer_attn.float().mean(dim=0)    # (seq_len, seq_len)
    return attn.sum(dim=0).cpu().numpy()


def compute_token_sink_aggregate(attentions: tuple) -> np.ndarray:
    """Compute per-token received attention averaged across all layers."""
    num_layers = len(attentions)
    sinks = [compute_token_sink(attentions, li) for li in range(num_layers)]
    return np.mean(sinks, axis=0)


# ---------------------------------------------------------------------------
# Step 6b: Compute answer-to-section attention
# ---------------------------------------------------------------------------


def compute_answer_to_section_attention(
    attentions: tuple,
    sections: List[Section],
    answer_start: int,
    layer_index: int,
) -> np.ndarray:
    """Compute attention from answer tokens to each section for one layer."""
    layer_attn = attentions[layer_index][0]  # (num_heads, seq_len, seq_len)
    attn = layer_attn.float().mean(dim=0)    # (seq_len, seq_len)

    answer_attn = attn[answer_start:, :]          # (num_answer_tokens, seq_len)
    answer_attn_mean = answer_attn.mean(dim=0)    # (seq_len,)

    n = len(sections)
    result = np.zeros(n, dtype=np.float64)
    for si, s in enumerate(sections):
        result[si] = answer_attn_mean[s.start:s.end].sum().item()

    return result


def compute_answer_to_section_attention_aggregate(
    attentions: tuple,
    sections: List[Section],
    answer_start: int,
) -> np.ndarray:
    """Compute answer-to-section attention averaged across all layers."""
    num_layers = len(attentions)
    vectors = [
        compute_answer_to_section_attention(attentions, sections, answer_start, li)
        for li in range(num_layers)
    ]
    return np.mean(vectors, axis=0)


# ---------------------------------------------------------------------------
# Step 7: Plot section-level heatmap
# ---------------------------------------------------------------------------

KIND_COLORS = {"image": "#1f77b4", "text": "#ff7f0e"}


def plot_section_heatmap(
    section_attn: np.ndarray,
    sections: List[Section],
    title: str,
    save_path: Union[str, Path],
) -> None:
    """Save an annotated section-level attention heatmap."""
    n = len(sections)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.8), max(5, n * 0.7)))

    im = ax.imshow(section_attn, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = [s.label for s in sections]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # Color tick labels by kind
    for idx, s in enumerate(sections):
        color = KIND_COLORS[s.kind]
        ax.get_xticklabels()[idx].set_color(color)
        ax.get_yticklabels()[idx].set_color(color)

    # Annotate cells
    for si in range(n):
        for sj in range(n):
            val = section_attn[si, sj]
            ax.text(
                sj, si, f"{val:.3f}",
                ha="center", va="center", fontsize=6,
                color="white" if val < section_attn.max() * 0.7 else "black",
            )

    ax.set_xlabel("Destination section", fontsize=10)
    ax.set_ylabel("Source section", fontsize=10)
    ax.set_title(title, fontsize=12)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 8: Plot token-sink
# ---------------------------------------------------------------------------


def plot_token_sink(
    column_sums: np.ndarray,
    sections: List[Section],
    title: str,
    save_path: Union[str, Path],
) -> None:
    """Save a 1-D token-sink plot with section-kind shading."""
    seq_len = len(column_sums)
    fig, ax = plt.subplots(figsize=(max(10, seq_len / 200), 4))

    # Shade background by section kind
    for s in sections:
        color = "#d0e8ff" if s.kind == "image" else "#ffe8d0"
        ax.axvspan(s.start, s.end, alpha=0.4, color=color, linewidth=0)

    # Draw section boundaries
    for s in sections:
        ax.axvline(s.start, color="gray", linewidth=0.3, linestyle="--")

    ax.plot(range(seq_len), column_sums, linewidth=0.5, color="black")

    ax.set_xlabel("Token position", fontsize=10)
    ax.set_ylabel("Total attention received", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, seq_len)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 8b: Plot answer-to-section heatmap
# ---------------------------------------------------------------------------


def plot_answer_section_heatmap(
    answer_section_attn: np.ndarray,
    sections: List[Section],
    title: str,
    save_path: Union[str, Path],
) -> None:
    """Save a 1×N heatmap showing answer-token attention to each section."""
    n = len(sections)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.8), 2.5))

    data = answer_section_attn.reshape(1, -1)
    im = ax.imshow(data, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = [s.label for s in sections]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([0])
    ax.set_yticklabels(["answer"], fontsize=8)

    # Color tick labels by kind
    for idx, s in enumerate(sections):
        color = KIND_COLORS[s.kind]
        ax.get_xticklabels()[idx].set_color(color)

    # Annotate cells
    for sj in range(n):
        val = answer_section_attn[sj]
        ax.text(
            sj, 0, f"{val:.3f}",
            ha="center", va="center", fontsize=6,
            color="white" if val < answer_section_attn.max() * 0.7 else "black",
        )

    ax.set_xlabel("Destination section", fontsize=10)
    ax.set_title(title, fontsize=12)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 9 + 10: Main loop
# ---------------------------------------------------------------------------


def run_attention_distribution(
    model_path: str,
    seed: int = 42,
    output_dir: str = "./output",
    single: bool = False,
    icl: bool = False,
    num_icl_examples: int = 2,
    num_icl_sample_sets: int = 1,
    query_gt_demo: Optional[str] = None,
    special: Optional[str] = None,
    cot: bool = False,
    layers: Optional[List[int]] = None,
    enable_thinking: bool = False,
    data_root: Optional[str] = None,
    data_source: str = "auto",
    **data_kwargs,
) -> None:
    """Run attention distribution analysis."""
    random.seed(seed)
    validate_special_mode(special, icl, num_icl_examples)

    # --- Step 2: Load model and data ---
    print(f"Loading model from {model_path}...")
    model_kwargs = {"attn_implementation": "eager"}
    model = load_model(model_path, enable_thinking=enable_thinking, **model_kwargs)

    require_capability(model, ATTENTIONS)
    num_layers = len(get_decoder_layers(model.model))
    print(f"Model loaded successfully! (Type: {type(model).__name__}, {num_layers} layers)")

    print("\nPreparing dataset: visual_reasoning...")
    data = prepare_data(**data_kwargs, data_root=data_root, data_source=data_source)
    print(f"Dataset prepared with {len(data)} entries")
    if not data:
        raise ValueError("No data entries found!")

    # ICL context pools (use full dataset when sample_per_split is used)
    icl_pool_data = None
    if icl and data_kwargs.get("sample_per_split") is not None:
        full_kwargs = {k: v for k, v in data_kwargs.items() if k != "sample_per_split"}
        icl_pool_data = prepare_data(**full_kwargs, data_root=data_root, data_source=data_source)
        print(f"ICL context pool: {len(icl_pool_data)} entries (full dataset)")

    context_pools = None
    demo_pool = None
    if icl:
        context_pools = build_icl_context_pools(icl_pool_data if icl_pool_data is not None else data)
        demo_pool = build_demo_pool(context_pools, seed=seed)

    answer_rng = random.Random(seed + 1)
    run_config = derive_run_config(special, cot)
    prefer_pil = True  # HF backend

    # Build output paths per task (lazy)
    output_paths: Dict[str, Path] = {}

    total_pairs = sum(len(entry["questions"]) for entry in data)
    print(f"\nRunning attention distribution analysis on {total_pairs} samples...")

    with tqdm(total=total_pairs, desc="Attention distribution", unit="sample") as pbar:
        for entry in data:
            image_obj = apply_special_mode(entry["image"], special, is_icd=False)
            image = resolve_image(image_obj, prefer_pil=prefer_pil) if image_obj is not None else None
            image2 = None
            image2_obj = entry.get("image2")
            if image2_obj is not None:
                image2_obj = apply_special_mode(image2_obj, special, is_icd=False)
                image2 = resolve_image(image2_obj, prefer_pil=prefer_pil) if image2_obj is not None else None

            for q_idx, question in enumerate(entry["questions"], 1):
                try:
                    _process_sample(
                        model=model,
                        entry=entry,
                        question=question,
                        q_idx=q_idx,
                        image=image,
                        image2=image2,
                        icl=icl,
                        num_icl_examples=num_icl_examples,
                        query_gt_demo=query_gt_demo,
                        demo_pool=demo_pool,
                        special=special,
                        cot=cot,
                        answer_rng=answer_rng,
                        prefer_pil=prefer_pil,
                        layers=layers,
                        num_layers=num_layers,
                        model_path=model_path,
                        seed=seed,
                        output_dir=output_dir,
                        run_config=run_config,
                        output_paths=output_paths,
                        enable_thinking=enable_thinking,
                    )
                except torch.cuda.OutOfMemoryError:
                    tqdm.write("  CUDA OOM error, skipping")
                    torch.cuda.empty_cache()
                except Exception as e:
                    tqdm.write(
                        f"  Error task={entry.get('task')} split={entry.get('split')} "
                        f"image_id={entry.get('image_id')} q={q_idx}: {e}"
                    )
                finally:
                    pbar.update(1)

                if single:
                    print("\n--single mode: stopping after first sample.")
                    return

    print("\nAttention distribution analysis complete.")


def _process_sample(
    *,
    model,
    entry: Dict[str, Any],
    question: Dict[str, Any],
    q_idx: int,
    image,
    image2,
    icl: bool,
    num_icl_examples: int,
    query_gt_demo: Optional[str] = None,
    demo_pool: Optional[Dict],
    special: Optional[str],
    cot: bool,
    answer_rng: random.Random,
    prefer_pil: bool,
    layers: Optional[List[int]],
    num_layers: int,
    model_path: str,
    seed: int,
    output_dir: str,
    run_config: str,
    output_paths: Dict[str, Path],
    enable_thinking: bool = False,
) -> None:
    """Process one sample: inference, segmentation, plotting, saving."""
    # --- Determine layer indices (before inference to enable selective capture) ---
    layer_map = {
        "early": num_layers // 4,
        "middle": num_layers // 2,
        "last": num_layers - 1,
    }
    if layers:
        for li in layers:
            idx = li if li >= 0 else num_layers + li
            layer_map[f"layer{idx}"] = idx

    # --- Step 3: Build ICL context and run inference ---
    context = None
    context_samples = None
    shuffled_answers = None
    if icl and demo_pool is not None:
        context_samples = select_from_demo_pool(
            entry, question, demo_pool, num_icl_examples, seed,
            query_gt_position=query_gt_demo,
            ensure_query_gt=(query_gt_demo is not None),
        )
        if context_samples is None:
            tqdm.write(f"  Insufficient ICL context for {entry['image_id']}, skipping")
            return
        shuffled_answers = shuffle_icl_answers(context_samples, special, answer_rng)
        context = process_icl_context_images(
            context_samples, shuffled_answers, special, format_icl_answer, prefer_pil=prefer_pil,
        )

    answer, attention_data = model.generate_with_attentions(
        image=image,
        prompt=question["prompt"],
        context=context,
        cot=cot,
        image2=image2,
        layer_indices=set(layer_map.values()),
    )

    attentions = attention_data["attentions"]
    full_input_ids = attention_data["full_input_ids"]
    tokenizer = attention_data["tokenizer"]
    answer_start = attention_data["input_ids"].shape[-1]

    # --- Step 4: Segment ---
    sections = segment_token_sequence(full_input_ids, tokenizer)

    # --- Build output directory ---
    task_key = entry.get("task", "unknown")
    if task_key not in output_paths:
        output_paths[task_key] = build_output_path(
            base_dir=output_dir,
            analysis_type="attention_dist",
            model_path=model_path,
            task_name=task_key,
            icl=icl,
            num_icl_examples=num_icl_examples if icl else 0,
            run_config=run_config,
            query_gt_demo=query_gt_demo,
            enable_thinking=enable_thinking,
        )

    sample_dir = (
        output_paths[task_key]
        / _safe_path(entry.get("split"))
        / _safe_path(entry.get("image_id"))
        / f"q{q_idx}"
    )
    sample_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 5 + 7: Section heatmaps per layer + aggregate ---
    for name, layer_idx in layer_map.items():
        section_attn = compute_section_attention(attentions, sections, layer_idx)
        plot_section_heatmap(
            section_attn, sections,
            title=f"Section Attention — Layer {layer_idx} ({name})",
            save_path=sample_dir / f"section_heatmap_{name}.png",
        )

    agg_section_attn = np.mean(
        [compute_section_attention(attentions, sections, li) for li in layer_map.values()], axis=0,
    )
    plot_section_heatmap(
        agg_section_attn, sections,
        title="Section Attention — Aggregate (sampled layers)",
        save_path=sample_dir / "section_heatmap_aggregate.png",
    )

    # --- Step 6 + 8: Token-sink per layer + aggregate ---
    for name, layer_idx in layer_map.items():
        col_sums = compute_token_sink(attentions, layer_idx)
        plot_token_sink(
            col_sums, sections,
            title=f"Token Sink — Layer {layer_idx} ({name})",
            save_path=sample_dir / f"token_sink_{name}.png",
        )

    agg_col_sums = np.mean(
        [compute_token_sink(attentions, li) for li in layer_map.values()], axis=0,
    )
    plot_token_sink(
        agg_col_sums, sections,
        title="Token Sink — Aggregate (sampled layers)",
        save_path=sample_dir / "token_sink_aggregate.png",
    )

    # --- Step 6b + 8b: Answer-to-section heatmaps per layer + aggregate ---
    for name, layer_idx in layer_map.items():
        ans_attn = compute_answer_to_section_attention(
            attentions, sections, answer_start, layer_idx,
        )
        plot_answer_section_heatmap(
            ans_attn, sections,
            title=f"Answer \u2192 Section Attention \u2014 Layer {layer_idx} ({name})",
            save_path=sample_dir / f"answer_section_heatmap_{name}.png",
        )

    agg_ans_attn = np.mean(
        [compute_answer_to_section_attention(attentions, sections, answer_start, li)
         for li in layer_map.values()], axis=0,
    )
    plot_answer_section_heatmap(
        agg_ans_attn, sections,
        title="Answer \u2192 Section Attention \u2014 Aggregate (sampled layers)",
        save_path=sample_dir / "answer_section_heatmap_aggregate.png",
    )

    # --- Step 9: Save metadata ---
    sections_data = [
        {"label": s.label, "start": s.start, "end": s.end, "kind": s.kind}
        for s in sections
    ]
    with open(sample_dir / "sections.json", "w") as f:
        json.dump(sections_data, f, indent=2)

    metadata = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model_path": model_path,
        "seed": seed,
        "image_id": entry.get("image_id"),
        "task": entry.get("task"),
        "topic": entry.get("topic"),
        "split": entry.get("split"),
        "question_type": question.get("type"),
        "prompt": question.get("prompt"),
        "ground_truth": question.get("ground_truth"),
        "expected_bias": question.get("expected_bias"),
        "model_answer": answer,
        "icl": icl,
        "num_icl_examples": num_icl_examples if icl else 0,
        "cot": cot,
        "special": special,
        "num_sections": len(sections),
        "num_layers": num_layers,
        "seq_len": int(full_input_ids.shape[-1]),
    }
    if icl and context_samples and shuffled_answers:
        metadata["context_examples"] = [
            {
                "image_id": ctx["entry"]["image_id"],
                "prompt": ctx["question"]["prompt"],
                "ground_truth": shuffled_answers[i],
            }
            for i, ctx in enumerate(context_samples)
        ]
    with open(sample_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def _safe_path(value: Any) -> str:
    s = str(value) if value is not None else "unknown"
    return s.replace("/", "_").replace("\\", "_").strip() or "unknown"


# ---------------------------------------------------------------------------
# Step 1: CLI argument parser
# ---------------------------------------------------------------------------


def main():
    from vlm_icl.args import add_model_args, add_data_args, add_icl_args, add_run_config_args

    parser = argparse.ArgumentParser(
        description="Visualize attention distribution across image/text sections"
    )
    add_model_args(parser)
    add_data_args(parser)
    add_icl_args(parser)
    add_run_config_args(parser)

    # Script-specific
    parser.add_argument("--single", action="store_true", help="Process one sample and exit")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Additional layer indices to visualize (default: early/middle/last + aggregate)")

    args = parser.parse_args()

    data_kwargs: Dict[str, Any] = {}
    if args.topics:
        data_kwargs["topics"] = args.topics
    if args.tasks:
        data_kwargs["tasks"] = args.tasks
    if args.sample_per_split:
        data_kwargs["sample_per_split"] = args.sample_per_split

    run_attention_distribution(
        model_path=args.model_path,
        seed=args.seed,
        output_dir=args.output_dir,
        single=args.single,
        icl=args.icl,
        num_icl_examples=args.num_icl_examples,
        num_icl_sample_sets=args.num_icl_sample_sets,
        query_gt_demo=args.query_gt_demo,
        special=args.special,
        cot=args.cot,
        layers=args.layers,
        enable_thinking=args.enable_thinking,
        data_root=args.data_root,
        data_source=args.data_source,
        **data_kwargs,
    )


if __name__ == "__main__":
    main()
