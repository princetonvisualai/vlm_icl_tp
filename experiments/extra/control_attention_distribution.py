"""Control attention distribution — visualize section-level attention across k values for clean vs corrupted demos."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

from vlm_icl.data import prepare_data
from vlm_icl.images import resolve_image
from vlm_icl.prompts import format_icl_answer
from vlm_icl.icl import (
    build_control_demo_set,
    build_icl_context_pools,
    process_icl_context_images,
)
from vlm_icl.paths import build_analysis_path
from vlm_icl.attention import detect_demo_spans
from vlm_icl.models import ATTENTIONS, load_model, require_capability
from vlm_icl.models.hf import get_decoder_layers
from experiments.extra.visualize_attention_distribution import (
    KIND_COLORS,
    Section,
    segment_token_sequence,
    compute_section_attention,
    compute_answer_to_section_attention,
)

PartitionKey = Tuple[str, str]


@dataclass(frozen=True)
class ControlAttnDistConfig:
    model_path: str
    seed: int = 42
    output_dir: str = "./output_vis"
    tasks: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    max_demos: int = 32
    k_values: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    enable_thinking: bool = False
    cot: bool = False
    layers: Optional[List[int]] = None
    data_root: Optional[str] = None
    data_source: str = "auto"


# ---------------------------------------------------------------------------
# Demo-aware section labeling
# ---------------------------------------------------------------------------


def build_demo_aware_sections(
    sections: List[Section],
    span_info,
) -> List[Section]:
    """Relabel generic sections (img_0, txt_0, ...) using demo span boundaries.

    Labels become: sys_img_0 / sys_txt_0 for system prefix,
    d{i}_img_0 / d{i}_txt_0 for demo i, q_img_0 / q_txt_0 for query.
    """
    first_demo_start = span_info.demo_spans[0][0] if span_info.demo_spans else span_info.query_start
    query_start = span_info.query_start

    # Build lookup: for a given token position, which demo does it belong to?
    def _region(pos: int) -> str:
        if pos < first_demo_start:
            return "sys"
        if pos >= query_start:
            return "q"
        for i, (ds, de) in enumerate(span_info.demo_spans):
            if ds <= pos < de:
                return f"d{i}"
        return "gap"

    result: List[Section] = []
    counters: Dict[str, Dict[str, int]] = {}  # region -> {kind: count}

    for s in sections:
        region = _region(s.start)
        kind_short = "img" if s.kind == "image" else "txt"
        counters.setdefault(region, {"img": 0, "txt": 0})
        idx = counters[region][kind_short]
        counters[region][kind_short] += 1
        label = f"{region}_{kind_short}_{idx}"
        result.append(Section(label, s.start, s.end, s.kind))

    return result


# ---------------------------------------------------------------------------
# Post-processing: remove system sections
# ---------------------------------------------------------------------------


def filter_sys_sections(sections, sec_attn, ans_attn):
    """Remove system-prompt sections from attention matrices and vectors."""
    keep = np.array([not s.label.startswith("sys_") for s in sections])
    idx = np.where(keep)[0]
    return (
        [sections[i] for i in idx],
        sec_attn[np.ix_(idx, idx)],
        ans_attn[idx],
    )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

# Shared epsilon floor for LogNorm (avoids log(0))
_LOG_EPS = 1e-10


def _draw_section_heatmap(ax, matrix, sections, title):
    """Draw an N×N section heatmap on the given axes with log color scale."""
    n = len(sections)
    vmin = max(matrix[matrix > 0].min(), _LOG_EPS) if (matrix > 0).any() else _LOG_EPS
    vmax = max(matrix.max(), vmin * 10)
    im = ax.imshow(matrix.clip(min=_LOG_EPS), cmap="viridis", aspect="auto",
                   norm=LogNorm(vmin=vmin, vmax=vmax))
    labels = [s.label for s in sections]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=5)
    ax.set_yticklabels(labels, fontsize=5)
    for idx, s in enumerate(sections):
        color = KIND_COLORS[s.kind]
        ax.get_xticklabels()[idx].set_color(color)
        ax.get_yticklabels()[idx].set_color(color)
    thresh = np.exp((np.log(vmin) + np.log(vmax)) * 0.7)
    for si in range(n):
        for sj in range(n):
            val = matrix[si, sj]
            ax.text(sj, si, f"{val:.3f}", ha="center", va="center", fontsize=4,
                    color="white" if val < thresh else "black")
    ax.set_title(title, fontsize=7)
    return im


def _draw_answer_heatmap(ax, vector, sections, title):
    """Draw a 1×N answer-to-section heatmap on the given axes with log color scale."""
    n = len(sections)
    data = vector.reshape(1, -1)
    vmin = max(vector[vector > 0].min(), _LOG_EPS) if (vector > 0).any() else _LOG_EPS
    vmax = max(vector.max(), vmin * 10)
    im = ax.imshow(data.clip(min=_LOG_EPS), cmap="viridis", aspect="auto",
                   norm=LogNorm(vmin=vmin, vmax=vmax))
    labels = [s.label for s in sections]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=5)
    ax.set_yticks([0])
    ax.set_yticklabels(["ans"], fontsize=5)
    for idx, s in enumerate(sections):
        ax.get_xticklabels()[idx].set_color(KIND_COLORS[s.kind])
    thresh = np.exp((np.log(vmin) + np.log(vmax)) * 0.7)
    for sj in range(n):
        val = vector[sj]
        ax.text(sj, 0, f"{val:.3f}", ha="center", va="center", fontsize=4,
                color="white" if val < thresh else "black")
    ax.set_title(title, fontsize=7)
    return im


def plot_paired_section_grid(all_data, layer_name, k_values, save_path):
    """Plot len(k_values)×2 grid of section heatmaps (clean vs corrupt)."""
    nrows = len(k_values)
    fig, axes = plt.subplots(nrows, 2, figsize=(16, 4 * nrows), squeeze=False)
    for row, k in enumerate(k_values):
        for col, cond in enumerate(("clean", "corrupt")):
            key = (k, cond, layer_name)
            if key not in all_data:
                axes[row, col].set_visible(False)
                continue
            entry = all_data[key]
            _draw_section_heatmap(axes[row, col], entry["section_attn"], entry["sections"],
                                  f"k={k} {cond}")
    fig.suptitle(f"Section Attention — {layer_name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_paired_answer_grid(all_data, layer_name, k_values, save_path):
    """Plot len(k_values)×2 grid of answer-to-section heatmaps."""
    nrows = len(k_values)
    fig, axes = plt.subplots(nrows, 2, figsize=(16, 2 * nrows), squeeze=False)
    for row, k in enumerate(k_values):
        for col, cond in enumerate(("clean", "corrupt")):
            key = (k, cond, layer_name)
            if key not in all_data:
                axes[row, col].set_visible(False)
                continue
            entry = all_data[key]
            _draw_answer_heatmap(axes[row, col], entry["answer_attn"], entry["sections"],
                                 f"k={k} {cond}")
    fig.suptitle(f"Answer → Section Attention — {layer_name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_tokenwise_heatmap(ax, token_attn, sections, answer_start, title):
    """Draw a token × token attention heatmap with section boundary overlays.
    LogNorm vmin/vmax are derived from the lower triangle only; upper-triangle
    values below vmin are clamped to the same color as vmin."""
    lower = np.tril(token_attn)
    lower_pos = lower[lower > 0]
    vmin = max(lower_pos.min(), _LOG_EPS) if lower_pos.size > 0 else _LOG_EPS
    vmax = max(lower_pos.max(), vmin * 10) if lower_pos.size > 0 else vmin * 10
    im = ax.imshow(token_attn.clip(min=vmin), cmap="viridis", aspect="auto",
                   interpolation="nearest", norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for s in sections:
        ax.axvline(s.start, color="white", lw=0.3, ls="--", alpha=0.5)
        ax.axhline(s.start, color="white", lw=0.3, ls="--", alpha=0.5)
    ax.axvline(answer_start, color="red", lw=0.5, alpha=0.7)
    ax.axhline(answer_start, color="red", lw=0.5, alpha=0.7)
    ax.set_xlabel("Key position", fontsize=6)
    ax.set_ylabel("Query position", fontsize=6)
    ax.set_title(title, fontsize=7)


def plot_paired_tokenwise_grid(all_data, layer_name, k_values, save_path):
    """Plot len(k_values)×2 grid of token-level attention heatmaps (clean vs corrupt)."""
    nrows = len(k_values)
    fig, axes = plt.subplots(nrows, 2, figsize=(16, 6 * nrows), squeeze=False)
    for row, k in enumerate(k_values):
        for col, cond in enumerate(("clean", "corrupt")):
            key = (k, cond, f"__tokenwise_{layer_name}__")
            if key not in all_data:
                axes[row, col].set_visible(False)
                continue
            entry = all_data[key]
            _draw_tokenwise_heatmap(
                axes[row, col], entry["token_attn"], entry["sections"],
                entry["answer_start"], title=f"k={k} {cond}",
            )
    fig.suptitle(f"Token-Level Attention (head-averaged, {layer_name})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_control_attention_distribution(config: ControlAttnDistConfig):
    rng = random.Random(config.seed)

    # 1. Load data
    data_kwargs: Dict[str, Any] = {}
    if config.tasks:
        data_kwargs["tasks"] = config.tasks
    if config.topics:
        data_kwargs["topics"] = config.topics
    full_data = prepare_data(**data_kwargs, seed=config.seed, data_root=config.data_root, data_source=config.data_source)
    print(f"Loaded {len(full_data)} entries")

    # 2. Pick one random query per task from the "simple" split
    by_pk: Dict[PartitionKey, List[Dict]] = {}
    for entry in full_data:
        if entry["split"] != "simple":
            continue
        pk = (entry["task"], entry["split"])
        by_pk.setdefault(pk, []).append(entry)

    queries: Dict[PartitionKey, Dict] = {}
    for pk in sorted(by_pk):
        queries[pk] = rng.choice(by_pk[pk])
    print(f"Selected {len(queries)} queries (one per task, simple split only)")

    # 3. Build context pools excluding query entries
    query_ids = {q["image_id"] for q in queries.values()}
    pool_data = [e for e in full_data if e["image_id"] not in query_ids]
    context_pools = build_icl_context_pools(pool_data)

    # 4. Control demos + corrupted answers
    control_demos, corrupted_answers = build_control_demo_set(
        context_pools, max_demos=config.max_demos, seed=config.seed,
    )

    # 5. Load model (HF backend, eager attention)
    model = load_model(config.model_path, enable_thinking=config.enable_thinking,
                       attn_implementation="eager")
    require_capability(model, ATTENTIONS)
    num_layers = len(get_decoder_layers(model.model))
    print(f"Model loaded: {type(model).__name__} ({num_layers} layers)")

    # Pre-compute layer map (shared across all queries)
    layer_map = {
        "early": num_layers // 4,
        "middle": num_layers // 2,
        "last": num_layers - 1,
    }
    if config.layers:
        for li in config.layers:
            idx = li if li >= 0 else num_layers + li
            layer_map[f"layer{idx}"] = idx
    layer_indices = set(layer_map.values())

    # 6. Process each query
    for pk, query_entry in queries.items():
        task, split = pk
        question = query_entry["questions"][0]
        image = resolve_image(query_entry["image"], prefer_pil=True)
        image2 = None
        if query_entry.get("image2") is not None:
            image2 = resolve_image(query_entry["image2"], prefer_pil=True)

        print(f"\n--- {task}/{split} (image_id={query_entry['image_id']}) ---")

        if pk not in control_demos:
            print(f"  No control demos for {pk}, skipping")
            continue

        demos = control_demos[pk]
        corrupt_ans = corrupted_answers[pk]
        all_data: Dict[tuple, Dict] = {}
        layer_names: List[str] = list(layer_map.keys())

        for k in config.k_values:
            if k > len(demos):
                print(f"  k={k} > available demos ({len(demos)}), skipping")
                continue

            subset = demos[:k]

            for cond in ("clean", "corrupt"):
                if cond == "clean":
                    answers = [d["question"]["ground_truth"] for d in subset]
                else:
                    answers = corrupt_ans[:k]

                context = process_icl_context_images(
                    subset, answers, None, format_icl_answer, prefer_pil=True,
                )

                try:
                    answer, attention_data = model.generate_with_attentions(
                        image=image, prompt=question["prompt"],
                        context=context, cot=config.cot, image2=image2,
                        layer_indices=layer_indices,
                    )
                except torch.cuda.OutOfMemoryError:
                    print(f"  OOM at k={k} {cond}, skipping")
                    torch.cuda.empty_cache()
                    continue

                attentions = attention_data["attentions"]
                full_input_ids = attention_data["full_input_ids"]
                tokenizer = attention_data["tokenizer"]
                answer_start = attention_data["input_ids"].shape[-1]

                # Segment + relabel
                sections = segment_token_sequence(full_input_ids, tokenizer)
                span_info = detect_demo_spans(model, context, image, question["prompt"], cot=config.cot, image2=image2)
                labeled = build_demo_aware_sections(sections, span_info)

                # Precompute sys-token indices (shared across layers)
                sys_tokens = set()
                for s in labeled:
                    if s.label.startswith("sys_"):
                        sys_tokens.update(range(s.start, s.end))
                non_sys = [s for s in labeled if not s.label.startswith("sys_")]
                sys_before_answer = sum(1 for t in sys_tokens if t < answer_start)

                # Compute matrices for each layer (filter sys sections)
                for lname, lidx in layer_map.items():
                    sec_attn = compute_section_attention(attentions, labeled, lidx)
                    ans_attn = compute_answer_to_section_attention(attentions, labeled, answer_start, lidx)
                    filt_sec, filt_sa, filt_aa = filter_sys_sections(labeled, sec_attn, ans_attn)
                    all_data[(k, cond, lname)] = {
                        "section_attn": filt_sa,
                        "answer_attn": filt_aa,
                        "sections": filt_sec,
                    }

                    # Tokenwise attention (sys tokens removed)
                    layer_attn = attentions[lidx][0].float().mean(0).cpu().numpy()
                    keep_idx = np.array([i for i in range(layer_attn.shape[0]) if i not in sys_tokens])
                    layer_attn = layer_attn[np.ix_(keep_idx, keep_idx)]
                    all_data[(k, cond, f"__tokenwise_{lname}__")] = {
                        "token_attn": layer_attn,
                        "sections": non_sys,
                        "answer_start": answer_start - sys_before_answer,
                    }

                # Free memory
                del attentions, attention_data
                torch.cuda.empty_cache()

                print(f"  k={k} {cond}: answer={answer!r}, sections={len(labeled)}")

        # 7. Plot grids
        model_name = Path(config.model_path).name
        out_dir = Path(config.output_dir) / "control_attention_dist" / task / model_name / split
        out_dir.mkdir(parents=True, exist_ok=True)

        active_k = [k for k in config.k_values if any((k, c, ln) in all_data for c in ("clean", "corrupt") for ln in layer_names)]

        for lname in layer_names:
            plot_paired_section_grid(all_data, lname, active_k,
                                    out_dir / f"section_heatmap_{lname}.png")
            plot_paired_answer_grid(all_data, lname, active_k,
                                   out_dir / f"answer_section_{lname}.png")
            plot_paired_tokenwise_grid(all_data, lname, active_k,
                                       out_dir / f"tokenwise_heatmap_{lname}.png")

        # 8. Save metadata
        metadata = {
            "model_path": config.model_path,
            "seed": config.seed,
            "query": {
                "image_id": query_entry["image_id"],
                "task": task, "split": split,
                "prompt": question["prompt"],
                "ground_truth": question["ground_truth"],
            },
            "k_values": list(active_k),
            "control_demos": [
                {"image_id": d["entry"]["image_id"], "gt": d["question"]["ground_truth"]}
                for d in demos
            ],
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> ControlAttnDistConfig:
    from vlm_icl.args import add_model_args, add_data_args, add_run_config_args

    parser = argparse.ArgumentParser(
        description="Control attention distribution — clean vs corrupt across k values."
    )
    add_model_args(parser)
    add_data_args(parser)
    add_run_config_args(parser)

    parser.add_argument("--max_demos", type=int, default=32)
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Additional layer indices to visualize")

    args = parser.parse_args(argv)
    return ControlAttnDistConfig(
        model_path=args.model_path,
        seed=args.seed,
        output_dir=args.output_dir,
        tasks=args.tasks,
        topics=args.topics,
        max_demos=args.max_demos,
        k_values=tuple(args.k_values),
        enable_thinking=args.enable_thinking,
        cot=args.cot,
        layers=args.layers,
        data_root=args.data_root,
        data_source=args.data_source,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    run_control_attention_distribution(config)


if __name__ == "__main__":
    main()
