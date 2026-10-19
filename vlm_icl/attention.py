"""Attention analysis: ICL demo token spans and attention heatmaps on image patches."""

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch

from .paths import build_output_path
from .lens import flatten_ids, resolve_grid_thw, resolve_image_token_id, split_runs


# ============================================================================
# Demo token spans
# ============================================================================

@dataclass
class SpanInfo:
    """Token-level span information for ICL demos and query."""
    demo_spans: List[Tuple[int, int]]  # per-demo (start, end) in token space
    query_start: int
    seq_len: int


def _divergence_sys_len(ids_0: "torch.Tensor", ids_1: "torch.Tensor") -> int:
    """Fallback: find first index where two token sequences diverge."""
    min_len = min(len(ids_0), len(ids_1))
    for i in range(min_len):
        if ids_0[i] != ids_1[i]:
            return i
    return min_len


def detect_demo_spans(model, context, query_image, query_prompt, **msg_kwargs) -> SpanInfo:
    """Detect token spans for each ICL demo by incrementally tokenizing with 0..N demos and diffing lengths."""
    num_demos = len(context)

    # Tokenize with 0 demos to get base length
    msgs_0 = model.build_messages(query_image, query_prompt, context=[], **msg_kwargs)
    inputs_0 = model.prepare_inputs(msgs_0)
    l_0 = inputs_0.input_ids.shape[1]

    # Find system prefix length: position of first <|vision_start|> token.
    # The old divergence-based approach fails because <|image_pad|> tokens from
    # different images share the same token ID, causing the first demo's image
    # to be misclassified as part of the system prefix.
    ids_0 = inputs_0.input_ids[0].cpu()
    vocab = model.processor.tokenizer.get_vocab()
    vision_start_id = vocab.get("<|vision_start|>")

    if vision_start_id is not None:
        vs_positions = (ids_0 == vision_start_id).nonzero(as_tuple=True)[0]
        if len(vs_positions) > 0:
            sys_len = vs_positions[0].item()
        else:
            # Fallback: no vision tokens (text-only demos)
            msgs_1 = model.build_messages(query_image, query_prompt, context=context[:1], **msg_kwargs)
            inputs_1 = model.prepare_inputs(msgs_1)
            sys_len = _divergence_sys_len(ids_0, inputs_1.input_ids[0].cpu())
    else:
        # Fallback: tokenizer lacks <|vision_start|>
        msgs_1 = model.build_messages(query_image, query_prompt, context=context[:1], **msg_kwargs)
        inputs_1 = model.prepare_inputs(msgs_1)
        sys_len = _divergence_sys_len(ids_0, inputs_1.input_ids[0].cpu())

    # Collect lengths for 1..N demos
    lengths = [l_0]
    for k in range(1, num_demos + 1):
        msgs_k = model.build_messages(query_image, query_prompt, context=context[:k], **msg_kwargs)
        inputs_k = model.prepare_inputs(msgs_k)
        lengths.append(inputs_k.input_ids.shape[1])

    # Compute demo spans
    demo_spans = []
    for k in range(1, num_demos + 1):
        start = sys_len + lengths[k - 1] - l_0
        end = sys_len + lengths[k] - l_0
        demo_spans.append((start, end))

    query_start = sys_len + lengths[num_demos] - l_0
    seq_len = lengths[num_demos]

    return SpanInfo(demo_spans=demo_spans, query_start=query_start, seq_len=seq_len)


# ============================================================================
# Attention heatmaps on image patches
# ============================================================================

def extract_answer_text(generated_text: str) -> str:
    """Return the text inside the first pair of braces, or the full text."""
    match = re.search(r"\{([^{}]+)\}", generated_text)
    if match:
        return match.group(1).strip()
    return generated_text.strip()


def _to_image(image: Union[str, Image.Image]) -> Image.Image:
    if isinstance(image, str):
        return Image.open(image)
    return image


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> List[int]:
    if not needle:
        return []
    matches = []
    limit = len(haystack) - len(needle) + 1
    for start in range(limit):
        if list(haystack[start:start + len(needle)]) == list(needle):
            matches.append(start)
    return matches


def _build_attention_grid(
    attn_vector: torch.Tensor,
    grid_thw: Tuple[int, int, int]
) -> np.ndarray:
    t, h, w = grid_thw
    attn = attn_vector.detach().float().cpu().numpy()
    attn = attn.reshape(t, h, w)
    attn = attn.mean(axis=0) if t > 1 else attn.squeeze(0)
    return attn


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = heatmap - heatmap.min()
    max_val = heatmap.max()
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap


def overlay_heatmap(
    image: Union[str, Image.Image],
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet"
) -> Image.Image:
    image = _to_image(image).convert("RGB")
    heatmap = _normalize_heatmap(heatmap)
    cmap = cm.get_cmap(colormap)
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(colored).resize(image.size, resample=Image.BILINEAR)
    return Image.blend(image, heat_img, alpha)


def _resolve_attention_positions(
    attention_data: Dict[str, Any],
    answer_text: Optional[str],
    image_index: int,
) -> Tuple[List[int], List[int], Tuple[int, int, int]]:
    """Resolve answer token positions, target image token positions, and grid THW."""
    tokenizer = attention_data.get("tokenizer")
    if tokenizer is None:
        raise ValueError("Tokenizer missing from attention data.")

    input_ids = flatten_ids(attention_data["input_ids"])
    generated_ids = flatten_ids(attention_data["generated_ids_trimmed"])
    image_grid_thw = attention_data.get("image_grid_thw")

    answer_text = answer_text or attention_data.get("answer_text", "")
    target_text = extract_answer_text(answer_text)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    if not target_ids:
        target_ids = tokenizer.encode(answer_text.strip(), add_special_tokens=False)

    generated_list = generated_ids.tolist()
    matches = _find_subsequence(generated_list, target_ids) if target_ids else []
    if matches:
        start = matches[-1]
        answer_positions = list(range(start, start + len(target_ids)))
    elif generated_list:
        # Skip trailing special tokens (EOS/PAD) when falling back
        eos_id = getattr(tokenizer, "eos_token_id", None)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        special_ids = {eos_id, pad_id} - {None}
        last_idx = len(generated_list) - 1
        while last_idx > 0 and generated_list[last_idx] in special_ids:
            last_idx -= 1
        answer_positions = [last_idx]
    else:
        raise ValueError("No generated tokens found for attention selection.")

    input_len = input_ids.shape[-1]
    answer_positions = [input_len + pos for pos in answer_positions]

    image_token_id = resolve_image_token_id(tokenizer, input_ids)
    if image_token_id is None:
        raise ValueError("Could not locate image token id in input sequence.")

    image_positions = torch.where(input_ids == image_token_id)[0].tolist()
    runs = split_runs(image_positions)
    if not runs:
        raise ValueError("No image tokens found in input ids.")

    # Select which image's tokens to use (image_index=-1 means last/query image)
    actual_idx = image_index if image_index >= 0 else len(runs) + image_index
    if actual_idx < 0 or actual_idx >= len(runs):
        raise ValueError(f"Invalid image_index {image_index} for {len(runs)} images.")
    target_positions = runs[actual_idx]
    grid_thw = resolve_grid_thw(image_grid_thw, actual_idx, len(target_positions))

    return answer_positions, target_positions, grid_thw


def _extract_attention_vector(
    attentions: tuple,
    layer_index: int,
    answer_positions: List[int],
    target_positions: List[int],
) -> torch.Tensor:
    """Return 1-D attention vector from answer tokens to target image tokens for one layer."""
    layer_attn = attentions[layer_index][0]        # (num_heads, seq_len, seq_len)
    attn = layer_attn.mean(dim=0)                  # average over heads
    answer_index = torch.tensor(answer_positions, device=attn.device)
    target_index = torch.tensor(target_positions, device=attn.device)
    attn_to_image = attn.index_select(0, answer_index).index_select(1, target_index)
    return attn_to_image.mean(dim=0)               # average over answer tokens


def save_answer_attention_overlay(
    attention_data: Dict[str, Any],
    image: Union[str, Image.Image],
    output_dir: Union[str, Path],
    answer_text: Optional[str] = None,
    file_prefix: str = "attention",
    layer_index: int = -1,
    image_index: int = -1
) -> Optional[Path]:
    """Save attention overlay for a specific image in the sequence (image_index=-1 = last/query)."""
    attentions = attention_data.get("attentions")
    if attentions is None:
        raise ValueError("Attention tensor missing from attention data.")
    if len(attentions) == 0:
        raise ValueError("Attention tensor is empty.")

    answer_positions, target_positions, grid_thw = _resolve_attention_positions(
        attention_data, answer_text, image_index
    )
    attn_vector = _extract_attention_vector(
        attentions, layer_index, answer_positions, target_positions
    )

    heatmap = _build_attention_grid(attn_vector, grid_thw)
    overlay = overlay_heatmap(image, heatmap)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{file_prefix}.png"
    overlay.save(output_path)
    return output_path


def save_aggregate_attention_overlay(
    attention_data: Dict[str, Any],
    image: Union[str, Image.Image],
    output_dir: Union[str, Path],
    answer_text: Optional[str] = None,
    file_prefix: str = "attention_aggregate",
    image_index: int = -1,
) -> Optional[Path]:
    """Save attention overlay aggregated across all layers (image_index=-1 = query)."""
    attentions = attention_data.get("attentions")
    if attentions is None:
        raise ValueError("Attention tensor missing from attention data.")
    if len(attentions) == 0:
        raise ValueError("Attention tensor is empty.")

    answer_positions, target_positions, grid_thw = _resolve_attention_positions(
        attention_data, answer_text, image_index
    )

    num_layers = len(attentions)
    vectors = [
        _extract_attention_vector(attentions, layer_idx, answer_positions, target_positions)
        for layer_idx in range(num_layers)
    ]
    agg_vector = torch.stack(vectors, dim=0).mean(dim=0)

    heatmap = _build_attention_grid(agg_vector, grid_thw)
    overlay = overlay_heatmap(image, heatmap)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{file_prefix}.png"
    overlay.save(output_path)
    return output_path


def save_multi_layer_attention_overlays(
    attention_data: Dict[str, Any],
    image: Union[str, Image.Image],
    output_dir: Union[str, Path],
    answer_text: Optional[str] = None,
    file_prefix: str = "attention",
    image_index: int = -1
) -> List[Path]:
    """Save attention overlays at early, middle, and last layers, plus an aggregate (all-layer average)."""
    attentions = attention_data.get("attentions")
    if attentions is None or len(attentions) == 0:
        raise ValueError("Attention tensor missing or empty.")

    num_layers = len(attentions)
    layer_indices = {
        "early": num_layers // 4,
        "middle": num_layers // 2,
        "last": -1
    }

    saved_paths = []
    for name, idx in layer_indices.items():
        try:
            path = save_answer_attention_overlay(
                attention_data=attention_data,
                image=image,
                output_dir=output_dir,
                answer_text=answer_text,
                file_prefix=f"{file_prefix}_{name}",
                layer_index=idx,
                image_index=image_index
            )
            if path:
                saved_paths.append(path)
        except Exception as e:
            print(f"Warning: Failed to save {name} layer attention: {e}")

    try:
        path = save_aggregate_attention_overlay(
            attention_data=attention_data,
            image=image,
            output_dir=output_dir,
            answer_text=answer_text,
            file_prefix=f"{file_prefix}_aggregate",
            image_index=image_index,
        )
        if path:
            saved_paths.append(path)
    except Exception as e:
        print(f"Warning: Failed to save aggregate layer attention: {e}")

    return saved_paths


# ============================================================================
# Query → Context attention mass maps
# ============================================================================


def _resolve_query_and_context_positions(
    attention_data: Dict[str, Any],
    num_query_images: int = 1,
) -> Tuple[List[List[int]], List[List[int]], List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """Identify query-image and context-image token positions and their grid shapes."""
    tokenizer = attention_data.get("tokenizer")
    if tokenizer is None:
        raise ValueError("Tokenizer missing from attention data.")

    input_ids = flatten_ids(attention_data["input_ids"])
    image_grid_thw = attention_data.get("image_grid_thw")

    image_token_id = resolve_image_token_id(tokenizer, input_ids)
    if image_token_id is None:
        raise ValueError("Could not locate image token id in input sequence.")

    image_positions = torch.where(input_ids == image_token_id)[0].tolist()
    runs = split_runs(image_positions)
    if len(runs) < num_query_images + 1:
        raise ValueError(
            f"Need at least {num_query_images + 1} images for query→context mass maps, "
            f"found {len(runs)}."
        )

    # Last num_query_images runs = query images, rest = context images
    query_positions = runs[-num_query_images:]
    context_positions = runs[:-num_query_images]

    query_grid_thws = [
        resolve_grid_thw(image_grid_thw, len(runs) - num_query_images + i, len(qp))
        for i, qp in enumerate(query_positions)
    ]
    context_grid_thws = [
        resolve_grid_thw(image_grid_thw, i, len(ctx))
        for i, ctx in enumerate(context_positions)
    ]

    return query_positions, context_positions, query_grid_thws, context_grid_thws


def _extract_query_to_context_vector(
    attentions: tuple,
    layer_index: int,
    query_positions: List[int],
    context_positions: List[int],
) -> torch.Tensor:
    """Return 1-D attention vector from query-image tokens to one context image's tokens."""
    layer_attn = attentions[layer_index][0]       # (num_heads, seq_len, seq_len)
    attn = layer_attn.mean(dim=0)                 # average over heads
    q_idx = torch.tensor(query_positions, device=attn.device)
    c_idx = torch.tensor(context_positions, device=attn.device)
    sub = attn.index_select(0, q_idx).index_select(1, c_idx)  # (|Q|, |C_k|)
    return sub.mean(dim=0)                        # average over query tokens


def save_query_context_mass_maps(
    attention_data: Dict[str, Any],
    context_images: List[Union[str, Image.Image]],
    output_dir: Union[str, Path],
    file_prefix: str = "qc_mass",
    num_query_images: int = 1,
) -> Optional[Dict[str, Any]]:
    """Generate query-to-context attention mass heatmaps for each context image."""
    attentions = attention_data.get("attentions")
    if attentions is None or len(attentions) == 0:
        return None

    query_pos_list, ctx_pos_list, _, ctx_grid_thws = _resolve_query_and_context_positions(
        attention_data, num_query_images=num_query_images,
    )

    # Flatten all query image positions into a single list for aggregation
    query_pos = [pos for qp in query_pos_list for pos in qp]

    if len(context_images) != len(ctx_pos_list):
        raise ValueError(
            f"Got {len(context_images)} context images but "
            f"{len(ctx_pos_list)} context token runs."
        )

    num_layers = len(attentions)
    layer_views = {
        "early": num_layers // 4,
        "middle": num_layers // 2,
        "last": -1,
    }

    output_dir = Path(output_dir)
    saved_paths: List[str] = []
    mass_summary: Dict[str, Any] = {}

    for ctx_i, (ctx_positions, ctx_thw, ctx_img) in enumerate(
        zip(ctx_pos_list, ctx_grid_thws, context_images)
    ):
        ctx_label = f"context_{ctx_i + 1}"
        ctx_dir = output_dir / ctx_label
        ctx_dir.mkdir(parents=True, exist_ok=True)

        ctx_mass: Dict[str, float] = {}

        # Per-layer views
        for view_name, layer_idx in layer_views.items():
            try:
                vec = _extract_query_to_context_vector(
                    attentions, layer_idx, query_pos, ctx_positions
                )
                total_mass = float(vec.sum().item())
                ctx_mass[view_name] = total_mass

                heatmap = _build_attention_grid(vec, ctx_thw)
                overlay = overlay_heatmap(ctx_img, heatmap)
                fname = f"{file_prefix}_{view_name}.png"
                overlay.save(ctx_dir / fname)
                saved_paths.append(f"{ctx_label}/{fname}")
            except Exception as exc:
                print(f"  {file_prefix} {view_name} layer for {ctx_label} skipped: {exc}")

        # Aggregate (mean across all layers)
        try:
            all_vectors = [
                _extract_query_to_context_vector(
                    attentions, li, query_pos, ctx_positions
                )
                for li in range(num_layers)
            ]
            agg_vec = torch.stack(all_vectors, dim=0).mean(dim=0)
            ctx_mass["aggregate"] = float(agg_vec.sum().item())

            heatmap = _build_attention_grid(agg_vec, ctx_thw)
            overlay = overlay_heatmap(ctx_img, heatmap)
            fname = f"{file_prefix}_aggregate.png"
            overlay.save(ctx_dir / fname)
            saved_paths.append(f"{ctx_label}/{fname}")
        except Exception as exc:
            print(f"  {file_prefix} aggregate for {ctx_label} skipped: {exc}")

        mass_summary[ctx_label] = ctx_mass

    # Write mass summary JSON
    summary_path = output_dir / f"{file_prefix}_mass_summary.json"
    with open(summary_path, "w") as f:
        json.dump(mass_summary, f, indent=2)
    saved_paths.append(summary_path.name)

    return {"saved_paths": saved_paths, "mass_summary": mass_summary}


# ============================================================================
# Heatmap output helpers
# ============================================================================


def build_heatmap_output_path(
    base_dir: str,
    model_path: str,
    task_name: str,
    icl: bool = False,
    num_icl_examples: int = 0,
    run_config: str = "default",
    enable_thinking: bool = False,
) -> Path:
    """Build output path under ``analysis_type="heat_maps"``."""
    return build_output_path(
        base_dir=base_dir,
        analysis_type="heat_maps",
        model_path=model_path,
        task_name=task_name,
        icl=icl,
        num_icl_examples=num_icl_examples,
        run_config=run_config,
        enable_thinking=enable_thinking,
    )


def _save_image_to_dir(image: Union[str, Image.Image], output_dir: Path) -> None:
    """Save *image* into *output_dir* as ``input_image.*``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(image, str):
        shutil.copy(image, output_dir / f"input_image{Path(image).suffix or '.jpg'}")
    elif isinstance(image, Image.Image):
        if image.mode in ("RGBA", "LA", "P"):
            rgb = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "P":
                image = image.convert("RGBA")
            rgb.paste(image, mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None)
            image = rgb
        elif image.mode != "RGB":
            image = image.convert("RGB")
        image.save(output_dir / "input_image.jpg")


def write_heatmap_metadata(
    base_path: Path,
    *,
    mode: str,
    model_path: str,
    seed: int,
    task_name: str,
    **kwargs: Any,
) -> None:
    """Write ``metadata.json`` at *base_path*."""
    base_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "mode": mode,
        "model_path": model_path,
        "seed": seed,
        "task_name": task_name,
        **kwargs,
    }
    with open(base_path / "metadata.json", "w") as f:
        json.dump(payload, f, indent=2)


def save_heatmap_sample(
    *,
    attention_data: Dict[str, Any],
    image: Union[str, Image.Image],
    sample_dir: Path,
    answer_text: Optional[str] = None,
    result_data: Optional[Dict[str, Any]] = None,
    icl_context_images: Optional[List[Optional[Union[str, Image.Image]]]] = None,
    icl_context_demo_sizes: Optional[List[int]] = None,
    query_images: Optional[List[Union[str, Image.Image]]] = None,
    query_file_prefix: str = "attention",
) -> Dict[str, Any]:
    """Save input image, multi-layer heatmaps, and results.json for one sample."""
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Default query_images to [image] for backward compat
    if query_images is None:
        query_images = [image] if image is not None else []
    num_query_images = len(query_images)

    # Save query input image(s)
    if num_query_images == 1 and query_images[0] is not None:
        _save_image_to_dir(query_images[0], sample_dir)
    elif num_query_images > 1:
        for qi, q_img in enumerate(query_images):
            if q_img is not None:
                q_dir = sample_dir / f"query_img{qi + 1}"
                q_dir.mkdir(parents=True, exist_ok=True)
                _save_image_to_dir(q_img, q_dir)

    attn_paths: List[Path] = []
    context_attn_dirs: List[str] = []

    if icl_context_images is not None:
        # --- ICL mode: context images first, then query ---
        num_context_runs = len([img for img in icl_context_images if img is not None])

        # Context image loop with demo_sizes-aware directory naming
        seq_idx = 0
        if icl_context_demo_sizes is not None:
            flat_idx = 0
            for demo_idx, demo_size in enumerate(icl_context_demo_sizes):
                for img_idx in range(demo_size):
                    if flat_idx >= len(icl_context_images):
                        break
                    ctx_img = icl_context_images[flat_idx]
                    if ctx_img is None:
                        flat_idx += 1
                        continue
                    # Directory naming: simple when 1 image per demo, indexed when >1
                    if demo_size == 1:
                        dir_name = f"context_{demo_idx + 1}"
                    else:
                        dir_name = f"context_{demo_idx + 1}_img{img_idx + 1}"
                    ctx_dir = sample_dir / dir_name
                    ctx_dir.mkdir(exist_ok=True)
                    try:
                        _save_image_to_dir(ctx_img, ctx_dir)
                        save_multi_layer_attention_overlays(
                            attention_data=attention_data,
                            image=ctx_img,
                            output_dir=ctx_dir,
                            answer_text=answer_text,
                            file_prefix="attention",
                            image_index=seq_idx,
                        )
                        context_attn_dirs.append(dir_name)
                    except Exception as exc:
                        print(f"  Context attention viz skipped: {exc}")
                    seq_idx += 1
                    flat_idx += 1
        else:
            # Legacy path: no demo_sizes, use original indexing
            present_indices = [
                i for i, img in enumerate(icl_context_images) if img is not None
            ]
            for si, ctx_list_idx in enumerate(present_indices):
                ctx_dir = sample_dir / f"context_{ctx_list_idx + 1}"
                ctx_dir.mkdir(exist_ok=True)
                try:
                    ctx_img = icl_context_images[ctx_list_idx]
                    if ctx_img is None:
                        continue
                    _save_image_to_dir(ctx_img, ctx_dir)
                    save_multi_layer_attention_overlays(
                        attention_data=attention_data,
                        image=ctx_img,
                        output_dir=ctx_dir,
                        answer_text=answer_text,
                        file_prefix="attention",
                        image_index=si,
                    )
                    context_attn_dirs.append(f"context_{ctx_list_idx + 1}")
                except Exception as exc:
                    print(f"  Context attention viz skipped: {exc}")
                seq_idx += 1

        # Query image heatmaps
        if num_query_images == 1:
            # Single query image: save to sample_dir with prefix (same as before)
            try:
                attn_paths = save_multi_layer_attention_overlays(
                    attention_data=attention_data,
                    image=query_images[0],
                    output_dir=sample_dir,
                    answer_text=answer_text,
                    file_prefix="attention_query",
                    image_index=-1,
                )
            except Exception as exc:
                print(f"  Query attention viz skipped: {exc}")
        else:
            # Multiple query images: per-image subdirs
            for qi, q_img in enumerate(query_images):
                if q_img is None:
                    continue
                q_dir = sample_dir / f"query_img{qi + 1}"
                q_dir.mkdir(parents=True, exist_ok=True)
                try:
                    q_image_index = num_context_runs + qi
                    paths = save_multi_layer_attention_overlays(
                        attention_data=attention_data,
                        image=q_img,
                        output_dir=q_dir,
                        answer_text=answer_text,
                        file_prefix="attention_query",
                        image_index=q_image_index,
                    )
                    attn_paths.extend(paths)
                except Exception as exc:
                    print(f"  Query image {qi + 1} attention viz skipped: {exc}")

        # Query→Context attention mass maps
        try:
            present_ctx = [img for img in icl_context_images if img is not None]
            qc_mass_info = save_query_context_mass_maps(
                attention_data=attention_data,
                context_images=present_ctx,
                output_dir=sample_dir,
                num_query_images=num_query_images,
            )
            if qc_mass_info:
                result_data = result_data if result_data is not None else {}
                result_data["query_context_mass"] = qc_mass_info
        except Exception as exc:
            print(f"  Query→Context mass viz skipped: {exc}")
    else:
        # --- Standard mode ---
        if num_query_images == 1:
            # Single image: unchanged behavior
            try:
                attn_paths = save_multi_layer_attention_overlays(
                    attention_data=attention_data,
                    image=query_images[0] if query_images else image,
                    output_dir=sample_dir,
                    answer_text=answer_text,
                    file_prefix=query_file_prefix,
                )
            except Exception as exc:
                print(f"  Attention viz skipped: {exc}")
        else:
            # Multiple query images: per-image subdirs
            for qi, q_img in enumerate(query_images):
                if q_img is None:
                    continue
                q_dir = sample_dir / f"query_img{qi + 1}"
                q_dir.mkdir(parents=True, exist_ok=True)
                try:
                    paths = save_multi_layer_attention_overlays(
                        attention_data=attention_data,
                        image=q_img,
                        output_dir=q_dir,
                        answer_text=answer_text,
                        file_prefix=query_file_prefix,
                        image_index=qi,
                    )
                    attn_paths.extend(paths)
                except Exception as exc:
                    print(f"  Query image {qi + 1} attention viz skipped: {exc}")

    # Build / augment result_data
    if result_data is None:
        result_data = {}
    if attn_paths:
        result_data["attention_visualizations"] = [p.name for p in attn_paths]
        result_data["attention_answer_index"] = 0
    if context_attn_dirs:
        result_data["context_attention_dirs"] = context_attn_dirs

    with open(sample_dir / "results.json", "w") as f:
        json.dump(result_data, f, indent=2)

    return result_data
