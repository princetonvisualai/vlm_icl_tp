"""Shared LatentLens / LogitLens utilities for formal analysis experiments."""

from __future__ import annotations

import math
import re
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Image-token grid helpers
# ---------------------------------------------------------------------------

def flatten_ids(input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.dim() == 2:
        return input_ids[0]
    return input_ids


def resolve_image_token_id(tokenizer, input_ids: torch.Tensor) -> Optional[int]:
    for attr in ("image_token_id", "image_pad_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None and token_id in input_ids:
            return int(token_id)

    vocab = getattr(tokenizer, "get_vocab", lambda: {})()
    candidates = []
    for token in vocab.keys():
        if "image_pad" in token or token in ("<|image_pad|>", "<image>", "<|image|>", "<|image_0|>"):
            candidates.append(vocab[token])

    input_set = set(input_ids.tolist())
    for token_id in candidates:
        if token_id in input_set:
            return int(token_id)
    return None


def split_runs(indices: Iterable[int]) -> List[List[int]]:
    indices = list(indices)
    if not indices:
        return []
    runs = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append(list(range(start, prev + 1)))
        start = idx
        prev = idx
    runs.append(list(range(start, prev + 1)))
    return runs


def _infer_grid_thw(num_tokens: int) -> Tuple[int, int, int]:
    side = int(math.sqrt(num_tokens))
    if side * side == num_tokens:
        return 1, side, side

    for h in range(side, 1, -1):
        if num_tokens % h == 0:
            return 1, h, num_tokens // h
    return 1, num_tokens, 1


def resolve_grid_thw(
    image_grid_thw: Optional[torch.Tensor],
    image_index: int,
    num_tokens: int
) -> Tuple[int, int, int]:
    if image_grid_thw is not None:
        thw = image_grid_thw
        # Handle batch dimension: (batch, num_images, 3) -> (num_images, 3)
        if thw.dim() == 3:
            thw = thw[0]
        grid = thw[image_index]
        if isinstance(grid, torch.Tensor):
            grid = grid.tolist()
        if len(grid) == 3 and grid[0] * grid[1] * grid[2] == num_tokens:
            return int(grid[0]), int(grid[1]), int(grid[2])
    return _infer_grid_thw(num_tokens)


# ---------------------------------------------------------------------------
# Layer presets for known models (early / middle / late)
# ---------------------------------------------------------------------------

LATENTLENS_LAYER_PRESETS: dict[str, list[int]] = {
    "qwen3-vl-8b":  [1, 18, 34],   # 36 decoder layers
    "qwen2.5-vl-7b": [1, 14, 26],  # 28 decoder layers
    "qwen3-vl-32b": [1, 32, 62],   # 64 decoder layers
}

_MODEL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"qwen3-vl-8b|qwen3_vl_8b", re.IGNORECASE), "qwen3-vl-8b"),
    (re.compile(r"qwen2[\._]5-vl-7b|qwen2[\._]5_vl_7b", re.IGNORECASE), "qwen2.5-vl-7b"),
    (re.compile(r"qwen3-vl-32b|qwen3_vl_32b", re.IGNORECASE), "qwen3-vl-32b"),
]


def resolve_latentlens_layers(
    model_path: str,
    layers: Optional[list[int]] = None,
    num_layers: Optional[int] = None,
) -> list[int]:
    """Resolve LatentLens layers: explicit > preset > auto from num_layers."""
    if layers is not None:
        return layers

    for pattern, key in _MODEL_PATTERNS:
        if pattern.search(model_path):
            resolved = LATENTLENS_LAYER_PRESETS[key]
            print(f"LatentLens layers (preset {key}): {resolved}")
            return list(resolved)

    if num_layers is not None and num_layers > 2:
        resolved = [1, num_layers // 2, num_layers - 2]
        print(f"LatentLens layers (auto, {num_layers} total): {resolved}")
        return resolved

    raise ValueError(
        f"Cannot resolve LatentLens layers: no preset for model_path={model_path!r} "
        f"and num_layers not provided. Pass --layers explicitly."
    )


def detect_num_layers(model) -> Optional[int]:
    """Detect number of decoder layers from model config."""
    try:
        cfg = model.model.config
    except AttributeError:
        return None
    for obj in (cfg, getattr(cfg, "text_config", None)):
        if obj is not None:
            n = getattr(obj, "num_hidden_layers", None)
            if n is not None:
                return n
    return None


# ---------------------------------------------------------------------------
# Query token helpers
# ---------------------------------------------------------------------------

def get_query_token_info(tokenizer, input_ids, image_grid_thw):
    """Find the last image-token run (query image) and resolve its grid shape."""
    flat_ids = flatten_ids(input_ids)
    image_token_id = resolve_image_token_id(tokenizer, flat_ids)
    if image_token_id is None:
        return None, None
    positions = torch.where(flat_ids == image_token_id)[0].tolist()
    runs = split_runs(positions)
    if not runs:
        return None, None
    query_run = runs[-1]
    grid_thw = resolve_grid_thw(image_grid_thw, len(runs) - 1, len(query_run))
    return query_run, grid_thw


def neighbor_to_dict(neighbor):
    return {
        "token_str": neighbor.token_str,
        "similarity": float(neighbor.similarity),
        "caption": neighbor.caption,
    }


# ---------------------------------------------------------------------------
# Kneedle threshold
# ---------------------------------------------------------------------------

def kneedle_threshold(similarities):
    """Find threshold at the knee point of sorted similarities.

    Sort descending, normalize to [0,1], find max perpendicular distance
    to the line connecting first and last points.
    """
    if len(similarities) <= 1:
        return similarities[0] if similarities else 0.0
    s = sorted(similarities, reverse=True)
    n = len(s)
    y_min, y_max = s[-1], s[0]
    if y_max == y_min:
        return s[0]
    x = [i / (n - 1) for i in range(n)]
    y = [(v - y_min) / (y_max - y_min) for v in s]
    a = y[0] - y[-1]
    b = 1.0
    c = -y[0]
    denom = (a * a + b * b) ** 0.5
    best_i = max(range(n), key=lambda i: abs(a * x[i] + b * y[i] + c))
    return s[best_i]


# ---------------------------------------------------------------------------
# Distractor identification
# ---------------------------------------------------------------------------

def identify_distracting_positions(
    model, index, image, prompt, top_k, distract_cue, context=None, layer=None,
):
    """Run LatentLens and return query-image token positions matching the distractor keyword.

    Returns (distract_positions, grid_thw, n_query_tokens, query_positions, hidden_states).
    """
    messages = model.build_messages(image, prompt, context=context)
    inputs = model.prepare_inputs(messages)

    with torch.no_grad():
        import latentlens

        hidden_states = latentlens.get_hidden_states(model.model, **inputs)

    tokenizer = getattr(model.processor, "tokenizer", model.processor)
    query_positions, grid_thw = get_query_token_info(
        tokenizer, inputs["input_ids"], inputs.get("image_grid_thw"),
    )
    if query_positions is None:
        return set(), None, 0, [], hidden_states

    if layer is None:
        layer = min(index.available_layers)
    keyword = distract_cue.lower()
    query = F.normalize(hidden_states[layer][0, query_positions, :], dim=-1)
    neighbors = index.search(query, top_k=top_k, layers=[layer])

    distract_positions = set()
    for idx, pos in enumerate(query_positions):
        nns = neighbors[idx]
        sims = [nns[k].similarity for k in range(len(nns))]
        threshold = kneedle_threshold(sims)
        for k in range(len(nns)):
            if keyword in nns[k].token_str.lower() and nns[k].similarity >= threshold:
                distract_positions.add(pos)
                break

    return distract_positions, grid_thw, len(query_positions), query_positions, hidden_states


def classify_query_tokens(
    model, index, image, prompt, top_k, prior_cue, *,
    grounding_cues: Optional[list[str]] = None,
    context=None, layer=None,
):
    """Run LatentLens and classify each query-image token into one of three buckets.

    Buckets: "prior_cues", "grounding_cues", "others".
    Priority: prior_cues > grounding_cues > others.
    Uses max(index.available_layers) by default (the "last layer" in LatentLens index).

    `grounding_cues` is a list of lowercase substrings; any hit classifies the token.
    Gated by `kneedle_threshold` and uses the current substring-match rule on
    `nns[k].token_str.lower()`.

    Returns (bucket_by_position, query_positions, n_query_tokens).
    """
    messages = model.build_messages(image, prompt, context=context)
    inputs = model.prepare_inputs(messages)

    with torch.no_grad():
        import latentlens

        hidden_states = latentlens.get_hidden_states(model.model, **inputs)

    tokenizer = getattr(model.processor, "tokenizer", model.processor)
    query_positions, _ = get_query_token_info(
        tokenizer, inputs["input_ids"], inputs.get("image_grid_thw"),
    )
    if query_positions is None:
        return {}, [], 0

    if layer is None:
        layer = max(index.available_layers)
    prior_kw = prior_cue.lower() if prior_cue else None
    grounding_kws = [g for g in (grounding_cues or []) if g]
    query = F.normalize(hidden_states[layer][0, query_positions, :], dim=-1)
    neighbors = index.search(query, top_k=top_k, layers=[layer])

    bucket_by_position: dict[int, str] = {}
    for idx, pos in enumerate(query_positions):
        nns = neighbors[idx]
        sims = [nns[k].similarity for k in range(len(nns))]
        threshold = kneedle_threshold(sims)

        prior_hit = False
        grounding_hit = False
        if prior_kw or grounding_kws:
            for k in range(len(nns)):
                if nns[k].similarity < threshold:
                    continue
                tok = nns[k].token_str.lower()
                if prior_kw and prior_kw in tok:
                    prior_hit = True
                    break
                if grounding_kws and any(g in tok for g in grounding_kws):
                    grounding_hit = True

        if prior_hit:
            bucket_by_position[pos] = "prior_cues"
        elif grounding_hit:
            bucket_by_position[pos] = "grounding_cues"
        else:
            bucket_by_position[pos] = "others"

    return bucket_by_position, query_positions, len(query_positions)


# ---------------------------------------------------------------------------
# Grid position mapping
# ---------------------------------------------------------------------------

def abs_to_grid_positions(distract_positions, query_run, grid_thw):
    """Convert absolute distractor positions to grid-relative [row, col] pairs."""
    _t, _h, w = grid_thw
    pos_to_idx = {pos: idx for idx, pos in enumerate(query_run)}
    return [
        [pos_to_idx[pos] // w, pos_to_idx[pos] % w]
        for pos in sorted(distract_positions)
        if pos in pos_to_idx
    ]


# ---------------------------------------------------------------------------
# LogitLens projection
# ---------------------------------------------------------------------------

def get_lm_head_and_norm(model):
    """Extract lm_head and final_norm from an HF model wrapper."""
    hf = model.model
    if hasattr(hf, "lm_head") and hasattr(hf, "model"):
        inner = hf.model
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "norm"):
            return hf.lm_head, inner.language_model.norm
        if hasattr(inner, "norm"):
            return hf.lm_head, inner.norm
    raise ValueError(
        f"Cannot locate lm_head/final_norm on {type(hf).__name__}. "
        "Add a branch for this model family in lens_utils.get_lm_head_and_norm()."
    )


def logitlens_project(hidden_states, layer, positions, lm_head, final_norm, tokenizer, top_k):
    """Project hidden states through norm -> lm_head -> softmax -> top-k decode.

    Returns list of lists of {"token_str", "similarity", "caption"} dicts.
    """
    hs = final_norm(hidden_states[layer][0, positions, :])
    logits = lm_head(hs)
    probs = F.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, top_k, dim=-1)

    return [
        [
            {"token_str": tokenizer.decode(top_ids[i, k].item()),
             "similarity": float(top_probs[i, k].item()),
             "caption": ""}
            for k in range(top_k)
        ]
        for i in range(len(positions))
    ]
