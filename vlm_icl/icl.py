"""ICL demo pools and selection, answer corruption, and special ablation modes."""

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .images import resolve_image


# ============================================================================
# Special Image Manipulation for Ablation Studies
# ============================================================================

_MODULE_DIR = Path(__file__).parent.parent  # repo root


BLANK_IMAGE_PATH = str(_MODULE_DIR / "data/visual_reasoning/blank.png")


def apply_special_mode(image_obj, special_mode: Optional[str], is_icd: bool):
    """Apply image manipulation for ablation studies."""
    if special_mode is None:
        return image_obj

    if special_mode == "remove_image":
        return None
    elif special_mode == "remove_image_icd" and is_icd:
        return None
    elif special_mode == "blank_image":
        return BLANK_IMAGE_PATH
    elif special_mode == "blank_image_icd" and is_icd:
        return BLANK_IMAGE_PATH

    return image_obj


def validate_special_mode(
    special: Optional[str],
    icl: bool,
    num_icl_examples: int,
) -> None:
    """Validate special mode configuration and warn on ineffective settings."""
    if special is None:
        return
    needs_icl = special.endswith("_icd") or special.startswith("corrupt_icd")
    if needs_icl and not icl:
        print("Warning: ICD-targeting modes require ICL (--icl flag)")
    if needs_icl and num_icl_examples == 0:
        print("Warning: ICD-targeting modes require ICL examples > 0")
    if special in {"corrupt_icd_beginning", "corrupt_icd_middle", "corrupt_icd_end"}:
        if num_icl_examples < 4:
            raise ValueError(
                f"Non-default corrupt_icd spans require num_icl_examples >= 4, got {num_icl_examples}"
            )
    if special in {"blank_image", "blank_image_icd"} and not Path(BLANK_IMAGE_PATH).exists():
        raise FileNotFoundError(f"Blank image not found at {BLANK_IMAGE_PATH}")


def shuffle_icl_answers(
    context: List[Dict[str, Any]],
    special: Optional[str],
    answer_rng: random.Random,
) -> List[str]:
    """Return ICL answers, shuffled if special mode is 'random_icd'."""
    answers = [ctx["question"]["ground_truth"] for ctx in context]
    if special == "random_icd":
        answer_rng.shuffle(answers)
    return answers


def process_icl_context_images(
    context: List[Dict[str, Any]],
    shuffled_answers: List[str],
    special: Optional[str],
    format_answer_fn: Optional[Callable[[Optional[str], Optional[str]], Optional[str]]] = None,
    prefer_pil: bool = False,
) -> List[Dict[str, Any]]:
    """Apply special-mode transforms to ICL images and build formatted context list."""
    if len(context) != len(shuffled_answers):
        raise ValueError(f"Context length ({len(context)}) != shuffled_answers length ({len(shuffled_answers)})")

    icl_context = []
    for ctx, answer in zip(context, shuffled_answers):
        ctx_image_obj = apply_special_mode(ctx["entry"]["image"], special, is_icd=True)
        ctx_image = resolve_image(ctx_image_obj, prefer_pil=prefer_pil) if ctx_image_obj is not None else None
        if format_answer_fn:
            formatted_answer = format_answer_fn(ctx["question"]["prompt"], answer)
        else:
            formatted_answer = answer
        item = {"image": ctx_image, "prompt": ctx["question"]["prompt"], "answer": formatted_answer}
        # Resolve extra images if present (multi-image datasets)
        for extra_key in ("image2", "image3", "image4"):
            extra_obj = ctx["entry"].get(extra_key)
            if extra_obj is not None:
                extra_obj = apply_special_mode(extra_obj, special, is_icd=True)
                if extra_obj is not None:
                    item[extra_key] = resolve_image(extra_obj, prefer_pil=prefer_pil)
        icl_context.append(item)
    return icl_context


# ============================================================================
# Demo Pools and Selection
# ============================================================================

def build_icl_context_pools(data: List[Dict[str, Any]]) -> Dict[tuple, List[Dict]]:
    """Build ICL context pools grouped by (task, split)."""
    pools = {}
    for entry in data:
        key = (entry["task"], entry["split"])
        for question in entry.get('questions', []):
            if key not in pools:
                pools[key] = []
            pools[key].append({
                'entry': entry,
                'question': question
            })
    return pools


def build_label_space(pool: Dict[tuple, List[Dict]]) -> Dict[tuple, List[str]]:
    """Return sorted unique ground-truth labels per (task, split) key."""
    return {
        key: sorted({item["question"]["ground_truth"] for item in items})
        for key, items in pool.items()
    }


def compute_corrupt_indices(num_demos: int, span: str) -> List[int]:
    """Return demo indices to corrupt for the given span."""
    if span == "default":
        return list(range(num_demos))
    quarter = num_demos // 4
    if span == "beginning":
        return list(range(quarter))
    if span == "middle":
        start = (num_demos - quarter) // 2
        return list(range(start, start + quarter))
    if span == "end":
        return list(range(num_demos - quarter, num_demos))
    raise ValueError(f"Unknown corruption span: {span!r}")


def corrupt_icl_answers(
    context: List[Dict[str, Any]],
    special: str,
    answer_rng: random.Random,
    label_space: List[str],
) -> List[str]:
    """Return ICL answers with corrupted positions replaced by random labels."""
    answers = [ctx["question"]["ground_truth"] for ctx in context]
    span = "default" if special == "corrupt_icd" else special.removeprefix("corrupt_icd_")
    for i in compute_corrupt_indices(len(answers), span):
        answers[i] = answer_rng.choice(label_space)
    return answers


def build_control_demo_set(
    pools: Dict[tuple, List[Dict]],
    max_demos: int = 32,
    seed: int = 42,
) -> Tuple[Dict[tuple, List[Dict]], Dict[tuple, List[str]]]:
    """Build a fixed, balanced, deterministic demo set per (task, split) key.

    Returns ``(demos, corrupted_answers)``. ``demos`` holds up to *max_demos* items per
    key, balanced across labels and shuffled into a single canonical ordering with a
    fixed seed. ``corrupted_answers`` holds, per key, the demos' answers with every
    position replaced by a random label from that key's label space.
    """
    rng = random.Random(seed)
    result: Dict[tuple, List[Dict]] = {}
    for key, items in pools.items():
        # Group by ground-truth label
        by_label: Dict[str, List[Dict]] = {}
        for item in items:
            gt = item["question"]["ground_truth"]
            by_label.setdefault(gt, []).append(item)
        labels = sorted(by_label)
        n_labels = len(labels)
        per_label = max_demos // n_labels if n_labels else 0

        selected: List[Dict] = []
        remainder_pool: List[Dict] = []
        for label in labels:
            pool = list(by_label[label])
            rng.shuffle(pool)
            selected.extend(pool[:per_label])
            remainder_pool.extend(pool[per_label:])

        # Fill remainder round-robin
        rng.shuffle(remainder_pool)
        need = max_demos - len(selected)
        selected.extend(remainder_pool[:need])

        # Canonical ordering
        rng.shuffle(selected)
        result[key] = selected

    label_spaces = build_label_space(pools)
    corrupted_answers = {
        key: corrupt_icl_answers(
            demos, special="corrupt_icd", answer_rng=random.Random(seed), label_space=label_spaces[key],
        )
        for key, demos in result.items()
    }
    return result, corrupted_answers


def build_demo_pool(
    context_pools: Dict[tuple, List[Dict]],
    seed: Optional[int] = None,
) -> Dict[tuple, List[Dict]]:
    """Build a shared, label-balanced demo pool per (task, split)."""
    rng = random.Random(seed)
    result: Dict[tuple, List[Dict]] = {}

    for key, candidates in context_pools.items():
        # Group candidates by ground_truth
        gt_groups: Dict[str, List[Dict]] = {}
        for item in candidates:
            gt_groups.setdefault(item["question"]["ground_truth"], []).append(item)

        min_count = min(len(group) for group in gt_groups.values())
        pool_size = min_count * len(gt_groups)

        if len(candidates) <= pool_size:
            # Use entire partition as pool
            pool = list(candidates)
        else:
            # Sample exactly min_count from each GT group (perfectly balanced)
            unique_gts = list(gt_groups.keys())
            rng.shuffle(unique_gts)

            pool = []
            for gt in unique_gts:
                pool.extend(rng.sample(gt_groups[gt], min_count))

        rng.shuffle(pool)
        result[key] = pool

    return result


def select_from_demo_pool(
    query_entry: Dict,
    query_question: Dict,
    demo_pool: Dict[tuple, List[Dict]],
    num_examples: int,
    seed: Optional[int] = None,
    query_gt_position: Optional[str] = None,
    ensure_query_gt: bool = False,
) -> Optional[List[Dict]]:
    """Select balanced ICL context from a pre-built demo pool, excluding the query sample.

    With ``ensure_query_gt``, the query's ground-truth label is allocated a demo first.
    ``query_gt_position`` ("beginning" or "end") then moves that demo after the shuffle.
    """
    key = (query_entry["task"], query_entry["split"])

    if key not in demo_pool:
        return None

    rng = random.Random(seed) if seed is not None else random.Random()

    # Exclude query by image_id
    candidates = [
        item for item in demo_pool[key]
        if item['entry']['image_id'] != query_entry['image_id']
    ]

    if len(candidates) < num_examples:
        return None

    # Group candidates by ground_truth for balanced sampling
    gt_groups: Dict[str, List[Dict]] = {}
    for item in candidates:
        gt_groups.setdefault(item["question"]["ground_truth"], []).append(item)

    unique_gts = sorted(gt_groups.keys())
    rng.shuffle(unique_gts)

    # Conditionally prioritize query GT for allocation
    if ensure_query_gt:
        query_gt = query_question["ground_truth"]
        if query_gt in unique_gts:
            unique_gts.remove(query_gt)
            unique_gts.insert(0, query_gt)

    n_unique = len(unique_gts)
    base = num_examples // n_unique
    remainder = num_examples % n_unique

    selected = []
    for i, gt in enumerate(unique_gts):
        group = gt_groups[gt]
        n = min(base + (1 if i < remainder else 0), len(group))
        selected.extend(rng.sample(group, n))

    # Fallback: fill deficit from remaining candidates
    if len(selected) < num_examples:
        selected_ids = {id(x) for x in selected}
        remaining = [c for c in candidates if id(c) not in selected_ids]
        deficit = num_examples - len(selected)
        if len(remaining) >= deficit:
            selected.extend(rng.sample(remaining, deficit))
        else:
            return None

    rng.shuffle(selected)

    if query_gt_position is not None:
        selected = reposition_gt_demo(selected, query_question["ground_truth"], query_gt_position)

    return selected[:num_examples]


def ensure_gt_demo(
    selected: List[Dict],
    query_gt: str,
    pool_candidates: List[Dict],
    rng: random.Random,
) -> List[Dict]:
    """Ensure at least one demo matching the query's ground truth is present, swapping one in if needed."""
    if any(d["question"]["ground_truth"] == query_gt for d in selected):
        return selected

    selected_ids = {id(x) for x in selected}
    gt_candidates = [
        c for c in pool_candidates
        if c["question"]["ground_truth"] == query_gt and id(c) not in selected_ids
    ]
    if not gt_candidates:
        return selected  # graceful fallback — no GT candidates available

    selected = list(selected)  # copy to avoid mutating caller's list
    selected[-1] = rng.choice(gt_candidates)
    return selected


def reposition_gt_demo(
    selected: List[Dict],
    query_gt: str,
    position: Optional[str],
) -> List[Dict]:
    """Move the first GT-matching demo to the beginning or end of the list."""
    if position in ("random", None):
        return selected

    gt_demo = next((d for d in selected if d["question"]["ground_truth"] == query_gt), None)
    if gt_demo is None:
        return selected

    selected = list(selected)  # copy to avoid mutating caller's list
    selected.remove(gt_demo)
    if position == "beginning":
        selected.insert(0, gt_demo)
    elif position == "end":
        selected.append(gt_demo)
    return selected


def sample_per_label(
    data: List[Dict[str, Any]],
    n_per_label: int,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Sample exactly n_per_label entries per (task, split, ground_truth) group."""
    rng = random.Random(seed)
    by_group: Dict[tuple, List[Dict]] = {}
    for entry in data:
        for question in entry.get("questions", []):
            key = (entry["task"], entry["split"], question["ground_truth"])
            by_group.setdefault(key, []).append(entry)
    selected = []
    seen_ids: set = set()
    for key in sorted(by_group):
        pool = list(by_group[key])
        rng.shuffle(pool)
        count = 0
        for entry in pool:
            if count >= n_per_label:
                break
            if entry["image_id"] not in seen_ids:
                selected.append(entry)
                seen_ids.add(entry["image_id"])
                count += 1
    return selected
