"""Discovery, loading and aggregation of ICL experiment results."""

from __future__ import annotations

import functools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .models import get_model_family


ModelFilter = Callable[[str, str], bool]  # (model_name, family) -> keep?


LeafFilter = Callable[[Path], bool]  # leaf_dir -> include?


def load_summaries(input_dir: Path) -> list[dict[str, Any]]:
    """Load all ``shots_*/summary.json`` from *input_dir*, sorted by shot count."""
    summaries = []
    for d in sorted(input_dir.iterdir()):
        if d.is_dir() and d.name.startswith("shots_"):
            summary_path = d / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summaries.append(json.load(f))
    summaries.sort(key=lambda s: s["n_shots"])
    return summaries


def discover_model_summaries(
    base_dir: Path,
    *,
    leaf_filter: LeafFilter | None = None,
) -> dict[str, dict[str, dict[str, tuple[Path, list[dict[str, Any]]]]]]:
    """Walk ``base_dir/{task}/{model}/{leaf}/shots_*/summary.json``.

    Returns ``{task: {leaf: {model: (leaf_path, summaries)}}}``.
    """
    result: dict[str, dict[str, dict[str, tuple[Path, list[dict[str, Any]]]]]] = {}
    for task_dir in sorted(base_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        for model_dir in sorted(task_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for leaf_dir in sorted(model_dir.iterdir()):
                if not leaf_dir.is_dir():
                    continue
                if leaf_filter is not None and not leaf_filter(leaf_dir):
                    continue
                if not any(
                    (leaf_dir / d.name / "summary.json").exists()
                    for d in leaf_dir.iterdir()
                    if d.is_dir() and d.name.startswith("shots_")
                ):
                    continue
                summaries = load_summaries(leaf_dir)
                if summaries:
                    result.setdefault(task_dir.name, {}).setdefault(
                        leaf_dir.name, {}
                    )[model_dir.name] = (leaf_dir, summaries)
    return result


def extract_split_from_leaf(leaf_name: str) -> str:
    """Extract split name from leaf like 'npl20_simple' -> 'simple'."""
    for split in ("simple", "noprior", "complex"):
        if split in leaf_name:
            return split
    return "unknown"


@functools.lru_cache(maxsize=None)
def load_demo_set_counts(
    leaf_path_str: str,
    n_shots: int,
    keys: tuple[str, ...] = ("overall", "original", "perturbed"),
) -> dict[str, dict[int, tuple[int, int]]]:
    """Read ``<leaf>/shots_<n>/results.jsonl`` and return per-demo_set counts.

    Returns ``{key: {demo_set_idx: (correct, total)}}``. Records without
    ``demo_set_idx`` are skipped (older runs). Keys ``"original"``/``"perturbed"``
    filter on ``record["alignment"]``; ``"overall"`` matches every record.
    Empty key slices are dropped from the result.
    """
    path = Path(leaf_path_str) / f"shots_{n_shots}" / "results.jsonl"
    records = load_jsonl(path)
    out: dict[str, dict[int, tuple[int, int]]] = {k: {} for k in keys}
    for r in records:
        d = r.get("demo_set_idx")
        if d is None:
            continue
        is_correct = 1 if r.get("correct") else 0
        align = r.get("alignment")
        for k in keys:
            if k == "overall" or k == align:
                c, t = out[k].get(d, (0, 0))
                out[k][d] = (c + is_correct, t + 1)
    return {k: v for k, v in out.items() if v}


@functools.lru_cache(maxsize=None)
def load_per_query_demo_correct(
    leaf_path_str: str,
    n_shots: int,
    keys: tuple[str, ...] = ("overall", "original", "perturbed"),
) -> dict[str, dict[str, dict[int, int]]]:
    """Read ``<leaf>/shots_<n>/results.jsonl`` and return per-query, per-demo correctness.

    Returns ``{key: {image_id: {demo_set_idx: 0|1}}}``. Used to compute the
    per-query std across the 10 demo-set outcomes for a query, which is then
    averaged across queries to get the (task, split) variance unit.
    Records without ``demo_set_idx`` or ``image_id`` are skipped. Empty key
    slices are dropped.
    """
    path = Path(leaf_path_str) / f"shots_{n_shots}" / "results.jsonl"
    records = load_jsonl(path)
    out: dict[str, dict[str, dict[int, int]]] = {k: {} for k in keys}
    for i, r in enumerate(records):
        d = r.get("demo_set_idx")
        if d is None:
            continue
        image_id = r.get("image_id")
        if image_id is None:
            image_id = f"_row{i}"
        is_correct = 1 if r.get("correct") else 0
        align = r.get("alignment")
        for k in keys:
            if k == "overall" or k == align:
                out[k].setdefault(image_id, {})[d] = is_correct
    return {k: v for k, v in out.items() if v}


def mean_query_std(
    per_query: dict[str, dict[int, int]],
) -> float | None:
    """(task, split) variance unit: mean over queries of per-query demo-set std.

    For each query, computes std (ddof=1) across the demo-set 0/1 correctness
    values for that query, then averages across queries. Queries with fewer
    than 2 demos are skipped. Returns ``None`` when no query qualifies.
    """
    stds: list[float] = []
    for by_demo in per_query.values():
        if len(by_demo) < 2:
            continue
        arr = np.asarray(list(by_demo.values()), dtype=float)
        stds.append(float(arr.std(ddof=1)))
    if not stds:
        return None
    return float(np.mean(stds))


def average_leaf_yerrs(
    yerrs: Sequence[float | None],
) -> float | None:
    """Average leaf-level yerrs (per-(task, split) variance units) for cross-task / cross-split.

    ``None`` entries are skipped. Returns ``None`` when no leaf qualifies.
    """
    vals = [y for y in yerrs if y is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def attach_variance_to_summaries(
    leaf_path: Path,
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Augment each summary in-place with sibling variance fields.

    For each summary, reads ``<leaf>/shots_<n>/results.jsonl`` and writes:

    - ``summary["accuracy_yerr"]: {key: float | None}`` — (task, split) variance
      unit: mean over queries of std-across-demo-sets-for-that-query.
    - ``summary["accuracy_per_demo_set"]: {key: {demo_idx: acc}}`` — per-seed
      pooled accuracy across queries (informational, not used for variance).
    - ``summary["_demo_set_counts"]: {key: {demo_idx: (correct, total)}}``

    Idempotent (skips summaries that already carry ``_demo_set_counts``). Keys
    are derived from each summary's existing ``accuracy`` dict — simple/noprior
    runs only carry ``"overall"``; complex runs additionally carry
    ``"original"``/``"perturbed"``.
    """
    for s in summaries:
        if "_demo_set_counts" in s:
            continue
        keys = tuple((s.get("accuracy") or {"overall": 0}).keys()) or ("overall",)
        counts = load_demo_set_counts(str(leaf_path), s["n_shots"], keys=keys)
        per_query = load_per_query_demo_correct(str(leaf_path), s["n_shots"], keys=keys)
        s["_demo_set_counts"] = counts
        yerrs: dict[str, float | None] = {}
        per_seed: dict[str, dict[int, float]] = {}
        for k in keys:
            slice_counts = counts.get(k, {})
            yerrs[k] = mean_query_std(per_query.get(k, {}))
            per_seed[k] = {
                d: c / t for d, (c, t) in slice_counts.items() if t > 0
            }
        s["accuracy_yerr"] = yerrs
        s["accuracy_per_demo_set"] = per_seed
    return summaries


def aggregate_across_tasks(
    model_summaries: dict[str, dict[str, dict[str, tuple[Path, list[dict[str, Any]]]]]],
    target_splits: list[str] | tuple[str, ...],
) -> dict[str, dict[str, dict[int, tuple[int, int]]]]:
    """Aggregate accuracy across all tasks for each (model, split, n_shots)."""
    split_data: dict[str, dict[str, dict[int, tuple[int, int]]]] = {
        split: {} for split in target_splits
    }
    split_data["overall"] = {}

    for leaf_dict in model_summaries.values():
        for leaf, model_dict in leaf_dict.items():
            split = extract_split_from_leaf(leaf)
            if split not in target_splits:
                continue

            for model_name, (_, summaries) in model_dict.items():
                for summary in summaries:
                    n_shots = summary["n_shots"]
                    n_samples = summary["n_samples"]
                    accuracy = summary["accuracy"]["overall"]
                    correct = round(accuracy * n_samples)

                    if model_name not in split_data[split]:
                        split_data[split][model_name] = {}
                    if n_shots not in split_data[split][model_name]:
                        split_data[split][model_name][n_shots] = (0, 0)
                    old_c, old_s = split_data[split][model_name][n_shots]
                    split_data[split][model_name][n_shots] = (
                        old_c + correct,
                        old_s + n_samples,
                    )

                    if model_name not in split_data["overall"]:
                        split_data["overall"][model_name] = {}
                    if n_shots not in split_data["overall"][model_name]:
                        split_data["overall"][model_name][n_shots] = (0, 0)
                    old_c, old_s = split_data["overall"][model_name][n_shots]
                    split_data["overall"][model_name][n_shots] = (
                        old_c + correct,
                        old_s + n_samples,
                    )

    return split_data


def to_summaries_format(
    aggregated: dict[str, dict[int, tuple[int, int]]]
) -> dict[str, list[dict[str, Any]]]:
    """Convert aggregated accuracy counts to the plotting summaries format."""
    result: dict[str, list[dict[str, Any]]] = {}
    for model, shots_data in aggregated.items():
        summaries = []
        for n_shots in sorted(shots_data.keys()):
            correct, samples = shots_data[n_shots]
            acc = correct / samples if samples > 0 else 0.0
            summaries.append({
                "n_shots": n_shots,
                "accuracy": {"overall": acc},
            })
        result[model] = summaries
    return result


VarianceCounts = dict[str, dict[int, tuple[int, int]]]
"""Per-key per-demo_set (correct, total): ``{key: {demo_idx: (c, t)}}``."""


ShotAggregate = dict[str, Any]
"""Per (model, n_shots) aggregate, carrying both pooled counts (for accuracy)
and per-leaf yerrs (for variance via :func:`average_leaf_yerrs`):

``{"counts": {key: {demo_idx: (c, t)}}, "leaf_yerrs": {key: [yerr | None]}}``
"""


def pool_variance_counts(dst: VarianceCounts, src: VarianceCounts) -> None:
    """Merge ``src`` into ``dst`` in place, summing per-demo_set counts."""
    for key, by_demo in src.items():
        dst_by_demo = dst.setdefault(key, {})
        for d, (c, t) in by_demo.items():
            oc, ot = dst_by_demo.get(d, (0, 0))
            dst_by_demo[d] = (oc + c, ot + t)


def new_shot_aggregate() -> ShotAggregate:
    return {"counts": {}, "leaf_yerrs": {}}


def absorb_leaf(slot: ShotAggregate, summary: dict[str, Any]) -> None:
    """Merge a leaf summary into a ``{counts, leaf_yerrs}`` aggregate slot."""
    counts = summary.get("_demo_set_counts") or {}
    pool_variance_counts(slot["counts"], counts)
    yerrs = summary.get("accuracy_yerr") or {}
    for key in counts.keys():
        slot["leaf_yerrs"].setdefault(key, []).append(yerrs.get(key))


def aggregate_across_tasks_with_variance(
    cross_model: dict[str, dict[str, dict[str, tuple[Path, list[dict[str, Any]]]]]],
    target_splits: list[str] | tuple[str, ...],
) -> dict[str, dict[str, dict[int, ShotAggregate]]]:
    """Cross-task aggregation per ``(split, model, n_shots)``.

    Counts are pooled (used to display the aggregated accuracy); per-leaf
    yerrs are collected as a list so the cross-task variance is the simple
    average of the per-(task, split) variance units.

    Returns ``{split: {model: {n_shots: ShotAggregate}}}`` with ``"overall"``
    spanning all listed splits.
    """
    out: dict[str, dict[str, dict[int, ShotAggregate]]] = {
        split: {} for split in target_splits
    }
    out["overall"] = {}

    for leaf_dict in cross_model.values():
        for leaf, model_dict in leaf_dict.items():
            split = extract_split_from_leaf(leaf)
            if split not in target_splits:
                continue
            for model_name, (leaf_path, summaries) in model_dict.items():
                attach_variance_to_summaries(leaf_path, summaries)
                for s in summaries:
                    n_shots = s["n_shots"]
                    for tgt in (split, "overall"):
                        per_model = out[tgt].setdefault(model_name, {})
                        slot = per_model.setdefault(n_shots, new_shot_aggregate())
                        absorb_leaf(slot, s)
    return out


def aggregate_per_task_with_variance(
    cross_model: dict[str, dict[str, dict[str, tuple[Path, list[dict[str, Any]]]]]],
    target_splits: tuple[str, ...] | list[str],
    *,
    model_filter: ModelFilter | None = None,
) -> dict[str, dict[str, dict[int, ShotAggregate]]]:
    """Cross-split aggregation per task.

    Counts are pooled across the listed splits within each task (used for
    accuracy); per-leaf yerrs are collected as a list so the cross-split
    variance is the simple average of the per-(task, split) variance units.

    Returns ``{task: {model: {n_shots: ShotAggregate}}}``.
    """
    out: dict[str, dict[str, dict[int, ShotAggregate]]] = {}
    for task_name, leaf_dict in cross_model.items():
        for leaf, model_dict in leaf_dict.items():
            if extract_split_from_leaf(leaf) not in target_splits:
                continue
            for model_name, (leaf_path, summaries) in model_dict.items():
                if model_filter is not None and not model_filter(
                    model_name, get_model_family(model_name)
                ):
                    continue
                attach_variance_to_summaries(leaf_path, summaries)
                per_model = out.setdefault(task_name, {}).setdefault(model_name, {})
                for s in summaries:
                    n_shots = s["n_shots"]
                    slot = per_model.setdefault(n_shots, new_shot_aggregate())
                    absorb_leaf(slot, s)
    return out


def aggregate_complex_alignments_from_results_with_variance(
    cross_model: dict[str, dict[str, dict[str, tuple[Path, list[dict[str, Any]]]]]],
    *,
    model_filter: ModelFilter | None = None,
) -> dict[str, dict[int, ShotAggregate]]:
    """Cross-task aggregation across all complex-split tasks per model.

    Pools demo-set counts across tasks (for accuracy) and collects per-leaf
    yerrs (for variance via simple averaging), keyed by
    ``"overall" / "real" / "counterfactual"``.

    Returns ``{model: {n_shots: ShotAggregate}}``.
    """
    out: dict[str, dict[int, ShotAggregate]] = {}
    align_to_key = {"original": "real", "perturbed": "counterfactual"}
    raw_keys = ("overall", "original", "perturbed")
    for leaf_dict in cross_model.values():
        for leaf, model_dict in leaf_dict.items():
            if extract_split_from_leaf(leaf) != "complex":
                continue
            for model_name, (leaf_path, summaries) in model_dict.items():
                if model_filter is not None and not model_filter(
                    model_name, get_model_family(model_name)
                ):
                    continue
                for s in summaries:
                    n_shots = s["n_shots"]
                    records = load_jsonl(
                        leaf_path / f"shots_{n_shots}" / "results.jsonl"
                    )
                    if not records:
                        continue
                    counts_raw: dict[str, dict[int, tuple[int, int]]] = {
                        k: {} for k in raw_keys
                    }
                    per_query_raw: dict[str, dict[str, dict[int, int]]] = {
                        k: {} for k in raw_keys
                    }
                    for i, r in enumerate(records):
                        d = r.get("demo_set_idx")
                        if d is None:
                            continue
                        is_correct = 1 if r.get("correct") else 0
                        align = r.get("alignment")
                        image_id = r.get("image_id") or f"_row{i}"
                        for k in raw_keys:
                            if k == "overall" or k == align:
                                c, t = counts_raw[k].get(d, (0, 0))
                                counts_raw[k][d] = (c + is_correct, t + 1)
                                per_query_raw[k].setdefault(image_id, {})[d] = is_correct
                    slot = out.setdefault(model_name, {}).setdefault(
                        n_shots, new_shot_aggregate()
                    )
                    for raw_k, by_demo in counts_raw.items():
                        if not by_demo:
                            continue
                        out_k = "overall" if raw_k == "overall" else align_to_key[raw_k]
                        dst = slot["counts"].setdefault(out_k, {})
                        for d, (c, t) in by_demo.items():
                            oc, ot = dst.get(d, (0, 0))
                            dst[d] = (oc + c, ot + t)
                        leaf_yerr = mean_query_std(per_query_raw[raw_k])
                        slot["leaf_yerrs"].setdefault(out_k, []).append(leaf_yerr)
    return out


def to_summaries_format_with_variance(
    aggregated: dict[str, dict[int, ShotAggregate]],
) -> dict[str, list[dict[str, Any]]]:
    """Convert ``{model: {n_shots: ShotAggregate}}`` to plot-ready summaries.

    Accuracy is computed by pooling counts across leaves; the yerr is the
    simple average of per-leaf (per-(task, split)) yerrs — matching the
    "average variance across task or splits" rule for cross-task / cross-split
    aggregations.
    """
    result: dict[str, list[dict[str, Any]]] = {}
    for model_name, shots_data in aggregated.items():
        summaries: list[dict[str, Any]] = []
        for n_shots in sorted(shots_data.keys()):
            slot = shots_data[n_shots]
            counts: VarianceCounts = slot.get("counts", {})
            leaf_yerrs: dict[str, list[float | None]] = slot.get("leaf_yerrs", {})
            accuracy: dict[str, float] = {}
            yerrs: dict[str, float | None] = {}
            per_seed: dict[str, dict[int, float]] = {}
            n_samples_total = 0
            for key, by_demo in counts.items():
                c_sum = sum(c for c, _ in by_demo.values())
                t_sum = sum(t for _, t in by_demo.values())
                accuracy[key] = (c_sum / t_sum) if t_sum > 0 else 0.0
                yerrs[key] = average_leaf_yerrs(leaf_yerrs.get(key, []))
                per_seed[key] = {
                    d: c / t for d, (c, t) in by_demo.items() if t > 0
                }
                if key == "overall":
                    n_samples_total = t_sum
            summaries.append({
                "n_shots": n_shots,
                "n_samples": n_samples_total,
                "accuracy": accuracy or {"overall": 0.0},
                "accuracy_yerr": yerrs,
                "accuracy_per_demo_set": per_seed,
            })
        result[model_name] = summaries
    return result


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a jsonl file, returning an empty list if the file does not exist."""
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pcorrect_per_sample(
    input_dir: Path,
    label_order: list[str],
    shot_counts: tuple[int, ...] = (0, 16),
    min_samples: int = 3,
) -> dict[int, dict[str, dict[str, dict[str, list[float]]]]] | None:
    """Per-sample P(correct) grouped by (n_shots, gt, alignment, image_id).

    Alignment key is ``"real"`` / ``"counterfactual"`` when ``expected_bias`` is
    present, else ``"all"``. Each ``image_id`` maps to the list of per-seed
    P(correct) values for that entry (length = number of demo sets at that shot
    count). Returns ``None`` only when every ``results.jsonl`` file is absent.
    Groups with fewer than ``min_samples`` total samples are dropped. Records
    missing ``image_id`` fall back to a synthetic per-row key, so each such
    sample becomes its own entry.
    """
    if not label_order:
        print("[load_pcorrect_per_sample] empty label_order → skipping")
        return None

    paths = [input_dir / f"shots_{n}" / "results.jsonl" for n in shot_counts]
    if not any(p.exists() for p in paths):
        return None
    with ProcessPoolExecutor(max_workers=len(shot_counts)) as ex:
        loaded = list(ex.map(load_jsonl, paths))

    result: dict[int, dict[str, dict[str, dict[str, list[float]]]]] = {}
    label_idx = {lbl: i for i, lbl in enumerate(label_order)}
    for n, path, records in zip(shot_counts, paths, loaded):
        if path.exists() and not records:
            print(f"[load_pcorrect_per_sample] {path} is empty")
        groups: dict[tuple[str, str], dict[str, list[float]]] = {}
        malformed = 0
        for i, r in enumerate(records):
            gt = r.get("ground_truth")
            dist = r.get("distribution")
            if gt is None or gt not in label_idx or dist is None or len(dist) != len(label_order):
                malformed += 1
                continue
            bias = r.get("expected_bias") or ""
            alignment = "all" if not bias else ("real" if bias == gt else "counterfactual")
            entry_id = r.get("image_id") or f"_row{i}"
            by_entry = groups.setdefault((gt, alignment), {})
            by_entry.setdefault(entry_id, []).append(float(dist[label_idx[gt]]))
        if malformed:
            print(f"[load_pcorrect_per_sample] {path}: skipped {malformed} malformed records")
        per_gt: dict[str, dict[str, dict[str, list[float]]]] = {}
        for (gt, alignment), by_entry in groups.items():
            total = sum(len(v) for v in by_entry.values())
            if total < min_samples:
                continue
            per_gt.setdefault(gt, {})[alignment] = by_entry
        result[n] = per_gt
    return result


def has_alignment_keys(raw: dict[int, dict[str, dict[str, dict[str, list[float]]]]]) -> bool:
    return any(
        align in ("real", "counterfactual")
        for by_gt in raw.values()
        for aligns in by_gt.values()
        for align in aligns
    )
