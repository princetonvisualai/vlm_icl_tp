"""Pure metric computation for VLM evaluation.

All functions in this module are side-effect-free (no IO, no display).
"""

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Answer normalization
# ---------------------------------------------------------------------------

def normalize_answer(answer: Any) -> str:
    """Normalize answer string; extracts last {…} token for CoT responses."""
    if answer is None:
        return ""
    answer = str(answer).strip()
    matches = re.findall(r'\{([^}]*)\}', answer)
    if matches:
        answer = matches[-1]
    return answer.strip().lower()


# ---------------------------------------------------------------------------
# Pass@k
# ---------------------------------------------------------------------------

def _comb(n: int, k: int) -> int:
    """Binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from *n* samples with *c* correct."""
    if n < k:
        return 1.0 if c > 0 else 0.0
    return 1.0 - _comb(n - c, k) / _comb(n, k)


# ---------------------------------------------------------------------------
# Sample-level evaluation
# ---------------------------------------------------------------------------

def evaluate_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Classify each answer in *sample* as correct / biased / other."""
    ground_truth = normalize_answer(sample["ground_truth"])
    expected_bias_raw = sample.get("expected_bias")
    expected_bias = normalize_answer(expected_bias_raw) if expected_bias_raw is not None else None

    answers = sample.get('model_answers', [sample.get('model_answer')])
    if not answers or answers[0] is None:
        return {'outcomes': ['other'], 'correct_count': 0, 'n': 1}

    outcomes = []
    for answer in answers:
        if answer is None:
            outcomes.append("other")
            continue
        model_answer = normalize_answer(answer)
        if model_answer == ground_truth:
            outcomes.append('correct')
        elif expected_bias is not None and model_answer == expected_bias:
            outcomes.append('biased')
        else:
            outcomes.append('other')

    return {
        'outcomes': outcomes,
        'correct_count': outcomes.count('correct'),
        'n': len(outcomes),
    }


# ---------------------------------------------------------------------------
# Stats helper (extracted from compute_metrics inner closure)
# ---------------------------------------------------------------------------

def _compute_stats(
    items: List[Dict],
    num_samples: int,
    k: int,
    compute_bias_on_biased: bool = False,
) -> Dict[str, Any]:
    """Aggregate evaluated items into accuracy / bias / pass@k stats."""
    total = len(items)
    if total == 0:
        return {'total': 0, 'correct': 0, 'accuracy': 0.0}

    all_outcomes = [o for item in items for o in item['outcomes']]
    correct = all_outcomes.count('correct')
    biased = all_outcomes.count('biased')
    other = all_outcomes.count('other')
    n_outcomes = len(all_outcomes)

    stats: Dict[str, Any] = {
        'total': total,
        'correct': correct,
        'accuracy': correct / n_outcomes if n_outcomes else 0.0,
        'biased': biased,
        'other': other,
        'bias_rate': biased / n_outcomes if n_outcomes else 0.0,
        'other_rate': other / n_outcomes if n_outcomes else 0.0,
    }

    if num_samples > 1:
        stats['pass@1'] = sum(pass_at_k(item['n'], item['correct_count'], 1) for item in items) / total
        stats[f'pass@{k}'] = sum(pass_at_k(item['n'], item['correct_count'], k) for item in items) / total

    if compute_bias_on_biased:
        biased_items = [item for item in items if item.get('is_biased', False)]
        biased_outcomes = [o for item in biased_items for o in item['outcomes']]
        stats['biased_entries'] = len(biased_items)
        stats['bias_rate_on_biased'] = biased_outcomes.count('biased') / len(biased_outcomes) if biased_outcomes else 0.0

    return stats


# ---------------------------------------------------------------------------
# Main metric computation
# ---------------------------------------------------------------------------

def compute_metrics(samples: List[Dict[str, Any]], k: int = 4) -> Dict[str, Any]:
    """Compute overall and breakdown metrics (accuracy, bias, pass@k) for *samples*."""
    evaluated = []
    for sample in samples:
        eval_result = evaluate_sample(sample)
        ground_truth_norm = normalize_answer(sample["ground_truth"])
        expected_bias_raw = sample["expected_bias"]
        expected_bias_norm = normalize_answer(expected_bias_raw) if expected_bias_raw is not None else None

        originality = "original" if ground_truth_norm == expected_bias_norm else "perturbed"

        split = sample.get("split")
        if split is None:
            task_str = str(sample.get("task", ""))
            if "_simple" in task_str or "_control" in task_str:
                split = "simple"
            elif "_perturbed" in task_str:
                split = "perturbed"
            else:
                split = "perturbed"

        evaluated.append({
            **eval_result,
            'task': sample["task"],
            'topic': sample["topic"],
            'question_type': sample.get('question_type', 'Q1'),
            'originality': originality,
            'split': split,
            'is_biased': expected_bias_norm is not None and ground_truth_norm != expected_bias_norm,
        })

    num_samples = evaluated[0]['n'] if evaluated else 1

    # Overall
    metrics: Dict[str, Any] = {
        'overall': _compute_stats(evaluated, num_samples, k),
        'num_samples': num_samples,
        'k': k,
    }

    # By task
    by_task: Dict[str, list] = defaultdict(list)
    for item in evaluated:
        by_task[item['task']].append(item)
    metrics['by_task'] = {t: _compute_stats(items, num_samples, k) for t, items in sorted(by_task.items())}

    # By task -> split
    by_task_split: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for item in evaluated:
        by_task_split[item['task']][item['split']].append(item)
    metrics['by_task_split'] = {
        task: {
            split: _compute_stats(items, num_samples, k, compute_bias_on_biased=any(i['is_biased'] for i in items))
            for split, items in sorted(splits.items())
        }
        for task, splits in sorted(by_task_split.items())
    }

    # By task -> originality (only tasks with perturbed entries)
    by_task_orig: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for item in evaluated:
        by_task_orig[item['task']][item['originality']].append(item)
    metrics['by_task_perturb_originality'] = {
        task: {orig: _compute_stats(origs[orig], num_samples, k) for orig in sorted(origs)}
        for task, origs in sorted(by_task_orig.items())
        if 'perturbed' in origs
    }

    # Empirical answer distribution (overall)
    answer_counts = Counter(
        normalize_answer(s["model_answers"][0]) for s in samples if s.get("model_answers")
    )
    total = sum(answer_counts.values())
    if total > 0:
        metrics['overall']['answer_distribution'] = {
            label: {'count': cnt, 'fraction': cnt / total}
            for label, cnt in sorted(answer_counts.items())
        }

    return metrics


# ---------------------------------------------------------------------------
# Multi-run aggregation
# ---------------------------------------------------------------------------

_RATE_FIELDS = ["accuracy", "bias_rate", "other_rate", "bias_rate_on_biased"]


def _augment_stats_block(
    stats: Dict[str, Any],
    values_map: Dict[str, List[float]],
) -> Dict[str, Any]:
    """Add mean/std/variance fields to *stats* for each rate field present."""
    augmented = dict(stats)
    for field in _RATE_FIELDS:
        if field not in augmented:
            continue
        values = values_map.get(field)
        if values is None:
            continue
        arr = np.array(values, dtype=float)
        if len(arr) == 1:
            augmented[f"{field}_mean"] = float(arr[0])
            augmented[f"{field}_std"] = 0.0
            augmented[f"{field}_variance"] = 0.0
        else:
            augmented[f"{field}_mean"] = float(np.mean(arr))
            augmented[f"{field}_std"] = float(np.std(arr, ddof=1))
            augmented[f"{field}_variance"] = float(np.var(arr, ddof=1))
        augmented[field] = augmented[f"{field}_mean"]
    return augmented


def _walk_granularity_levels(metrics: Dict[str, Any]):
    """Yield ``(path_key, stats_dict)`` for every granularity level."""
    if "overall" in metrics:
        yield ("overall",), metrics["overall"]
    for task, stats in metrics.get("by_task", {}).items():
        yield ("by_task", task), stats
    for task, splits in metrics.get("by_task_split", {}).items():
        for split, stats in splits.items():
            yield ("by_task_split", task, split), stats
    for task, origs in metrics.get("by_task_perturb_originality", {}).items():
        for orig, stats in origs.items():
            yield ("by_task_perturb_originality", task, orig), stats


def _set_nested(d: Dict, path: tuple, value: Any) -> None:
    """Set a value in a nested dict by *path* tuple."""
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value


def augment_metrics_with_zero_variance(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap single-run metrics with zero-variance fields for consistent schema."""
    result = json.loads(json.dumps(metrics))
    for path, stats in _walk_granularity_levels(result):
        values_map = {f: [stats[f]] for f in _RATE_FIELDS if f in stats}
        _set_nested(result, path, _augment_stats_block(stats, values_map))
    return result


def aggregate_multi_run_metrics(per_run_metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate N>1 run metrics into a single dict with mean/std/variance at every level."""
    template = json.loads(json.dumps(per_run_metrics_list[0]))

    for path, stats in list(_walk_granularity_levels(template)):
        values_map: Dict[str, List[float]] = {}
        for field in _RATE_FIELDS:
            values = []
            for run_metrics in per_run_metrics_list:
                node = run_metrics
                for key in path:
                    node = node.get(key, {})
                if field in node:
                    values.append(node[field])
            if values:
                values_map[field] = values
        _set_nested(template, path, _augment_stats_block(stats, values_map))

    return template


def cross_entropy(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """H(p, q) = -sum(p * log(q)). p is the target, q is the model distribution."""
    q_safe = np.clip(q, eps, None)
    return float(-np.sum(p * np.log(q_safe)))


def one_hot(label: str, label_order: List[str]) -> np.ndarray:
    v = np.zeros(len(label_order), dtype=np.float64)
    if label in label_order:
        v[label_order.index(label)] = 1.0
    return v
