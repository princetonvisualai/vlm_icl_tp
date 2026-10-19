"""Export accuracy + std table for ~4B and ~32B models across shot counts.

Writes a long-format CSV with one row per (model, n_shots, task, split) leaf
-- no aggregation across tasks or splits. The ``std`` column is the per-leaf
demo-set variance unit (``mean_query_std``: mean over queries of std-across-
the-10-demo-sets-for-that-query). It is a std on a 0/1 outcome, *not* a
standard error of the mean. Empty for 0-shot (variance only defined when
there are >=2 demo sets per query).

Usage:
    uv run python -m figures.table_accuracy \
        --results_dir output
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Sequence

from figures._plotting.io import (
    attach_variance_to_summaries,
    discover_model_summaries,
    extract_split_from_leaf,
)
from figures._plotting.models import (
    FAMILY_DISPLAY_NAMES,
    FAMILY_ORDER,
    get_model_family,
)


_TARGET_SPLITS: tuple[str, ...] = ("simple", "complex", "noprior")
_TARGET_SHOTS: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)


def _classify_size(model_name: str) -> str | None:
    """Return ``"4B"`` for ~4B variants (3-5B), ``"32B"`` for ~32B variants
    (25-40B), and ``None`` to drop the model from the table.
    """
    match = re.search(r"(\d+(?:\.\d+)?)[Bb](?![A-Za-z])", model_name)
    if not match:
        if "E4B" in model_name:
            return "4B"
        return None
    size = float(match.group(1))
    if 3.0 <= size <= 5.0:
        return "4B"
    if 25.0 <= size <= 40.0:
        return "32B"
    return None


def _family_sort_key(family: str) -> int:
    try:
        return FAMILY_ORDER.index(family)
    except ValueError:
        return len(FAMILY_ORDER)


def _size_sort_key(size_class: str) -> int:
    return {"4B": 0, "32B": 1}.get(size_class, 2)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Root dir containing {task}/{model}/{leaf}/shots_*/summary.json "
        "(default: <results_dir>/formal_icl_scale/visual_reasoning).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="output",
        help="Root of the experiment results.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="output_vis/export_accuracy_table/accuracy_table.csv",
        help="Path to write the CSV.",
    )
    args = parser.parse_args(argv)

    base_path = Path(args.base_dir or Path(args.results_dir) / "formal_icl_scale" / "visual_reasoning")
    cross_model = discover_model_summaries(base_path)
    # cross_model is {task: {leaf: {model: (leaf_path, summaries)}}}.

    rows: list[dict[str, str]] = []
    for task_name, leaf_dict in cross_model.items():
        for leaf, model_dict in leaf_dict.items():
            split = extract_split_from_leaf(leaf)
            if split not in _TARGET_SPLITS:
                continue
            for model_name, (leaf_path, summaries) in model_dict.items():
                size_class = _classify_size(model_name)
                if size_class is None:
                    continue
                attach_variance_to_summaries(leaf_path, summaries)
                family = get_model_family(model_name)
                family_display = FAMILY_DISPLAY_NAMES.get(family, family)
                for s in summaries:
                    n = s["n_shots"]
                    if n not in _TARGET_SHOTS:
                        continue
                    acc = (s.get("accuracy") or {}).get("overall")
                    yerr = (s.get("accuracy_yerr") or {}).get("overall")
                    n_samples = s.get("n_samples", 0)
                    rows.append({
                        "family": family_display,
                        "model": model_name,
                        "size_class": size_class,
                        "task": task_name,
                        "split": split,
                        "n_shots": str(n),
                        "accuracy": f"{acc:.6f}" if acc is not None else "",
                        "std": f"{yerr:.6f}" if yerr is not None else "",
                        "n_samples": str(n_samples),
                    })

    split_order = {s: i for i, s in enumerate(_TARGET_SPLITS)}
    rows.sort(key=lambda r: (
        _family_sort_key(get_model_family(r["model"])),
        _size_sort_key(r["size_class"]),
        r["model"],
        r["task"],
        split_order.get(r["split"], 99),
        int(r["n_shots"]),
    ))

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "family", "model", "size_class",
            "task", "split", "n_shots",
            "accuracy", "std", "n_samples",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
