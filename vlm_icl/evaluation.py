"""Loading, saving and displaying evaluation results.

Metric computation lives in ``metrics.py``; ``experiments/extra/evaluate.py`` is the evaluation entry point.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_results(input_path: Path) -> List[Dict[str, Any]]:
    """Load results from JSON/JSONL file or directory (recursively finds results.json)."""
    results = []

    if input_path.is_file():
        with open(input_path, 'r') as f:
            if input_path.suffix == '.jsonl':
                for line in f:
                    results.append(json.loads(line))
            else:
                results.append(json.load(f))
    elif input_path.is_dir():
        for json_file in input_path.rglob('results.json'):
            try:
                with open(json_file, 'r') as f:
                    results.append(json.load(f))
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON file: {json_file}")
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    return results


def save_results(
    results: Union[List[Dict[str, Any]], Any],
    output_path: Union[str, Path],
    model_path: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
    *,
    format: str = "jsonl",
) -> None:
    """Save results and metadata to disk in the specified format."""
    from tqdm import tqdm

    output_path = Path(output_path)

    metadata = {
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    if format == "jsonl":
        output_path.mkdir(parents=True, exist_ok=True)

        results_path = output_path / "results.jsonl"
        with open(results_path, "w") as f:
            for result in tqdm(results, desc="Saving results", unit="sample"):
                f.write(json.dumps(result) + "\n")
        print(f"Saved {len(results)} results to: {results_path}")

        metadata["total_results"] = len(results)
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    elif format == "json":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "metadata": metadata,
            "results": results,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

    else:
        raise ValueError(f"Unknown format: {format!r}. Expected 'jsonl' or 'json'.")


def save_metrics(metrics: Dict[str, Any], output_path: Path, extra_metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save metrics dict to JSON file at *output_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': metrics['overall']['total'],
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    output_data = {
        'metadata': metadata,
        'metrics': metrics,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nMetrics saved to: {output_path}")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_results(metrics: Dict[str, Any]) -> None:
    """Pretty print evaluation metrics including pass@k breakdowns."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    num_samples = metrics.get('num_samples', 1)
    k = metrics.get('k', 4)
    overall = metrics['overall']

    print(f"\nOverall Metrics (N={overall['total']}, samples/query={num_samples}):")
    print(f"  Accuracy:    {overall['accuracy']*100:6.2f}%")
    if num_samples > 1:
        print(f"  Pass@1:      {overall['pass@1']*100:6.2f}%")
        print(f"  Pass@{k}:      {overall[f'pass@{k}']*100:6.2f}%")
    print(f"  Bias Rate:   {overall['bias_rate']*100:6.2f}%")
    print(f"  Other Rate:  {overall['other_rate']*100:6.2f}%")

    # Breakdown by task
    if metrics.get("by_task"):
        print("\n" + "-" * 80)
        print("Breakdown by Task:")
        print("-" * 80)
        if num_samples > 1:
            print(f"{'Task':<20} {'N':>5}  {'Acc':>7}  {'P@1':>7}  {'P@'+str(k):>7}")
            for task, stats in metrics["by_task"].items():
                print(f"{task:<20} {stats['total']:>5}  "
                      f"{stats['accuracy']*100:>6.1f}%  "
                      f"{stats['pass@1']*100:>6.1f}%  "
                      f"{stats[f'pass@{k}']*100:>6.1f}%")
        else:
            print(f"{'Task':<20} {'N':>5}  {'Accuracy':>9}  {'Bias Rate':>10}")
            for task, stats in metrics["by_task"].items():
                print(f"{task:<20} {stats['total']:>5}  "
                      f"{stats['accuracy']*100:>8.2f}%  "
                      f"{stats['bias_rate']*100:>9.2f}%")

    # Breakdown by task -> split
    if metrics.get("by_task_split"):
        print("\n" + "-" * 80)
        print("Breakdown by Task -> Split:")
        print("-" * 80)
        if num_samples > 1:
            print(f"{'Task':<15} {'Split':<10} {'N':>5}  {'Acc':>7}  {'P@1':>7}  {'P@'+str(k):>7}  {'BiasOnBiased':>12}")
            for task, splits in metrics["by_task_split"].items():
                for split, stats in splits.items():
                    bias_on_biased = stats.get('bias_rate_on_biased')
                    bias_str = f"{bias_on_biased*100:>11.1f}%" if bias_on_biased is not None else "         N/A"
                    print(f"{task:<15} {split:<10} {stats['total']:>5}  "
                          f"{stats['accuracy']*100:>6.1f}%  "
                          f"{stats['pass@1']*100:>6.1f}%  "
                          f"{stats[f'pass@{k}']*100:>6.1f}%  {bias_str}")
        else:
            print(f"{'Task':<15} {'Split':<10} {'N':>5}  {'Accuracy':>9}  {'Bias Rate':>10}  {'BiasOnBiased':>12}")
            for task, splits in metrics["by_task_split"].items():
                for split, stats in splits.items():
                    bias_on_biased = stats.get('bias_rate_on_biased')
                    bias_str = f"{bias_on_biased*100:>11.1f}%" if bias_on_biased is not None else "         N/A"
                    print(f"{task:<15} {split:<10} {stats['total']:>5}  "
                          f"{stats['accuracy']*100:>8.2f}%  "
                          f"{stats['bias_rate']*100:>9.2f}%  {bias_str}")

    # Breakdown by task -> originality
    if metrics.get("by_task_perturb_originality"):
        print("\n" + "-" * 80)
        print("Breakdown by Task -> Originality (tasks with gt != expected_bias):")
        print("-" * 80)
        if num_samples > 1:
            print(f"{'Task':<15} {'Originality':<12} {'N':>5}  {'Acc':>7}  {'P@1':>7}  {'P@'+str(k):>7}")
            for task, origs in metrics["by_task_perturb_originality"].items():
                for orig, stats in origs.items():
                    print(f"{task:<15} {orig:<12} {stats['total']:>5}  "
                          f"{stats['accuracy']*100:>6.1f}%  "
                          f"{stats['pass@1']*100:>6.1f}%  "
                          f"{stats[f'pass@{k}']*100:>6.1f}%")
        else:
            print(f"{'Task':<15} {'Originality':<12} {'N':>5}  {'Accuracy':>9}  {'Bias Rate':>10}")
            for task, origs in metrics["by_task_perturb_originality"].items():
                for orig, stats in origs.items():
                    print(f"{task:<15} {orig:<12} {stats['total']:>5}  "
                          f"{stats['accuracy']*100:>8.2f}%  "
                          f"{stats['bias_rate']*100:>9.2f}%")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
