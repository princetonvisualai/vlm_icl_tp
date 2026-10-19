"""Visual-reasoning datasets: partition tables, loading from local Arrow files or the Hub, routing."""

import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.ipc as ipc

GLOBAL_DATA_DIR = Path(__file__).parent.parent / "data"

# Where prepare_data reads from: "local" Arrow files under the data root, the "hub" mirror,
# or "auto" (local when the dataset directory exists, else the Hub unless a data root is given).
DATA_SOURCES = ("auto", "local", "hub")

TASK_PARTITIONS = {
    "direction": [
        ("direction_simple.arrow", "simple"),
        ("direction_simple_noprior.arrow", "noprior"),
        ("direction_complex.arrow", "complex"),
    ],
    "relation": [
        ("relation_simple.arrow", "simple"),
        ("relation_simple_noprior.arrow", "noprior"),
        ("relation_complex.arrow", "complex"),
    ],
    "counting": [
        ("counting_simple.arrow", "simple"),
        ("counting_simple_noprior.arrow", "noprior"),
        ("counting_complex.arrow", "complex"),
    ],
    "shape_identification": [
        ("shape_identification_simple.arrow", "simple"),
        ("shape_identification_simple_noprior.arrow", "noprior"),
        ("shape_identification_complex.arrow", "complex"),
    ],
    "color_identification": [
        ("color_identification_simple.arrow", "simple"),
        ("color_identification_simple_noprior.arrow", "simple_noprior"),
        ("color_identification_complex.arrow", "complex"),
    ],
}

SPLIT_ALIASES = {
    "control": "simple",
    "control_noprior": "noprior",
    "simple_noprior": "noprior",
}

TASK_PARTITIONS_MANY = {
    "counting": [("counting_control_many.arrow", "simple")],
    "direction": [("direction_control_many.arrow", "simple")],
    "relation": [("relation_control_many.arrow", "simple")],
}

TASK_PARTITIONS_CONTROL = {
    "color_identification": [
        ("color_identification_control.arrow", "control"),
        ("color_identification_control2.arrow", "control2"),
        ("color_identification_expand.arrow", "expand"),
    ],
    "shape_identification": [
        ("shape_identification_control.arrow", "control"),
        ("shape_identification_control2.arrow", "control2"),
        ("shape_identification_expand.arrow", "expand"),
    ],
}

TASK_PARTITIONS_LENS = {
    "counting": [("counting_complex.arrow", "complex")],
    "color_identification": [("color_identification_complex.arrow", "complex")],
    "shape_identification": [("shape_identification_complex.arrow", "complex")],
}

TASK_PARTITIONS_STRESSTEST = {
    "counting": [
        ("counting_simple_stresstest.arrow", "simple"),
        ("counting_simple_noprior_stresstest.arrow", "noprior"),
        ("counting_complex_stresstest.arrow", "complex"),
    ],
    "color_identification": [
        ("color_identification_complex_stresstest.arrow", "complex"),
    ],
    "shape_identification": [
        ("shape_identification_complex_stresstest.arrow", "complex"),
    ],
    "direction": [
        ("direction_complex_stresstest.arrow", "complex"),
    ],
    "relation": [
        ("relation_complex_stresstest.arrow", "complex"),
    ],
}

# dataset name -> (partitions, split aliases)
VR_DATASETS = {
    "visual_reasoning": (TASK_PARTITIONS, SPLIT_ALIASES),
    "visual_reasoning_stresstest": (TASK_PARTITIONS_STRESSTEST, SPLIT_ALIASES),
    "visual_reasoning_lens": (TASK_PARTITIONS_LENS, SPLIT_ALIASES),
    "visual_reasoning_control": (TASK_PARTITIONS_CONTROL, {}),
    "visual_reasoning_many": (TASK_PARTITIONS_MANY, SPLIT_ALIASES),
    "visual_reasoning_sample": (TASK_PARTITIONS, SPLIT_ALIASES),
}

# Hub mirrors (data_hf/push_dataset.py): one config per task, one split per partition file.
HUB_REPOS = {
    "visual_reasoning": "kunwang0129/visual-reasoning",
    "visual_reasoning_stresstest": "kunwang0129/visual-reasoning-stresstest",
    "visual_reasoning_lens": "kunwang0129/visual-reasoning-lens",
    "visual_reasoning_control": "kunwang0129/visual-reasoning-control",
}


def hub_split_name(task: str, partition_file: str) -> str:
    """Name of the Hub split holding a partition file: its stem without the task and stresstest affixes."""
    stem = Path(partition_file).stem.removesuffix("_stresstest")
    return stem[len(task) + 1:]


def dataset_dir(dataset: str, data_root: Optional[str] = None) -> Path:
    """Local directory of ``dataset``: ``<data_root>/<dataset>``, or ``data_root`` itself if it is that directory."""
    root = Path(data_root).expanduser() if data_root else GLOBAL_DATA_DIR
    if root.name == dataset and not (root / dataset).is_dir():
        return root
    return root / dataset


def resolve_data_source(dataset: str, data_root: Optional[str] = None, data_source: str = "auto") -> str:
    """Return ``"local"`` or ``"hub"`` for ``data_source`` (see DATA_SOURCES)."""
    if data_source not in DATA_SOURCES:
        raise ValueError(f"data_source must be one of {DATA_SOURCES}, got {data_source!r}")
    if data_source != "auto":
        return data_source
    if data_root is not None or dataset_dir(dataset).is_dir():
        return "local"
    return "hub"


def read_hub_split(repo_id: str, config: str, split: str) -> pa.Table:
    """Download one split of a Hub dataset config as an Arrow table (images stay ``{bytes, path}`` structs)."""
    from datasets import load_dataset

    return load_dataset(repo_id, config, split=split).with_format("arrow")[:]


def prepare_visual_reasoning(
    data_dir: Path = GLOBAL_DATA_DIR / "visual_reasoning",
    tasks: Optional[List[str]] = None,
    topics: Optional[List[str]] = None,
    verified_only: Optional[bool] = None,
    partitions: Optional[Dict] = None,
    split_aliases: Optional[Dict[str, str]] = None,
    hub_repo: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load a visual_reasoning dataset from its partitions with optional task/topic filtering.

    Partitions are read from Arrow files in ``data_dir``, or from the Hub dataset
    ``hub_repo`` when it is given.
    """
    if hub_repo is None and not data_dir.exists():
        raise FileNotFoundError(f"visual_reasoning directory not found at {data_dir}")

    task_partitions = partitions if partitions is not None else TASK_PARTITIONS
    aliases = split_aliases if split_aliases is not None else SPLIT_ALIASES
    topic_set = set(topics) if topics else None

    # Determine which tasks to load
    # tasks=None or tasks=["all"] loads all partitions
    if tasks is None or "all" in tasks:
        tasks_to_load = list(task_partitions.keys())
    else:
        tasks_to_load = tasks

    prepared: List[Dict[str, Any]] = []

    for task in tasks_to_load:
        if task not in task_partitions:
            warnings.warn(f"Unknown task '{task}', skipping")
            continue

        for partition_file, split in task_partitions[task]:
            if hub_repo is not None:
                table = read_hub_split(hub_repo, task, hub_split_name(task, partition_file))
            else:
                partition_path = data_dir / partition_file

                if not partition_path.exists():
                    warnings.warn(f"Partition file not found: {partition_path}, skipping")
                    continue

                reader = ipc.open_file(pa.memory_map(str(partition_path), "r"))
                table = reader.read_all()

            # Get column names to handle optional columns
            columns = set(table.column_names)

            for idx in range(table.num_rows):
                topic = table["topic"][idx].as_py()
                verified = table["verified"][idx].as_py() if "verified" in columns else True

                if topic_set is not None and topic not in topic_set:
                    continue
                if verified_only is not None and verified != verified_only:
                    continue

                entry = {
                    "image_id": table["ID"][idx].as_py(),
                    "image": table["image"][idx],
                    "task": task,
                    "topic": topic,
                    "split": split,
                    "pixels": table["pixels"][idx].as_py() if "pixels" in columns else None,
                    "verified": verified,
                    "image_path": table["image_path"][idx].as_py() if "image_path" in columns else None,
                    "object_name": table["object_name"][idx].as_py() if "object_name" in columns else None,
                    "ground_truth_caption": table["ground_truth_caption"][idx].as_py() if "ground_truth_caption" in columns else None,
                    "distract": table["distract"][idx].as_py() if "distract" in columns else None,
                    "questions": [
                        {
                            "type": "Q1",
                            "prompt": table["prompt"][idx].as_py(),
                            "ground_truth": table["ground_truth"][idx].as_py(),
                            "expected_bias": table["expected_bias"][idx].as_py(),
                            "metadata": {},
                        }
                    ],
                }

                # Include second image for multi-image tasks (e.g., BLINK)
                if "image2" in columns:
                    entry["image2"] = table["image2"][idx]

                entry["split"] = aliases.get(entry["split"], entry["split"])
                prepared.append(entry)

    return prepared


def prepare_data(
    dataset: str = "visual_reasoning",
    tasks: Optional[List[str]] = None,
    topics: Optional[List[str]] = None,
    verified_only: Optional[bool] = None,
    sample_per_split: Optional[int] = None,
    seed: Optional[int] = None,
    data_root: Optional[str] = None,
    data_source: str = "auto",
    **kwargs,
) -> List[Dict[str, Any]]:
    """Load a dataset with optional filtering and stratified sampling.

    ``data_root`` holds one directory per dataset (default: the repo's ``data/``);
    ``data_source`` is one of DATA_SOURCES.
    """
    source = resolve_data_source(dataset, data_root, data_source)
    if dataset == "blink":
        from .blink import prepare_blink
        data = prepare_blink(
            data_dir=dataset_dir(dataset, data_root), tasks=tasks, from_hub=source == "hub", **kwargs,
        )
    elif dataset in VR_DATASETS:
        partitions, split_aliases = VR_DATASETS[dataset]
        hub_repo = None
        if source == "hub":
            if dataset not in HUB_REPOS:
                raise ValueError(
                    f"{dataset} has no Hub mirror; place it under {dataset_dir(dataset, data_root)} "
                    "or pass --data_root."
                )
            hub_repo = HUB_REPOS[dataset]
        data = prepare_visual_reasoning(
            data_dir=dataset_dir(dataset, data_root),
            tasks=tasks, topics=topics, verified_only=verified_only,
            partitions=partitions, split_aliases=split_aliases, hub_repo=hub_repo, **kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset {dataset!r}; expected 'blink' or one of {sorted(VR_DATASETS)}")

    if sample_per_split is not None:
        # Group by (task, split) and sample N from each
        groups = defaultdict(list)
        for entry in data:
            groups[(entry['task'], entry['split'])].append(entry)

        rng = random.Random(seed)
        data = []
        for key in sorted(groups.keys()):
            entries = groups[key]
            n = min(sample_per_split, len(entries))
            data.extend(rng.sample(entries, n))

    return data
