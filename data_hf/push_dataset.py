"""Push a local visual_reasoning dataset variant to its Hugging Face Hub mirror.

    python -m data_hf.push_dataset visual_reasoning_stresstest [--tasks counting ...] [--data_root DIR]

Each task becomes one config of the variant's repo (``vlm_icl.data.HUB_REPOS``), with one
split per partition file, named by ``vlm_icl.data.hub_split_name``; ``prepare_data`` with
``--data_source hub`` reads them back. Two fixes are applied on the way: images stored only
by path get their bytes embedded, and ``image_path`` values are cut down to the part after
``data_create/raw/``, so no local filesystem layout is published.

Requires an authenticated ``hf auth login``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import Dataset, DatasetDict, Image

from vlm_icl.data import HUB_REPOS, VR_DATASETS, dataset_dir, hub_split_name

RAW_MARKER = "data_create/raw/"


def _read_arrow(path: Path) -> pa.Table:
    with pa.memory_map(str(path), "r") as source:
        return ipc.open_file(source).read_all()


def _fix_table(table: pa.Table, stats: Dict[str, int]) -> pa.Table:
    new_images = []
    for image in table.column("image").to_pylist():
        if image.get("bytes") is None and image.get("path"):
            local_path = Path(image["path"])
            if local_path.exists():
                image = {"bytes": local_path.read_bytes(), "path": None}
                stats["bytes_embedded"] += 1
        new_images.append(image)
    table = table.set_column(
        table.schema.get_field_index("image"), "image",
        pa.array(new_images, type=table.schema.field("image").type),
    )

    if "image_path" in table.column_names:
        new_paths = []
        for image_path in table.column("image_path").to_pylist():
            if image_path and RAW_MARKER in image_path:
                image_path = image_path.split(RAW_MARKER, 1)[1]
                stats["paths_normalized"] += 1
            new_paths.append(image_path)
        table = table.set_column(
            table.schema.get_field_index("image_path"), "image_path", pa.array(new_paths, type=pa.string()),
        )
    return table


def _harmonize_features(splits: Dict[str, Dataset]) -> Dict[str, Dataset]:
    """Give every split of a config the same columns (``push_to_hub`` requires it); missing ones are null."""
    all_features = {}
    for dataset in splits.values():
        for name, feature in dataset.features.items():
            all_features.setdefault(name, feature)

    harmonized = {}
    for split, dataset in splits.items():
        for name, feature in all_features.items():
            if name not in dataset.features:
                dataset = dataset.add_column(name, [None] * len(dataset))
                dataset = dataset.cast_column(name, feature)
        harmonized[split] = dataset
    return harmonized


def build_config(variant: str, task: str, data_root: Optional[str] = None,
                 stats: Optional[Dict[str, int]] = None) -> DatasetDict:
    """Build the Hub config of one task: one split per partition file that exists locally."""
    stats = stats if stats is not None else {"bytes_embedded": 0, "paths_normalized": 0}
    partitions, _ = VR_DATASETS[variant]
    directory = dataset_dir(variant, data_root)
    splits = {}
    for partition_file, _ in partitions[task]:
        path = directory / partition_file
        if not path.exists():
            continue
        dataset = Dataset(_fix_table(_read_arrow(path), stats)).cast_column("image", Image())
        splits[hub_split_name(task, partition_file)] = dataset
        print(f"  {task}/{hub_split_name(task, partition_file)}: {len(dataset)} rows from {path.name}")
    return DatasetDict(_harmonize_features(splits))


def main() -> None:
    parser = argparse.ArgumentParser(description="Push a visual_reasoning variant to the Hub")
    parser.add_argument("variant", choices=sorted(HUB_REPOS))
    parser.add_argument("--tasks", nargs="+", default=None, help="Tasks to push (default: all)")
    parser.add_argument("--data_root", default=None, help="Directory holding the variant (default: data/)")
    args = parser.parse_args()

    repo_id = HUB_REPOS[args.variant]
    partitions, _ = VR_DATASETS[args.variant]
    for task in args.tasks or list(partitions):
        stats = {"bytes_embedded": 0, "paths_normalized": 0}
        ddict = build_config(args.variant, task, args.data_root, stats)
        if not ddict:
            print(f"{task}: no partition files, skipped")
            continue
        print(
            f"{task}: pushing {sum(len(d) for d in ddict.values())} rows "
            f"({stats['bytes_embedded']} bytes re-embedded, {stats['paths_normalized']} paths normalized)"
        )
        ddict.push_to_hub(repo_id, config_name=task, commit_message=f"Add {task} config")

    print(f"Done. https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
