"""BLINK dataset loader (14 tasks): converted Arrow files, or the upstream Hub dataset."""

import warnings
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image

from .data import GLOBAL_DATA_DIR, read_hub_split

BLINK_HUB_REPO = "BLINK-Benchmark/BLINK"

# Upstream config name -> task name
BLINK_RAW_TASKS = {
    "Art_Style": "art_style",
    "Counting": "counting",
    "Forensic_Detection": "forensic_detection",
    "Functional_Correspondence": "functional_correspondence",
    "IQ_Test": "iq_test",
    "Jigsaw": "jigsaw",
    "Multi-view_Reasoning": "multi_view_reasoning",
    "Object_Localization": "object_localization",
    "Relative_Depth": "relative_depth",
    "Relative_Reflectance": "relative_reflectance",
    "Semantic_Correspondence": "semantic_correspondence",
    "Spatial_Relation": "spatial_relation",
    "Visual_Correspondence": "visual_correspondence",
    "Visual_Similarity": "visual_similarity",
}

IMAGE_STRUCT_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])

ARROW_SCHEMA = pa.schema([
    ("ID", pa.string()),
    ("prompt", pa.string()),
    ("ground_truth", pa.string()),
    ("pixels", pa.int64()),
    ("image", IMAGE_STRUCT_TYPE),
    ("image2", IMAGE_STRUCT_TYPE),
    ("image3", IMAGE_STRUCT_TYPE),
    ("image4", IMAGE_STRUCT_TYPE),
])

BLINK_TASK_PARTITIONS = {
    "art_style": [
        ("art_style_val.arrow", "val"),
        ("art_style_test.arrow", "test"),
        ("art_style_default.arrow", "default"),
    ],
    "counting": [
        ("counting_val.arrow", "val"),
        ("counting_test.arrow", "test"),
        ("counting_default.arrow", "default"),
    ],
    "forensic_detection": [
        ("forensic_detection_val.arrow", "val"),
        ("forensic_detection_test.arrow", "test"),
        ("forensic_detection_default.arrow", "default"),
    ],
    "functional_correspondence": [
        ("functional_correspondence_val.arrow", "val"),
        ("functional_correspondence_test.arrow", "test"),
        ("functional_correspondence_default.arrow", "default"),
    ],
    "iq_test": [
        ("iq_test_val.arrow", "val"),
        ("iq_test_test.arrow", "test"),
        ("iq_test_default.arrow", "default"),
    ],
    "jigsaw": [
        ("jigsaw_val.arrow", "val"),
        ("jigsaw_test.arrow", "test"),
        ("jigsaw_default.arrow", "default"),
    ],
    "multi_view_reasoning": [
        ("multi_view_reasoning_val.arrow", "val"),
        ("multi_view_reasoning_test.arrow", "test"),
        ("multi_view_reasoning_default.arrow", "default"),
    ],
    "object_localization": [
        ("object_localization_val.arrow", "val"),
        ("object_localization_test.arrow", "test"),
        ("object_localization_default.arrow", "default"),
    ],
    "relative_depth": [
        ("relative_depth_val.arrow", "val"),
        ("relative_depth_test.arrow", "test"),
        ("relative_depth_default.arrow", "default"),
    ],
    "relative_reflectance": [
        ("relative_reflectance_val.arrow", "val"),
        ("relative_reflectance_test.arrow", "test"),
        ("relative_reflectance_default.arrow", "default"),
    ],
    "semantic_correspondence": [
        ("semantic_correspondence_val.arrow", "val"),
        ("semantic_correspondence_test.arrow", "test"),
        ("semantic_correspondence_default.arrow", "default"),
    ],
    "spatial_relation": [
        ("spatial_relation_val.arrow", "val"),
        ("spatial_relation_test.arrow", "test"),
        ("spatial_relation_default.arrow", "default"),
    ],
    "visual_correspondence": [
        ("visual_correspondence_val.arrow", "val"),
        ("visual_correspondence_test.arrow", "test"),
        ("visual_correspondence_default.arrow", "default"),
    ],
    "visual_similarity": [
        ("visual_similarity_val.arrow", "val"),
        ("visual_similarity_test.arrow", "test"),
        ("visual_similarity_default.arrow", "default"),
    ],
}


def _has_image(table: pa.Table, col: str, idx: int) -> bool:
    """Check whether an image struct column has non-null bytes at the given row."""
    if col not in set(table.column_names):
        return False
    val = table[col][idx]
    try:
        b_scalar = val["bytes"]
    except Exception:
        return False
    return getattr(b_scalar, "is_valid", False) and b_scalar.as_py() is not None


def _strip_answer(raw_answer: str) -> str:
    """Strip parentheses from answer: '(A)' → 'A'."""
    s = raw_answer.strip()
    if s.startswith("(") and s.endswith(")"):
        return s[1:-1]
    return s


def _get_pixels(image_struct) -> int:
    """Get pixel count from an image struct dict."""
    if image_struct is None:
        return 0
    b = image_struct.get("bytes")
    if b is None:
        return 0
    try:
        img = Image.open(BytesIO(b))
        return img.width * img.height
    except Exception:
        return 0


def _make_prompt(raw_prompt: str) -> str:
    """Append curly-bracket instruction to the raw BLINK prompt."""
    return raw_prompt.rstrip() + "\nPlease answer with just the letter in curly brackets, e.g. {A}."


def _image_or_none(row, col: str):
    """Extract image struct from a parquet row, returning dict or None."""
    val = row.get(col)
    if val is None:
        return None
    if isinstance(val, dict):
        b = val.get("bytes")
        if b is None:
            return None
        return {"bytes": b, "path": val.get("path", "")}
    return None


def convert_rows(table: pa.Table, task: str, split: str) -> List[Dict[str, Any]]:
    """Convert one upstream BLINK split to row dicts in the Arrow schema.

    Test rows with hidden answers and rows without a first image are dropped.
    """
    rows = []
    for i in range(table.num_rows):
        row = {col: table[col][i].as_py() for col in table.column_names}

        answer = row.get("answer", "")
        if answer == "hidden":
            continue  # Skip test samples with hidden answers

        img1 = _image_or_none(row, "image_1")
        if img1 is None:
            continue  # Must have at least image_1

        rows.append({
            "ID": row.get("idx", f"{task}_{split}_{i}"),
            "prompt": _make_prompt(row.get("prompt", "")),
            "ground_truth": _strip_answer(answer),
            "pixels": _get_pixels(row.get("image_1")),
            "image": img1,
            "image2": _image_or_none(row, "image_2"),
            "image3": _image_or_none(row, "image_3"),
            "image4": _image_or_none(row, "image_4"),
        })

    return rows


def rows_to_table(rows: List[Dict[str, Any]]) -> pa.Table:
    """Build an Arrow table in ARROW_SCHEMA from converted rows."""
    def _img_val(r, key):
        v = r.get(key)
        if v is None:
            return {"bytes": None, "path": None}
        return v

    arrays = {
        "ID": pa.array([r["ID"] for r in rows], type=pa.string()),
        "prompt": pa.array([r["prompt"] for r in rows], type=pa.string()),
        "ground_truth": pa.array([r["ground_truth"] for r in rows], type=pa.string()),
        "pixels": pa.array([r["pixels"] for r in rows], type=pa.int64()),
        "image": pa.array([_img_val(r, "image") for r in rows], type=IMAGE_STRUCT_TYPE),
        "image2": pa.array([_img_val(r, "image2") for r in rows], type=IMAGE_STRUCT_TYPE),
        "image3": pa.array([_img_val(r, "image3") for r in rows], type=IMAGE_STRUCT_TYPE),
        "image4": pa.array([_img_val(r, "image4") for r in rows], type=IMAGE_STRUCT_TYPE),
    }
    return pa.table(arrays, schema=ARROW_SCHEMA)


def _read_hub_partition(task: str, split: str) -> pa.Table:
    """Convert a BLINK partition from the upstream Hub dataset; ``default`` is val + test."""
    raw_task = {v: k for k, v in BLINK_RAW_TASKS.items()}[task]
    rows = []
    for upstream_split in (("val", "test") if split == "default" else (split,)):
        rows.extend(convert_rows(read_hub_split(BLINK_HUB_REPO, raw_task, upstream_split), task, upstream_split))
    return rows_to_table(rows)


def prepare_blink(
    data_dir: Path = GLOBAL_DATA_DIR / "blink",
    tasks: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
    from_hub: bool = False,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Load BLINK with optional task/split filtering.

    Reads the Arrow files ``data_hf/convert_blink.py`` writes to ``data_dir``, or with
    ``from_hub`` converts the upstream Hub dataset the same way.
    """
    if not from_hub and not data_dir.exists():
        raise FileNotFoundError(
            f"BLINK data directory not found at {data_dir}. "
            "Run: uv run python -m data_hf.convert_blink, or pass --data_source hub"
        )

    # Default: load only the 'default' combined split
    if splits is None:
        split_set = {"default"}
    else:
        split_set = set(splits)

    if tasks is None or "all" in tasks:
        tasks_to_load = list(BLINK_TASK_PARTITIONS.keys())
    else:
        tasks_to_load = tasks

    prepared: List[Dict[str, Any]] = []

    for task in tasks_to_load:
        if task not in BLINK_TASK_PARTITIONS:
            warnings.warn(f"Unknown BLINK task '{task}', skipping")
            continue

        for partition_file, split in BLINK_TASK_PARTITIONS[task]:
            if split not in split_set:
                continue

            if from_hub:
                table = _read_hub_partition(task, split)
            else:
                partition_path = data_dir / partition_file
                if not partition_path.exists():
                    warnings.warn(f"Partition file not found: {partition_path}, skipping")
                    continue

                reader = ipc.open_file(pa.memory_map(str(partition_path), "r"))
                table = reader.read_all()
            columns = set(table.column_names)

            for idx in range(table.num_rows):
                entry: Dict[str, Any] = {
                    "image_id": table["ID"][idx].as_py(),
                    "image": table["image"][idx],
                    "task": task,
                    "topic": None,
                    "split": split,
                    "pixels": table["pixels"][idx].as_py() if "pixels" in columns else None,
                    "verified": True,
                    "questions": [
                        {
                            "type": "Q1",
                            "prompt": table["prompt"][idx].as_py(),
                            "ground_truth": table["ground_truth"][idx].as_py(),
                            "expected_bias": None,
                            "metadata": {},
                        }
                    ],
                }

                # Multi-image fields
                if _has_image(table, "image2", idx):
                    entry["image2"] = table["image2"][idx]
                if _has_image(table, "image3", idx):
                    entry["image3"] = table["image3"][idx]
                if _has_image(table, "image4", idx):
                    entry["image4"] = table["image4"][idx]

                prepared.append(entry)

    return prepared
