# Data

This page covers data formats, storage layout, data sources (local Arrow files and the Hugging Face Hub), loading logic, filtering, sampling, image resolution, and ICL context construction.

## Getting the data

Every experiment accepts `--data_root` and `--data_source {auto,local,hub}`.

| `--data_source` | Behavior |
|---|---|
| `local` | Read Arrow files from `<data_root>/<dataset>` (default root: `data/`) |
| `hub` | Read the Hub mirror of the dataset (`HUB_REPOS` in `vlm_icl/data.py`; BLINK: upstream repo) |
| `auto` (default) | `local` if `<data_root>/<dataset>` exists or `--data_root` is given, else `hub` |

- Without local files, any run with the default `auto` downloads the mirror from the Hub. SLURM jobs run with `HF_HUB_OFFLINE=1`, so place the data under `data/` or run a command with `--data_source hub` once on a node with network access before submitting.
- `python -m data_hf.convert_blink` converts BLINK parquet files in `data/raw/blink/` to the local Arrow layout (see [BLINK Conversion Pipeline](#blink-conversion-pipeline)).
- `python -m data_hf.push_dataset <variant>` publishes a local visual_reasoning variant to its Hub mirror (see [Hub mirrors](#hub-mirrors)).
- `blank.png` (`data/visual_reasoning/blank.png`), used by the `blank_image*` ablation modes, must be provided by you; it is always read from the repo's `data/`, not from `--data_root`.

## Supported Datasets

| Dataset | Tasks | Splits | Multi-image | Source module | Hub mirror |
|---|---|---|---|---|---|
| `visual_reasoning` | direction, relation, counting, shape_identification, color_identification | simple, noprior, complex | image | `vlm_icl/data.py` | [`kunwang0129/visual-reasoning`](https://huggingface.co/datasets/kunwang0129/visual-reasoning) |
| `visual_reasoning_stresstest` | counting, color_identification, shape_identification, direction, relation | simple, noprior, complex (counting); complex (color_identification, shape_identification, direction, relation) | image | `vlm_icl/data.py` | [`kunwang0129/visual-reasoning-stresstest`](https://huggingface.co/datasets/kunwang0129/visual-reasoning-stresstest) |
| `visual_reasoning_lens` | counting, color_identification, shape_identification | complex | image | `vlm_icl/data.py` | [`kunwang0129/visual-reasoning-lens`](https://huggingface.co/datasets/kunwang0129/visual-reasoning-lens) |
| `visual_reasoning_control` | color_identification, shape_identification | control, control2, expand | image | `vlm_icl/data.py` | [`kunwang0129/visual-reasoning-control`](https://huggingface.co/datasets/kunwang0129/visual-reasoning-control) |
| `visual_reasoning_many` | counting, direction, relation | simple | image | `vlm_icl/data.py` | local only |
| `visual_reasoning_sample` | same as `visual_reasoning` | simple, noprior, complex | image | `vlm_icl/data.py` | local only |
| `blink` | 14 tasks (see below) | val, test, default | up to 4 images | `vlm_icl/blink.py` | upstream [`BLINK-Benchmark/BLINK`](https://huggingface.co/datasets/BLINK-Benchmark/BLINK) |

**BLINK tasks (14):** art_style, counting, forensic_detection, functional_correspondence, iq_test, jigsaw, multi_view_reasoning, object_localization, relative_depth, relative_reflectance, semantic_correspondence, spatial_relation, visual_correspondence, visual_similarity

## File Layout on Disk

Local data lives under a data root (default `data/`, or `--data_root`) with one directory per dataset.

```
data/
├── visual_reasoning/
│   ├── {task}_{split}.arrow          # 15 partition files
│   └── blank.png                     # Used by blank_image ablation modes
├── visual_reasoning_stresstest/
│   └── {task}_{split}_stresstest.arrow  # 8 partition files (synthetic stress test images)
├── visual_reasoning_lens/
│   └── {task}_complex.arrow             # 3 partition files (subset of visual_reasoning complex splits)
├── visual_reasoning_control/
│   └── {task}_{control|control2|expand}.arrow  # 6 partition files (color_identification, shape_identification)
├── visual_reasoning_many/
│   └── {task}_control_many.arrow        # counting, direction, relation (local only)
├── visual_reasoning_sample/
│   └── {task}_{split}.arrow          # small per-label sample of visual_reasoning (local only)
├── blink/
│   └── {task}_{split}.arrow          # val, test, default per task (written by data_hf/convert_blink.py)
└── raw/
    └── blink/{TaskName}/             # Raw parquet files from HuggingFace
        ├── val-00000-of-00001.parquet
        └── test-00000-of-00001.parquet
```

### visual_reasoning Partition Files

| Task | Partition files |
|---|---|
| direction | `direction_simple.arrow`, `direction_noprior.arrow`, `direction_complex.arrow` |
| relation | `relation_simple.arrow`, `relation_noprior.arrow`, `relation_complex.arrow` |
| counting | `counting_simple.arrow`, `counting_noprior.arrow`, `counting_complex.arrow` |
| shape_identification | `shape_identification_simple.arrow`, `shape_identification_simple_noprior.arrow`, `shape_identification_complex.arrow` |
| color_identification | `color_identification_simple.arrow`, `color_identification_simple_noprior.arrow`, `color_identification_complex.arrow` |

### visual_reasoning_stresstest Partition Files

Synthetic stress test images designed to probe VLM counting and identification robustness with controlled distractors.

| Task | Partition files | Rows | Topics |
|---|---|---|---|
| counting | `counting_simple_stresstest.arrow`, `counting_simple_noprior_stresstest.arrow`, `counting_complex_stresstest.arrow` | 1000, 1000, 188 | Shapes (simple), Animals (complex) |
| color_identification | `color_identification_complex_stresstest.arrow` | 240 | Animals |
| shape_identification | `shape_identification_complex_stresstest.arrow` | 240 | Signs |
| direction | `direction_complex_stresstest.arrow` | 240 | Animals, Vehicles, Other |
| relation | `relation_complex_stresstest.arrow` | 240 | Animals, Appliances, Furniture, Kitchen, Kitchenware, Outdoor, Vehicles, Other |

### BLINK Partition Files

Each of the 14 tasks has up to 3 Arrow files: `{task}_val.arrow`, `{task}_test.arrow`, `{task}_default.arrow`. The `default` split is val + usable test rows combined (test rows with `answer == "hidden"` are excluded).

## Storage Format: Arrow IPC

All data is stored as **Apache Arrow IPC** (Feather v2) files, read via `pyarrow.ipc.open_file()` with memory-mapped I/O (`pa.memory_map()`). Images are embedded directly as binary blobs inside struct columns - no external image files are referenced at runtime.

### visual_reasoning Arrow Schema

| Column | Type | Description |
|---|---|---|
| `ID` | `string` | Unique image identifier |
| `image` | `struct{bytes: binary, path: string}` | Primary image (bytes embedded) |
| `image2` | `struct{bytes: binary, path: string}` | Second image (optional) |
| `topic` | `string` | Content topic (animals, food, scenes, objects, people, nature) |
| `prompt` | `string` | Question text |
| `ground_truth` | `string` | Correct answer |
| `expected_bias` | `string` | Expected biased answer (for bias analysis) |
| `pixels` | `int64` | Pixel count (width x height) |
| `verified` | `bool` | Whether the sample has been verified |
| `image_path` | `string` | Original image file path (optional) |
| `object_name` | `string` | Object name (optional) |
| `ground_truth_caption` | `string` | Ground-truth caption (optional) |

### visual_reasoning_stresstest Arrow Schema

Same schema as visual_reasoning with an additional `distract` column for distractor metadata.

| Column | Type | Description |
|---|---|---|
| `image` | `struct{bytes: binary, path: string}` | Synthetic image (bytes embedded) |
| `ID` | `string` | Unique identifier (e.g. `simple_stresstest_000000`) |
| `image_path` | `string` | Synthetic image path (e.g. `synthetic/simple_stresstest_000000.png`) |
| `task` | `string` | Task name (counting, color_identification, shape_identification, direction, relation) |
| `topic` | `string` | Content topic (Shapes, Animals, Signs, Vehicles, Appliances, Furniture, Kitchen, Kitchenware, Outdoor, Other) |
| `prompt` | `string` | Question text |
| `ground_truth` | `string` | Correct answer |
| `expected_bias` | `string` | Expected biased answer |
| `verified` | `bool` | Always `True` for synthetic data |
| `pixels` | `int64` | Pixel count (512 for simple, varies for complex) |
| `distract` | `string` | Distractor information (null for simple splits, animal/sign name for complex) |

**Distractor types by task:**
- **counting (complex):** Animal names with optional grid coordinates (e.g. `"zebra"`, `"giraffe 3 2 1152"`)
- **color_identification (complex):** Animal species (e.g. `"Flamingo"`, `"Tiger"`, `"Raven"`)
- **shape_identification (complex):** Sign types (e.g. `"STOP"`, `"YIELD"`, `"Speed limit"`)

### BLINK Arrow Schema

| Column | Type | Description |
|---|---|---|
| `ID` | `string` | Unique identifier (e.g. `counting_val_3`) |
| `prompt` | `string` | Question text (with appended curly-bracket instruction) |
| `ground_truth` | `string` | Correct answer letter (stripped of parentheses: `A`, `B`, etc.) |
| `pixels` | `int64` | Pixel count of primary image |
| `image` | `struct{bytes: binary, path: string}` | Primary image |
| `image2` | `struct{bytes: binary, path: string}` | Second image (null if absent) |
| `image3` | `struct{bytes: binary, path: string}` | Third image (null if absent) |
| `image4` | `struct{bytes: binary, path: string}` | Fourth image (null if absent) |

Multi-image columns use `{bytes: None, path: None}` when no image is present.

## Data Sources

See [Getting the data](#getting-the-data) for `--data_source`. `visual_reasoning_many` and `visual_reasoning_sample` have no Hub mirror and need local files.

### Hub mirrors

Each visual_reasoning variant in `HUB_REPOS` is one dataset repo with one config per task and one split per partition file. The split name is given by `hub_split_name(task, partition_file)`: the file stem without the task prefix and the `_stresstest` suffix (for example `counting_simple_noprior_stresstest.arrow` is split `simple_noprior` of config `counting`). Loaded rows have the same columns as the local Arrow files, so both sources feed the same loader.

`python -m data_hf.push_dataset <variant> [--tasks ...] [--data_root DIR]` publishes a local variant to its repo (requires `hf auth login`). It embeds image bytes for images stored by path and cuts `image_path` values down to the part after `data_create/raw/`.

BLINK is read from the upstream repo (one config per raw task name, splits `val` and `test`) and converted on the fly with the same row mapping as `data_hf/convert_blink.py`; `default` is val plus the usable test rows.

## Loading Pipeline

### Entry Point: `prepare_data()`

**Module:** `vlm_icl/data.py`

```python
prepare_data(
    dataset="visual_reasoning",  # or "visual_reasoning_stresstest", "visual_reasoning_lens", "blink", ...
    tasks=None,                  # list of task names, or None/["all"] for all
    topics=None,                 # list of topics (visual_reasoning variants only)
    verified_only=None,          # True/False/None filter (visual_reasoning variants only)
    sample_per_split=None,       # stratified sampling N per (task, split)
    seed=None,                   # RNG seed for sampling
    data_root=None,              # directory holding one folder per dataset (default: data/)
    data_source="auto",          # "auto", "local" or "hub"
)
```

This is the unified entry point. It resolves the data source, dispatches to the dataset-specific loader, then optionally applies stratified sampling.

**Example: Load stresstest counting data**
```python
data = prepare_data(dataset="visual_reasoning_stresstest", tasks=["counting"])
```

### Dataset-Specific Loaders

#### `prepare_visual_reasoning()`

1. Looks up the dataset's partition table in `VR_DATASETS` (`TASK_PARTITIONS` for `visual_reasoning`) to find partition files
2. Opens each `.arrow` file with `ipc.open_file(pa.memory_map(path, "r"))`, or reads the matching Hub split
3. Reads all rows into a PyArrow table
4. Filters rows by `topic` and `verified` if specified
5. Builds entry dicts row by row (images kept as `pyarrow.StructScalar`, decoded lazily)
6. Maps split names through `SPLIT_ALIASES` (`control` to `simple`, `control_noprior` and `simple_noprior` to `noprior`)

#### `prepare_blink()`

1. Iterates over `BLINK_TASK_PARTITIONS` dict
2. Default behavior loads only the `default` split (val + usable test combined); the `splits` argument selects others
3. Same memory-mapped Arrow IPC reading, or conversion of the upstream Hub splits
4. Checks multi-image columns via `_has_image()` helper (validates non-null bytes)
5. Sets `topic=None` and `expected_bias=None` (not applicable to BLINK)

### Returned Data Format

Both loaders return `List[Dict[str, Any]]` where each entry is:

```python
{
    "image_id": str,                    # unique identifier
    "image": pyarrow.StructScalar,      # primary image (decoded lazily)
    "task": str,                        # task name
    "topic": str | None,               # content topic (None for BLINK)
    "split": str,                       # originality split
    "pixels": int | None,              # pixel count
    "verified": bool,                   # verification status
    "questions": [
        {
            "type": "Q1",
            "prompt": str,              # question text
            "ground_truth": str,        # correct answer
            "expected_bias": str | None, # expected bias (None for BLINK)
            "metadata": {},
        }
    ],
    # Optional fields (visual_reasoning):
    "image_path": str | None,
    "object_name": str | None,
    "ground_truth_caption": str | None,
    "distract": str | None,
    # Optional multi-image fields (present when dataset has them):
    "image2": pyarrow.StructScalar,
    "image3": pyarrow.StructScalar,     # BLINK only
    "image4": pyarrow.StructScalar,     # BLINK only
}
```

**Key design choice:** Images are kept as `pyarrow.StructScalar` (zero-copy reference into the memory-mapped Arrow file) and only decoded to PIL at inference time via `resolve_image()`. This avoids loading all images into memory upfront.

## Filtering

### Task Filtering (`--tasks`)

Pass a list of task names. `None` or `["all"]` loads all tasks.

```bash
--tasks counting direction
```

### Topic Filtering (`--topics`, visual_reasoning only)

Filter by content topic. Available topics: animals, food, scenes, objects, people, nature.

```bash
--topics animals food
```

### Verified Filtering (`--verified_only`, visual_reasoning only)

Filter to only verified or unverified samples. `None` = no filter (default).

### Split Filtering (`--splits`, BLINK only)

Control which splits to load (available on the `icl_*` experiments). Default loads only `default` (combined val + usable test).

## Stratified Sampling (`--sample_per_split`)

When `sample_per_split=N` is specified, `prepare_data()` applies post-load stratified sampling:

1. Groups all entries by `(task, split)` tuple
2. Samples `min(N, group_size)` entries from each group
3. Uses deterministic RNG with configurable `seed` (default 42)

This ensures balanced representation across all task/split combinations. Used by variance ablation and quick test modes.

## Image Resolution: `resolve_image()`

**Module:** `vlm_icl/images.py`

Converts any image representation to a `PIL.Image` or path string for model consumption. Handles these input types:

| Input type | Resolution |
|---|---|
| `PIL.Image` | Returned as-is |
| `str` (file path or URL) | Returned as-is, or opened as PIL if `prefer_pil=True` |
| `bytes` | Decoded via `PIL.Image.open(BytesIO(b))` |
| `dict{"bytes", "path"}` | Tries bytes first, falls back to path |
| `pyarrow.StructScalar` | Extracts bytes/path from struct, same fallback chain |

All decoded images are converted to RGB mode (`.convert("RGB")`).

The `prefer_pil=True` flag forces local file paths to be opened as PIL images instead of passed as strings. This is used by API backends that need base64-encoded image data.

## BLINK Conversion Pipeline

**Module:** `data_hf/convert_blink.py` (row mapping in `vlm_icl/blink.py`: `convert_rows()`, `rows_to_table()`)

Converts raw BLINK parquet files (from HuggingFace) to Arrow IPC format. One-time operation; the `hub` data source applies the same row mapping on the fly.

```bash
uv run python -m data_hf.convert_blink
```

### Conversion steps per task:

1. Reads `data/raw/blink/{TaskName}/{split}-00000-of-00001.parquet`
2. Skips test rows with `answer == "hidden"` (no ground truth)
3. Strips parentheses from answers: `(A)` -> `A`
4. Appends curly-bracket instruction to prompts: `"Please answer with just the letter in curly brackets, e.g. {A}."`
5. Computes pixel count from primary image
6. Maps raw task names to snake_case (e.g. `Art_Style` -> `art_style`)
7. Writes per-split Arrow files (`{task}_val.arrow`, `{task}_test.arrow`)
8. Writes combined `{task}_default.arrow` (val + usable test)

### Task name mapping:

| Raw directory name | Arrow task name |
|---|---|
| Art_Style | art_style |
| Counting | counting |
| Forensic_Detection | forensic_detection |
| Functional_Correspondence | functional_correspondence |
| IQ_Test | iq_test |
| Jigsaw | jigsaw |
| Multi-view_Reasoning | multi_view_reasoning |
| Object_Localization | object_localization |
| Relative_Depth | relative_depth |
| Relative_Reflectance | relative_reflectance |
| Semantic_Correspondence | semantic_correspondence |
| Spatial_Relation | spatial_relation |
| Visual_Correspondence | visual_correspondence |
| Visual_Similarity | visual_similarity |

## ICL Context Construction

See [ICL.md](ICL.md) for the full ICL pipeline: context pool building (`build_icl_context_pools()`), demo pool construction (`build_demo_pool()`), per-query selection (`select_from_demo_pool()`), GT guarantee helpers, answer shuffling/corruption, and prompt templates.

### ICL Image Processing: `process_icl_context_images()`

Applies ablation transforms (remove/blank image) to ICL demo images and resolves them to PIL/path. Also handles multi-image fields (`image2`, `image3`, `image4`).

## Ablation Image Transforms

**Module:** `vlm_icl/icl.py`

The `apply_special_mode()` function modifies images for ablation studies:

| Mode | Query image | ICL demo images |
|---|---|---|
| `remove_image` | Removed (None) | Unchanged |
| `remove_image_icd` | Unchanged | Removed (None) |
| `blank_image` | Replaced with `blank.png` | Unchanged |
| `blank_image_icd` | Unchanged | Replaced with `blank.png` |
| `random_icd` | Unchanged | Unchanged (but labels shuffled) |

The blank image is located at `data/visual_reasoning/blank.png`.

## Code Reference

| File | Purpose |
|---|---|
| `vlm_icl/data.py` | Partition tables, `HUB_REPOS`, visual_reasoning loader, `prepare_data()` dispatcher, stratified sampling |
| `vlm_icl/blink.py` | BLINK loader (local Arrow or Hub), row conversion, multi-image handling |
| `data_hf/convert_blink.py` | Parquet-to-Arrow conversion for BLINK |
| `data_hf/push_dataset.py` | Publish a visual_reasoning variant to its Hub mirror |
| `data_hf/create_vr_sample.py` | Build `visual_reasoning_sample` from `visual_reasoning` |
| `vlm_icl/images.py` | `resolve_image()` |
| `vlm_icl/icl.py` | ICL pool building/selection, ablation transforms |
