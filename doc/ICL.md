# In-Context Learning

This page describes the full ICL pipeline: how demonstration examples are selected, balanced, transformed, and presented to the model.

## Overview

ICL provides the model with labeled demonstration examples (image + question + answer) before the actual query. The pipeline has three phases:

1. **Build demo pool** (run-level, once per seed) — group all dataset entries by `(task, split)` into context pools, then pre-select a balanced demo pool via `build_demo_pool()`
2. **Select from demo pool** (per query) — balanced sampling from the pre-built pool via `select_from_demo_pool()`, excluding the query; optional GT post-processing via `ensure_gt_demo()` / `reposition_gt_demo()`
3. **Process + format** — shuffle answers (optional, for `random_icd`), apply special-mode image transforms, resolve images, and render demos using ICL templates

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--icl` | `False` | Enable ICL mode |
| `--num_icl_examples` | `2` | Number of demonstration examples per query |
| `--num_icl_sample_sets` | `1` | Number of different ICL sample sets per query (re-samples context for each set) |
| `--query_gt_demo [POS]` | `None` | Guarantee one demo with query's GT label; optional position: `random` (default), `end`, `beginning` |

## Context Pool Construction

**Source:** `build_icl_context_pools()` in `vlm_icl/icl.py`

Context pools are dictionaries keyed by `(task, split)`. Each entry in the dataset is expanded into one pool item per question:

```python
pools[(task, split)] = [
    {"entry": <dataset_entry>, "question": <question_dict>},
    ...
]
```

**Key constraint:** Context examples always come from the same `(task, split)` as the query. There is no cross-split or cross-task sampling. This ensures demonstrations match the query's task type and originality split (e.g. `simple`, `noprior`, `complex` for visual_reasoning; `val`, `test`, `default` for BLINK).

### Full Dataset Pool When Using `--sample_per_split`

When `--sample_per_split` is used (stratified sampling), the query set is a subset of the full dataset. To avoid restricting ICL demos to only the sampled subset, the system loads the **full unsampled dataset** separately for the ICL context pool:

```python
if icl and data_kwargs.get("sample_per_split") is not None:
    icl_pool_data = prepare_data(dataset=dataset, **full_data_kwargs, data_root=data_root, data_source=data_source)  # no sample_per_split
```

This ensures demonstration diversity is not artificially limited by the query sampling.

## Selection Algorithm

**Source:** `build_demo_pool()` and `select_from_demo_pool()` in `vlm_icl/icl.py`

All ICL sampling flows through a unified two-step mechanism: a run-level demo pool is built once, then per-query selections draw from that pool.

### Step 1: Build Demo Pool (`build_demo_pool()`)

For each `(task, split)` partition in the context pools:

1. **Group by ground truth** — partition candidates by `question["ground_truth"]` label.
2. **Compute pool size** — `pool_size = min(count_per_label) * num_unique_labels`. This guarantees a perfectly balanced pool.
3. **Sample** — if the partition is larger than `pool_size`, sample exactly `min_count` items from each GT group; otherwise use the entire partition.
4. **Shuffle** — the pool is shuffled and stored.

The pool is deterministic (seeded, default `42`) and independent of `num_icl_examples`, so the same pool serves all ICL counts in a run.

### Step 2: Select From Demo Pool (`select_from_demo_pool()`)

For each query:

1. **Lookup pool** — find `demo_pool[(query_task, query_split)]`. Return `None` if missing.
2. **Exclude query** — remove all candidates sharing the query's `image_id` (prevents self-demonstration).
3. **Insufficient check** — return `None` if `len(candidates) < num_examples`.
4. **Group by ground truth** — partition remaining candidates by GT label.
5. **Shuffle groups** — randomly shuffle the list of unique GT labels (for unbiased remainder allocation).
6. **Optional GT priority** — if `ensure_query_gt=True`, move the query's GT label to position 0, guaranteeing it receives at least one demo during allocation.
7. **Balanced allocation** — distribute `num_examples` across `n_unique` GT groups:
   - `base = num_examples // n_unique` (minimum per group)
   - `remainder = num_examples % n_unique` (extra demos go to the first `remainder` groups)
   - Each group contributes `min(base + (1 if i < remainder else 0), len(group))` demos
   - Within each group, demos are selected via `random.sample()`
8. **Deficit fallback** — if some groups were too small to fill their allocation, remaining demos are sampled from the candidate pool (excluding already-selected items).
9. **Final shuffle** — the selected demos are shuffled to avoid grouped-by-label ordering bias.
10. **Truncate** — return exactly `num_examples` demos.

### Post-Selection Helpers

After `select_from_demo_pool()` returns, callers can apply:

- **`ensure_gt_demo()`** — if no selected demo matches the query's GT, swap the last demo for a random GT-matching candidate from the pool.
- **`reposition_gt_demo()`** — move the first GT-matching demo to a specified position (`beginning`, `end`, or `random`/no-op).

### Seeding

- The RNG is seeded with the `seed` parameter (default `42`), ensuring reproducible context selection.
- For multiple ICL sample sets (`num_icl_sample_sets > 1`), set 0 uses `seed`, set `i` uses `seed + i`.
- A separate `answer_rng` (seeded with `seed + 1`) controls answer shuffling for the `random_icd` ablation.

## Answer Shuffling (Ablation)

**Source:** `shuffle_icl_answers()` and `corrupt_icl_answers()` in `vlm_icl/icl.py`

When `--special random_icd` is active, the ground-truth answers of the selected ICL demos are randomly shuffled (permuted) before being placed into the prompt. This tests whether the model relies on the correctness of ICL labels or only on the structural pattern.

The `corrupt_icd` modes (`corrupt_icd`, `corrupt_icd_beginning`, `corrupt_icd_middle`, `corrupt_icd_end`) instead replace the answers at the selected demo positions with random labels from the `(task, split)` label space (`corrupt_icl_answers()`). In all other modes, answers are returned in their original (correct) order.

## Image Processing for ICL Demos

**Source:** `process_icl_context_images()` in `vlm_icl/icl.py`

Each selected demo's image is processed through `apply_special_mode()`:

| `--special` mode | Effect on ICL demo images |
|---|---|
| *(none)* | Images used as-is |
| `remove_image` | No effect on demos (only affects query image) |
| `remove_image_icd` | Demo images removed (`None`) — text-only demos |
| `blank_image` | No effect on demos (only affects query image) |
| `blank_image_icd` | Demo images replaced with `data/visual_reasoning/blank.png` |
| `random_icd` | Images unchanged; only answers are shuffled |

For multi-image datasets (BLINK), extra images (`image2`, `image3`, `image4`) in demos receive the same special-mode treatment.

### Answer Formatting

ICL answers pass through `format_icl_answer()` (`vlm_icl/prompts.py`) which wraps answers in curly braces `{answer}` when the question prompt mentions "curly bracket(s)" or contains `{`/`}` characters. This ensures demo answers match the expected output format.

## ICL Prompt Templates

**Source:** `vlm_icl/prompts.py`

### Single-image datasets

**Demo template:**
```
Given the image above, answer the following question.
Question: {question}
Answer: {answer}
```

**Query template:**
```
Given the image above, answer the following question.
Question: {question}
```

### Multi-image datasets (BLINK)

**Demo template:**
```
Given the images above, answer the following question.
Question: {question}
Answer: {answer}
```

**Query template:**
```
Given the images above, answer the following question.
Question: {question}
```

## Message Construction

`vlm_icl.prompts.build_conversation()` builds the user turn as backend-neutral `("text", ...)` / `("image", ...)` parts, and each backend converts the parts to its own message format (`_hf_messages` in `vlm_icl/models/hf.py`, `openai_chat_messages` in `vlm_icl/models/base.py` for vLLM and GPT; Gemini and InternVL convert inline). The logical order is the same for all backends:

1. **System prompt** — Model-specific default (e.g. "You are a helpful assistant."), added by the backend
2. **CoT prompt** (optional) — Prepended if `--cot` is enabled. Uses ICL-aware variant when ICL is active:
   - Zero-shot: *"Let's think step by step. Give reasoning rationales before providing the final answer."*
   - ICL: *"Let's first think step by step and analyze the relationship between the given few-shot question-answer pairs. Give reasoning rationales before providing the final answer."*
3. **ICL demonstrations** — For each demo in order:
   - Demo image(s) (if not removed by special mode)
   - Demo text (question + answer, formatted via demo template)
4. **Query image(s)** (if not removed by special mode)
5. **Query text** (question only, formatted via query template)

All content is placed in a single `user` message (multi-turn is not used).

## Multiple ICL Sample Sets

When `--num_icl_sample_sets K` is set with `K > 1`, each query is evaluated `K` times, each with a **different** set of ICL demonstrations. All sample sets draw from the same pre-built demo pool; only the per-set seed varies:

- **Set 0:** Uses the base `seed` for `select_from_demo_pool()`
- **Set i (i > 0):** Uses `seed + i`, producing a different random sample from the same pool

This enables measuring sensitivity to the specific choice of demonstrations. Each set produces a separate result entry and, when attention visualization is enabled, separate output subdirectories (`set1/`, `set2/`, etc.).

## Query-GT Guarantee (`--query_gt_demo`)

When `num_icl_examples < num_unique_labels`, balanced sampling may not include any demo with the query's ground-truth label. The `--query_gt_demo` flag guarantees at least one demo shares the query's ground-truth. This is implemented via three mechanisms:

1. **Allocation-time priority** — `select_from_demo_pool()` accepts `ensure_query_gt` and `query_gt_position` parameters. When `ensure_query_gt` is `True`, the query's GT label is moved to position 0 in the balanced allocation, guaranteeing it receives at least one slot; `query_gt_position` then places that demo. The answer-distribution and attention-distribution analyses use this path.
2. **Post-selection swap** — `ensure_gt_demo()` checks the selected demos and, if none match the query's GT, swaps the last demo for a GT-matching candidate from the pool.
3. **Position placement** — `reposition_gt_demo()` moves the GT demo to a specified position in the final sequence. `experiments/extra/inference.py` (and so `experiments.extra.evaluate`) uses mechanisms 2 and 3.

The optional position argument controls where the query-GT demo appears:

| Position | Behavior |
|---|---|
| `random` (default when flag is used without value) | Query-GT demo is shuffled randomly among other demos |
| `end` | Query-GT demo is placed last (closest to the query) |
| `beginning` | Query-GT demo is placed first |

```bash
# Default position (random)
uv run python -m experiments.extra.evaluate --model_path "/path/to/model" --icl --query_gt_demo

# Query-GT demo placed last (closest to query)
uv run python -m experiments.extra.evaluate --model_path "/path/to/model" --icl --query_gt_demo end

# Query-GT demo placed first
uv run python -m experiments.extra.evaluate --model_path "/path/to/model" --icl --query_gt_demo beginning
```

This is useful for studying whether seeing the correct answer class among demos affects model performance, and whether its position (recency vs primacy) matters.

## Edge Cases and Failure Modes

- **Insufficient context:** If a `(task, split)` pool has fewer candidates than `num_icl_examples` (after excluding the query), `select_from_demo_pool()` returns `None` and the query is skipped with a warning.
- **Empty groups:** If a ground-truth group is smaller than its allocation, the deficit is filled from remaining candidates across all groups.
- **API models:** Gemini and GPT backends receive ICL context in the same logical format; image encoding differs (base64 for Gemini, URL/base64 for GPT).

## Code References

| Component | File | Function/Class |
|---|---|---|
| Pool construction | `vlm_icl/icl.py` | `build_icl_context_pools()` |
| Demo pool construction | `vlm_icl/icl.py` | `build_demo_pool()` |
| Demo pool selection | `vlm_icl/icl.py` | `select_from_demo_pool()` |
| GT guarantee (swap) | `vlm_icl/icl.py` | `ensure_gt_demo()` |
| GT repositioning | `vlm_icl/icl.py` | `reposition_gt_demo()` |
| Answer shuffling | `vlm_icl/icl.py` | `shuffle_icl_answers()` |
| Answer corruption | `vlm_icl/icl.py` | `corrupt_icl_answers()`, `build_control_demo_set()` |
| Image processing | `vlm_icl/icl.py` | `process_icl_context_images()` |
| Special mode application | `vlm_icl/icl.py` | `apply_special_mode()` |
| ICL templates | `vlm_icl/prompts.py` | `ICL_DEMO_TEMPLATE`, `ICL_QUERY_TEMPLATE`, `get_icl_templates()` |
| CoT prompts | `vlm_icl/prompts.py` | `get_cot_prompt()` |
| Answer formatting | `vlm_icl/prompts.py` | `format_icl_answer()` |
| Conversation building | `vlm_icl/prompts.py` | `build_conversation()` |
| Batch ICL inference | `experiments/extra/inference.py` | `_run_batch()`, `_icl_context_builder()` |
| HF message building | `vlm_icl/models/hf.py` | `_hf_messages()` |
| vLLM / GPT message building | `vlm_icl/models/base.py` | `openai_chat_messages()` |
