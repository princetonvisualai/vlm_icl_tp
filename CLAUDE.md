# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research code for in-context learning (ICL) and LatentLens analysis of vision-language models
on the `visual_reasoning` benchmark (plus its `_stresstest`, `_lens` and `_control` variants)
and BLINK. `README.md` is the user-facing guide: setup, data, the experiment launcher and the
figure reproduce table.

**Models:** Qwen3-VL, Qwen3.5, Qwen2.5-VL, InternVL 3.5, Gemma 4 (HF Transformers or vLLM),
Gemini and GPT (API).
**Environment:** SLURM cluster; jobs run offline (`HF_HUB_OFFLINE=1`).

## Environment

Two uv environments, because vLLM and LatentLens need different `transformers` versions:

```bash
bash env_setup.sh eval      # .venv-eval: vLLM, API clients (+ dev group)
bash env_setup.sh lens      # .venv-lens: LatentLens (+ dev group)
source login_setup.sh eval  # select an env for `uv run` (or: lens)
uv run python -m experiments.icl_scale --model_path /path/to/model --backend vllm
uv run pytest               # no GPU needed
```

`pyproject.toml` is the only dependency source (groups `eval`, `lens`, `dev`; `eval` and
`lens` conflict). vLLM is pinned to a specific nightly build from its per-commit wheel index.

## Code Architecture

```
vlm_icl/                    library (never imports experiments/ or figures/)
├── data.py                 visual_reasoning loader, partition tables, dataset routing,
│                           --data_root / --data_source {auto,local,hub}, HUB_REPOS
├── blink.py                BLINK loader and upstream-row conversion
├── icl.py                  demo pools and selection, answer shuffle/corruption, special modes
├── prompts.py              ICL templates, CoT/system prompts, generation parameters,
│                           build_conversation() (backend-neutral content parts)
├── images.py               resolve_image, to_pil, pil_to_data_url
├── models/
│   ├── base.py             VLModel protocol, capabilities, openai_chat_messages adapter
│   ├── registry.py         detect_family, load_model ((family, backend) → wrapper)
│   ├── hf.py               Qwen3VLModel, Qwen25VLModel, Qwen35Model, Gemma4VLModel, InternVL35Model
│   ├── vllm.py             VLLMModel (family-agnostic)
│   └── api.py              GeminiVLModel, OpenAIVLModel
├── scoring.py              HFAnswerScorer / VLLMAnswerScorer (answer + label logprobs)
├── metrics.py, evaluation.py   answer normalization, metrics, result IO
├── paths.py                output directory grammar
├── attention.py            demo token spans, attention heatmaps
├── lens.py                 LatentLens helpers (latentlens imported lazily)
└── args.py, eb_distance.py shared CLI argument groups, embedding distance
experiments/                paper experiments: icl_scale (--preset stresstest), icl_unlearn,
│                           icl_reasoning, lens_visualize, lens_ablate_tokens, lens_divert, lens_rewrite
└── extra/                  inference, evaluate, answer_distribution(_shift), control_*,
                            hidden_embedding, linear_probe, variance_ablation, visualize_attention_*
figures/                    fig_*.py, table_accuracy.py (--results_dir, writes output_vis/<name>/)
├── extra/                  plots for experiments/extra
└── _plotting/              io.py, plots.py, style.py, models.py (model colors, order, names)
data_hf/                    convert_blink.py, create_vr_sample.py, push_dataset.py <variant>
slurm/                      launch.sh <experiment> <model> [tasks], job.sbatch, models.tsv
tests/                      pytest suite (generated Arrow fixture, no GPU, no network)
dev/                        release verification: snapshot.py, smoke run, figures_baseline.py
```

### Model dispatch

`load_model(model_path, backend)` in `vlm_icl/models/registry.py`:
- `Gemini` / `GPT` (exact, case-insensitive) → `GeminiVLModel` (`GOOGLE_API_KEY`) /
  `OpenAIVLModel` (`OPENAI_API_KEY`); not available with `--backend vllm`.
- `--backend vllm` → `VLLMModel` for any local path.
- HF backend, first matching substring: `internvl`/`intern_vl` → `InternVL35Model`, `gemma` →
  `Gemma4VLModel`, `qwen2.5`/`qwen2_5` → `Qwen25VLModel`, `qwen3.5`/`qwen3_5` → `Qwen35Model`,
  `qwen` → `Qwen3VLModel`.

Analysis hooks are declared in each wrapper's `capabilities` (`attentions`, `hidden_states`,
`isolation_mask`; only `Qwen3VLModel` has them). Check them with `require_capability`, not
`hasattr`. Backend quirks to keep: eager attention when attentions are requested, Gemma's
`<|think|>` system-prompt prefix on vLLM, HF-style vs OpenAI-style system messages, and
`inference(model=...)` reusing a loaded model.

### Datasets

| Dataset | Tasks |
|---|---|
| `visual_reasoning` | counting, shape_identification, color_identification, direction, relation |
| `visual_reasoning_stresstest` | the same five |
| `visual_reasoning_lens` | counting, shape_identification, color_identification |
| `visual_reasoning_control` | shape_identification, color_identification |
| `blink` | 14 tasks (art_style, counting, forensic_detection, ..., visual_similarity) |

Splits are `simple`, `noprior` and `complex` (control: `control`, `control2`, `expand`).
Details in `doc/DATA.md`; demo selection in `doc/ICL.md`.

### Output structure

`output/{analysis_type}/{dataset}/{task}/{model_dir_name}/{leaf}/`. `analysis_type` strings
keep their historical names (`formal_icl_scale`, `formal_icl_stresstest`,
`formal_lens_visualize`, `evaluation`, ...), so existing results stay readable. Paper
experiments use leaves like `npl20_complex`; the extras build the leaf with
`vlm_icl/paths.py` (`sps10_thinking_cot_icl2_<timestamp>`, `standard`, ...). Figures go to
`output_vis/<name>/`.

## Invariants

The release refactor kept results reproducible; keep these when changing code:
- RNG call order in demo selection, answer shuffling and sampling.
- Prompts, templates, generation parameters, output layout, leaf grammar, file names and the
  `metrics.json` schema.
- `.venv-lens/bin/python dev/snapshot.py --check <baseline>.json` records every engine input
  of every paper entry point on CPU with fake engines and must stay byte-identical to the
  baseline captured from the pre-release code (kept in the gitignored `dev/baselines/`).

## File Locations

- **Models:** `$VLM_ICL_MODELS` (checkpoints and LatentLens indices; keys in `slurm/models.tsv`)
- **Data:** `data/<dataset>/*.arrow` (gitignored; or `--data_source hub`)
- **Results:** `output/` (or `--output_dir`); **figures:** `output_vis/`
