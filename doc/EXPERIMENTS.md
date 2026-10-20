# Running experiments

How to run the paper experiments locally and on SLURM with `slurm/launch.sh`, and how to add a model.

## Experiments

Run from the repo root with `uv run python -m experiments.<name>`. Environments are described in [SETUP.md](SETUP.md).

| Module | What it does | Env |
|---|---|---|
| `experiments.icl_scale` | Accuracy against number of demos (0 to 32) on the `simple`, `noprior` and `complex` splits | eval |
| `experiments.icl_scale --preset stresstest` | The same on `visual_reasoning_stresstest`; writes under `formal_icl_stresstest` | eval |
| `experiments.icl_unlearn` | ICL with corrupted demo labels: does the output move toward uniform | eval |
| `experiments.icl_reasoning` | ICL scaling with thinking mode (generative, no logprobs) | eval |
| `experiments.lens_visualize` | Decode the text each query-image token encodes at each layer (LatentLens or Logit Lens) | lens |
| `experiments.lens_ablate_tokens` | Find distractor tokens with LatentLens, then compare baseline, ablated and 8-shot | lens |
| `experiments.lens_divert` | Attention diversion: per-entry ICL minus 0-shot attention density by token group | lens |
| `experiments.lens_rewrite` | Decoded tokens at distractor positions under baseline and 8-shot | lens |

The lens experiments use the HF backend and need a LatentLens index ([LATENTLENS.md](LATENTLENS.md)).
Further analyses are in [ANALYSES.md](ANALYSES.md).

## Single run

```bash
uv run python -m experiments.icl_scale --model_path /path/to/Qwen3-VL-8B-Instruct \
    --backend vllm --tasks counting --shot_counts 0 2 8 --n_per_label 20 --seed 42
```

Results go to `output/`; see [OUTPUT.md](OUTPUT.md).

## Common flags

Shared flags are defined in [../vlm_icl/args.py](../vlm_icl/args.py); `icl_scale`, `icl_unlearn` and `icl_reasoning` add the sampling flags.

| Flag | Default | Meaning |
|---|---|---|
| `--model_path` | required | Local checkpoint directory, or `Gemini` / `GPT` |
| `--backend {default,vllm}` | `default` | `default` is HF Transformers |
| `--tensor_parallel_size` | 1 | GPUs for vLLM |
| `--max_model_len`, `--max_num_images` | 32768, 64 | vLLM context length and images per prompt |
| `--dataset` | `visual_reasoning` | Dataset ([DATA.md](DATA.md)); `--preset stresstest` sets the stress-test dataset |
| `--tasks` | all five tasks | Tasks to run |
| `--data_root`, `--data_source {auto,local,hub}` | `data/`, `auto` | Where the data is read from ([DATA.md](DATA.md)) |
| `--output_dir` | `./output` | Result root |
| `--seed` | none (non-deterministic) | Random seed; the paper runs use 42 |
| `--shot_counts` | `0 1 2 4 8 16 32` | Demo counts (`icl_scale`, `icl_unlearn`, `icl_reasoning`) |
| `--splits` | all three | Splits to run (same three experiments) |
| `--n_per_label` | 40 | Queries per label (the paper runs use 20) |
| `--num_demo_sets` | 10 | Independent demo sets per query |
| `--enable_thinking`, `--cot` | off | Thinking mode, chain-of-thought prompt |

Demo selection is described in [ICL.md](ICL.md). The lens experiments take `--latentlens_index`, `--top_k`, `--n_per_label` and `--num_icl_examples` (not `--shot_counts`); `lens_visualize` takes `--icl_counts`.

## Models

`load_model` ([../vlm_icl/models/registry.py](../vlm_icl/models/registry.py)) picks the wrapper:

- `--model_path Gemini` or `GPT` (exact, case-insensitive) uses the API client (`GOOGLE_API_KEY` / `OPENAI_API_KEY`). These cannot be combined with `--backend vllm`.
- `--backend vllm` serves any local path with one family-agnostic wrapper.
- With the HF backend the first substring match in the path decides: `internvl`/`intern_vl`, `gemma`, `qwen2.5`/`qwen2_5`, `qwen3.5`/`qwen3_5`, then `qwen` (Qwen3-VL).

## SLURM

```bash
export VLM_ICL_MODELS=/path/to/models      # checkpoints and LatentLens indices
bash slurm/launch.sh --dry-run icl_scale qwen3vl-8b        # print the sbatch commands
bash slurm/launch.sh icl_scale qwen3vl-8b counting direction
bash slurm/launch.sh --help
```

Usage: `slurm/launch.sh [options] <experiment> <model> [tasks...] [-- extra args]`. `<model>` is a key of `slurm/models.tsv`. Tasks default to those of the experiment; arguments after `--` are appended to the module command line. Each job runs [../slurm/job.sbatch](../slurm/job.sbatch), which selects the env through `login_setup.sh`, sets `HF_HUB_OFFLINE=1`, and runs `uv run python -m <module>`.

Experiments: `icl_scale`, `icl_stresstest` (`icl_scale --preset stresstest`), `icl_unlearn`, `icl_reasoning`, `lens_visualize`, `lens_ablate_tokens`, `lens_divert`, `lens_rewrite`, and the extras listed in [ANALYSES.md](ANALYSES.md). The `icl_*` experiments run with vLLM and the resources from `models.tsv`; each submits one job per task, and per shot group when `shots` contains `;`. Lens experiments submit one job per task (`lens_visualize` also one per lens type) with the `gpu80` constraint.

Environment variables:

| Variable | Effect |
|---|---|
| `VLM_ICL_MODELS` | Required. Directory with checkpoints and LatentLens indices |
| `VLM_ICL_PARTITION` | Passed as `--partition` (not applied to lens jobs or to `--short` jobs with a `short_constraint`) |
| `VLM_ICL_MAIL` | Passed as `--mail-user`, with `--mail-type=END,FAIL` |
| `VLM_ICL_ROOT` | Repository root used by the worker (default: parent of `slurm/`) |

Options:

| Option | Effect |
|---|---|
| `--dry-run` | Print the `sbatch` commands |
| `--short` | 1-hour `icl_*` jobs using `shots_short`; models with `split_jobs=1` get one job per split |
| `--splits "a b"` | One job per listed split (`icl_*`) |
| `--time T`, `--mem M`, `--constraint C` | Override the resources of every job |
| `--lens-type {latentlens,logitlens}` | `lens_visualize` only (default: both) |
| `--distract-cue X`, `--expected-bias Y` | Cues for `lens_ablate_tokens` and `lens_rewrite` (defaults: `stop`/`octagon` for shape, `tiger`/`orange` for color) |
| `--prior-cue Z` | Cue for `lens_divert` |

### models.tsv

One tab-separated row per model; the first row is the header.

| Column | Meaning |
|---|---|
| `key` | Name passed to `launch.sh` |
| `path` | Checkpoint directory under `$VLM_ICL_MODELS` |
| `class` | `s` or `l`; selects the small or large time and memory of non-`icl_*` experiments |
| `gres` | `--gres` for the job |
| `tp` | `--tensor_parallel_size` for vLLM jobs |
| `mem` | `--mem` for `icl_*` jobs |
| `time` | Time limit for `icl_*` jobs |
| `time_reasoning` | Time limit for `icl_reasoning`; `-` falls back to `time` |
| `shots` | Shot counts for `icl_*` jobs. Groups separated by `;` become separate jobs (named `..._s<last shot>`), for example `0 1 2 4 8 16;32`. For extras the `;` is read as a space |
| `shots_short` | The same for `--short` |
| `split_jobs` | `1`: with `--short`, one job per split |
| `short_constraint` | `--constraint` for `--short` jobs; `-` for none. When set, the partition is not applied |
| `lens_index` | LatentLens index directory under `$VLM_ICL_MODELS`; `-` means the lens experiments are unavailable for this model |

To add a model, append a row with a new key and fill in all 13 columns (use `-` for unset values). The `path` must name a directory under `$VLM_ICL_MODELS` that matches the family rules above, or the HF backend cannot detect it. Add `lens_index` only if a LatentLens index exists for the model ([LATENTLENS.md](LATENTLENS.md)).
