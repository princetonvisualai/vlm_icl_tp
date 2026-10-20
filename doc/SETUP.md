# Setup

Installation, environments, credentials and the offline-cluster setup for running the code in this repository.

## Requirements

- [uv](https://docs.astral.sh/uv/) and Python 3.12 (`.python-version`).
- A CUDA GPU for model runs (vLLM and Hugging Face backends). The tests and the API models need none.

## Environments

vLLM and LatentLens need different `transformers` versions, so there are two environments, built from the `eval` and `lens` dependency groups of [`pyproject.toml`](../pyproject.toml). The groups conflict and cannot be installed together.

| Environment | Directory | Contents | Used by |
|---|---|---|---|
| `eval` | `.venv-eval` | vLLM, `transformers>=5.5.0`, Gemini and OpenAI clients | `experiments/icl_*.py`, `experiments/extra/*`, `figures/` |
| `lens` | `.venv-lens` | LatentLens, `transformers` | `experiments/lens_*.py`, `figures/fig_lens_visualize.py` |

Both also install the `dev` group (pytest). vLLM is pinned to one nightly build (`0.19.1rc1.dev341+g324a3d2bd`), fetched from its per-commit wheel index through `[tool.uv.sources]`, so installing `eval` needs access to `wheels.vllm.ai`.

### Building

```bash
UV_PROJECT_ENVIRONMENT=.venv-eval uv sync --no-default-groups --group eval --group dev
UV_PROJECT_ENVIRONMENT=.venv-lens uv sync --no-default-groups --group lens --group dev
```

[`env_setup.sh`](../env_setup.sh) runs the same command for one environment: `bash env_setup.sh eval` or `bash env_setup.sh lens`. It first runs `module purge`, `module load anaconda3/2025.6` and `conda activate uv` (a conda environment named `uv` that provides the `uv` command).

### Selecting

Set `UV_PROJECT_ENVIRONMENT` to the environment directory and `UV_NO_SYNC=1`, then run from the repo root:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-eval UV_NO_SYNC=1
uv run python -m experiments.icl_scale --help
```

Without `UV_NO_SYNC`, `uv run` re-syncs the environment to the default groups alone, which resolve some packages (numpy) differently from the `eval` group.

[`login_setup.sh`](../login_setup.sh) does this for a session: `source login_setup.sh eval` (or `lens`). It loads the same modules and conda environment as `env_setup.sh`, exports `UV_PROJECT_ENVIRONMENT` and `UV_NO_SYNC=1`, prepends `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` (conda's `libstdc++` must come before the system one for vLLM), changes to the repo root, and lists the GPUs. The SLURM worker sources it too. On another cluster, edit its `module` and `conda` lines (and those in `env_setup.sh`), or skip both scripts and set the two variables yourself.

## API keys

| Model | Variable |
|---|---|
| `Gemini` | `GOOGLE_API_KEY` |
| `GPT` | `OPENAI_API_KEY` |

API models are selected with `--model_path Gemini` or `--model_path GPT` and use the `eval` environment. They are not available with `--backend vllm`.

## Models

`VLM_ICL_MODELS` is the directory holding the model checkpoints and the LatentLens indices. [`slurm/launch.sh`](../slurm/launch.sh) requires it; the keys, checkpoint directory names and index names are in [`slurm/models.tsv`](../slurm/models.tsv). Direct runs take a checkpoint path through `--model_path`. See [EXPERIMENTS.md](EXPERIMENTS.md) and [LATENTLENS.md](LATENTLENS.md).

## Offline compute nodes

[`slurm/job.sbatch`](../slurm/job.sbatch) sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so jobs cannot download anything. Before submitting:

- Put the checkpoints and indices under `$VLM_ICL_MODELS`.
- Put the datasets under `data/`, or download them once on a node with network access by running any command with `--data_source hub`. See [DATA.md](DATA.md).

## Tests

```bash
uv run pytest
```

The tests need no GPU and no network. Data loading runs on a generated Arrow fixture, including the local and Hub paths. The suite covers:

- data loading: task, topic and verified filters, split aliases, stresstest partitions, stratified sampling, data-source resolution and Hub split names (`tests/test_data.py`);
- ICL demo pools and selection: determinism, balance, seed dependence, ground-truth guarantee, answer shuffling and corruption (`tests/test_icl.py`);
- output-path grammar and run-config derivation (`tests/test_paths.py`);
- model-family detection, wrapper capabilities and conversation building (`tests/test_capabilities.py`);
- a scan for private paths and e-mail addresses in tracked files, and the presence of the license and README (`tests/test_release.py`).
