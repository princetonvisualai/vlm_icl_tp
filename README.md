# vlm-icl

In-context learning (ICL) and LatentLens analysis of vision-language models on the
`visual_reasoning` benchmark and its stress-test, lens and control variants, plus BLINK.

Supported models: Qwen3-VL, Qwen3.5, Qwen2.5-VL, InternVL 3.5 and Gemma 4 (Hugging Face
Transformers or vLLM), and Gemini and GPT through their APIs.

## Layout

```
vlm_icl/            library: data loading, ICL demo selection, prompts, model wrappers,
                    answer scoring, metrics, output paths, attention and LatentLens helpers
experiments/        paper experiments (python -m experiments.<name>)
experiments/extra/  further analyses, inference.py and evaluate.py
figures/            paper figures (fig_*.py) and the accuracy table (table_accuracy.py)
figures/extra/      plots for the extra analyses
figures/_plotting/  shared loading, plotting, style and model registry
data_hf/            BLINK conversion and Hub publishing of the datasets
slurm/              job launcher (launch.sh), generic worker (job.sbatch), model table
tests/              unit tests (no GPU needed)
dev/                release-verification tools (prompt snapshot, GPU smoke run, figure check)
doc/                DATA.md, ICL.md, LATENTLENS.md
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12. vLLM and LatentLens need
different `transformers` versions, so there are two environments, built from the `eval` and
`lens` dependency groups of `pyproject.toml`:

```bash
UV_PROJECT_ENVIRONMENT=.venv-eval uv sync --no-default-groups --group eval --group dev
UV_PROJECT_ENVIRONMENT=.venv-lens uv sync --no-default-groups --group lens --group dev
```

| Environment | Used by |
|---|---|
| `eval` (vLLM, Gemini/GPT clients) | `experiments/icl_*.py`, `experiments/extra/*`, `figures/` |
| `lens` (LatentLens) | `experiments/lens_*.py`, `figures/fig_lens_visualize.py` |

To use an environment, set `UV_PROJECT_ENVIRONMENT=.venv-eval` (or `.venv-lens`) and
`UV_NO_SYNC=1`, then run commands from the repo root with `uv run python -m ...`. Without
`UV_NO_SYNC`, `uv run` re-syncs the environment to the default groups alone, which resolve some
packages (numpy) differently from the `eval` group. `env_setup.sh <eval|lens>` and
`source login_setup.sh <eval|lens>` do this on a module-based SLURM cluster; adapt their
`module` and `conda` lines to yours.

API models read `GOOGLE_API_KEY` (Gemini) and `OPENAI_API_KEY` (GPT) from the environment.

## Data

| Dataset | Tasks | Hub mirror |
|---|---|---|
| `visual_reasoning` | counting, shape_identification, color_identification, direction, relation | [`kunwang0129/visual-reasoning`](https://huggingface.co/datasets/kunwang0129/visual-reasoning) |
| `visual_reasoning_stresstest` | the same five | `kunwang0129/visual-reasoning-stresstest` |
| `visual_reasoning_lens` | counting, shape_identification, color_identification | `kunwang0129/visual-reasoning-lens` |
| `visual_reasoning_control` | shape_identification, color_identification | `kunwang0129/visual-reasoning-control` |
| `blink` | 14 BLINK tasks | upstream [`BLINK-Benchmark/BLINK`](https://huggingface.co/datasets/BLINK-Benchmark/BLINK) |

Every experiment takes `--data_root` and `--data_source {auto,local,hub}`:

- `auto` (default) reads `<data_root>/<dataset>/` when it exists (default root: `data/`)
  and downloads the Hub mirror otherwise.
- `local` and `hub` force one source.

The SLURM worker runs with `HF_HUB_OFFLINE=1`, so place the data under `data/` or download it
once on a node with network access (any command with `--data_source hub`) before submitting.
`python -m data_hf.convert_blink` converts BLINK parquet files in `data/raw/blink/` to the
local Arrow layout, and `python -m data_hf.push_dataset <variant>` publishes a local variant to
its Hub mirror. File layout, schemas and splits are in [doc/DATA.md](doc/DATA.md).

## Running experiments

One run:

```bash
uv run python -m experiments.icl_scale --model_path /path/to/Qwen3-VL-8B-Instruct \
    --backend vllm --tasks counting --shot_counts 0 2 8 --n_per_label 20 --seed 42
```

The paper runs go through `slurm/launch.sh`, which submits one job per task (and per split or
shot group for the large models) with the arguments and resources the paper used:

```bash
export VLM_ICL_MODELS=/path/to/models      # model checkpoints and LatentLens indices
bash slurm/launch.sh --dry-run icl_scale qwen3vl-8b        # print the sbatch commands
bash slurm/launch.sh icl_scale qwen3vl-8b counting direction
```

Models are the keys of `slurm/models.tsv` (checkpoint directory under `$VLM_ICL_MODELS`, GPUs,
time, tensor parallelism, shot counts and LatentLens index). `VLM_ICL_PARTITION` and
`VLM_ICL_MAIL` set the partition and the mail address; `bash slurm/launch.sh --help` lists
the options. Results go to `output/` (see [Output layout](#output-layout)).

## Reproducing the figures and tables

Run each experiment for the models in the paper, then the figure command from the repo root.
Figure scripts read `--results_dir` (default `output`) and write to `output_vis/<name>/`.

| Output (`output_vis/...`) | Experiment (`slurm/launch.sh <experiment> <model>`) | Figure command (`uv run python -m ...`) |
|---|---|---|
| `formal_icl_scale/` | `icl_scale` | `figures.fig_icl_scale` |
| `formal_icl_useful/` | `icl_scale` | `figures.fig_icl_useful` |
| `formal_vlm_biased/` | `icl_scale` | `figures.fig_vlm_biased` |
| `export_accuracy_table/` | `icl_scale` | `figures.table_accuracy` |
| `formal_icl_stresstest/` | `icl_stresstest` (`experiments.icl_scale --preset stresstest`) | `figures.fig_icl_useful --preset stresstest` |
| `formal_icl_unlearn/` | `icl_unlearn` | `figures.fig_icl_unlearn` |
| `formal_icl_reasoning/` | `icl_reasoning`, compared against `icl_scale` | `figures.fig_icl_reasoning` |
| `formal_lens_visualize/` | `lens_visualize` | `figures.fig_lens_visualize --dataset visual_reasoning_lens` (lens env) |
| `formal_lens_ablate_tokens/` | `lens_ablate_tokens` | `figures.fig_lens_ablate_tokens --dataset visual_reasoning_lens` |
| `formal_lens_divert/` | `lens_divert` | `figures.fig_lens_divert` |
| `formal_lens_rewrite/` | `lens_rewrite` | `figures.fig_lens_rewrite --dataset visual_reasoning_lens` |

The lens experiments need a model with a LatentLens index in `slurm/models.tsv`; see
[doc/LATENTLENS.md](doc/LATENTLENS.md).

## Further analyses

`experiments/extra/` holds analyses that are not in the paper, each with a plot script of
the same name in `figures/extra/` where one exists: answer distributions and their shift
under ICL, the fixed-demo-set control experiments (`control_*`), hidden-embedding divergence,
variance across seeds, attention distributions and maps, `inference.py` (batch inference,
optional attention heatmaps) and `evaluate.py` (inference plus metrics). `slurm/launch.sh`
has an entry for each of them except `visualize_attention_token`.

`experiments/extra/linear_probe.py` is not used in the paper, and its default dataset,
`visual_reasoning_many`, is not released.

### Ablation modes

`--special` applies an ablation to the inputs (`inference.py`, `evaluate.py`,
`variance_ablation`, the answer-distribution analyses and the ICL experiments):

| Mode | Effect |
|---|---|
| `remove_image` / `remove_image_icd` | drop the query image / the demo images (text only) |
| `blank_image` / `blank_image_icd` | replace the query image / the demo images with `data/visual_reasoning/blank.png`, which you provide |
| `random_icd` | shuffle the answers among the demos |
| `corrupt_icd` | replace every demo answer with a random label of the task |
| `corrupt_icd_beginning` / `_middle` / `_end` | corrupt a quarter of the demo answers at that position (needs at least 4 demos) |

Demo selection, templates and the query-GT options are described in [doc/ICL.md](doc/ICL.md).

## Output layout

Results go to `output/{analysis_type}/{dataset}/{task}/{model}/{leaf}/`, where `{model}` is
the checkpoint directory name (`vlm_icl/paths.py`). The paper experiments name the leaf after
their sampling and split (for example `npl20_complex`). `inference.py`, `evaluate.py` and the
other extras join, in order: `sps<N>` (`--sample_per_split`), `thinking`, the run config
unless it is `default` (`cot`, `special_<mode>` or `cot_special_<mode>`), and either
`icl<k>[_qgt_<position>]_<timestamp>` or `standard`:

| Options | Leaf |
|---|---|
| (none) | `standard` |
| `--icl --num_icl_examples 2` | `icl2_<timestamp>` |
| `--sample_per_split 10` | `sps10_standard` |
| `--special blank_image --icl --num_icl_examples 2` | `special_blank_image_icl2_<timestamp>` |
| `--enable_thinking` | `thinking_standard` |
| `--sample_per_split 10 --enable_thinking --cot --icl --num_icl_examples 2` | `sps10_thinking_cot_icl2_<timestamp>` |

## Tests

```bash
uv run pytest
```

The tests cover data loading (local and Hub paths, on a generated fixture), demo-selection
determinism, the output-path grammar, model-family detection and capabilities, and a scan for
private paths in tracked files.

## License

MIT; see [LICENSE](LICENSE).
