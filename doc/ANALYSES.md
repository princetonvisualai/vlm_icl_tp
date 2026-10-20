# Further Analyses

Analyses in `experiments/extra/` that are not in the paper, and the `--special` ablation modes.

Run them from the repo root in the `eval` environment (see [SETUP.md](SETUP.md)) with `uv run python -m experiments.extra.<name>`. Demo selection is described in [ICL.md](ICL.md); output locations in [OUTPUT.md](OUTPUT.md).

## Scripts

| Script | What it does | Plot script (`figures/extra/`) | `slurm/launch.sh` entry |
|---|---|---|---|
| `inference` | Batch inference with optional ICL; `--image_attentionmap` saves attention heatmap overlays (Qwen3-VL, HF backend) | | `attention_map` |
| `evaluate` | `inference` plus metrics; `--num_eval_runs` repeats with different seeds and aggregates | | `evaluate` |
| `variance_ablation` | Wrapper around `evaluate` with `--num_runs` seeds, for accuracy variance across ICL demos | `variance_ablation` | `variance_ablation` |
| `answer_distribution` | Distribution over the answer labels at the answer position, per (task, split) partition | `answer_distribution` | `answer_distribution` |
| `answer_distribution_shift` | Top-k token distributions at the answer position, zero-shot against ICL | | `answer_distribution_shift` |
| `control_accuracy` | Accuracy with a fixed demo set, clean against corrupted answers, across k and runs | `control_accuracy` | `control_accuracy` |
| `control_output_dist` | Answer distributions for the same fixed clean and corrupted demo sets | `control_output_dist` | `control_output_dist` |
| `control_hidden_emb` | Layer-wise cosine divergence of hidden embeddings, clean against corrupted demos | `control_hidden_emb` | `control_hidden_emb` |
| `control_attention_distribution` | Section-level attention across k for clean and corrupted demos (HF) | | `control_attention_distribution` |
| `hidden_embedding` | Last-token hidden embeddings of every layer, zero-shot against ICL (HF) | `hidden_embedding` | `hidden_embedding` |
| `linear_probe` | Per-layer linear probe on frozen hidden embeddings, zero-shot against ICL (HF) | `linear_probe` | `linear_probe` |
| `visualize_attention_distribution` | Attention over the image and text sections of the prompt (HF) | | `attention_distribution` |
| `visualize_attention_token` | Token-level attention maps averaged over heads and examples, one PNG per layer (HF) | | none |

The `control_*` scripts take the dataset `visual_reasoning` and use one fixed demo set per (task, split) (`build_control_demo_set` in [`vlm_icl/icl.py`](../vlm_icl/icl.py)). `linear_probe` defaults to the dataset `visual_reasoning_many`, which is not released; pass `--dataset` to use another one. Launch entries are `bash slurm/launch.sh <entry> <model>`; see [EXPERIMENTS.md](EXPERIMENTS.md).

All scripts take `--model_path`, `--backend {default,vllm}` and the shared data, ICL and run-config flags from [`vlm_icl/args.py`](../vlm_icl/args.py) (`--dataset`, `--tasks`, `--sample_per_split`, `--icl`, `--num_icl_examples`, `--seed`, `--output_dir`, `--special`, `--cot`, `--enable_thinking`).

## Inference

Batch inference, zero-shot and with 4 demos; the first command also writes attention heatmaps (HF backend, Qwen3-VL only):

```bash
uv run python -m experiments.extra.inference --model_path /path/to/Qwen3-VL-8B-Instruct \
    --tasks counting --sample_per_split 10 --image_attentionmap
uv run python -m experiments.extra.inference --model_path /path/to/Qwen3-VL-8B-Instruct \
    --backend vllm --tasks counting --sample_per_split 10 --icl --num_icl_examples 4
```

`inference` prints the predictions and returns them; it writes files only for the heatmaps (analysis type `heat_maps`).

## Evaluation

Inference followed by metrics, saved per task:

```bash
uv run python -m experiments.extra.evaluate --model_path /path/to/Qwen3-VL-8B-Instruct \
    --backend vllm --tasks counting direction --sample_per_split 50 \
    --icl --num_icl_examples 8 --enable_thinking --cot --seed 42
```

`--num_eval_runs [N]` repeats the evaluation with different seeds (10 when given without a value) and saves the aggregated metrics with the per-run metrics embedded. `--k` sets the number of samples per query for pass@k, and `--test` runs 10 samples.

## Ablation modes

`--special <mode>` changes the inputs. It is used by `inference`, `evaluate`, `variance_ablation`, `answer_distribution`, `answer_distribution_shift`, `visualize_attention_distribution` and the ICL experiments `icl_scale`, `icl_unlearn` and `icl_reasoning`; the other scripts accept the flag and ignore it.

| Mode | Effect |
|---|---|
| `remove_image` / `remove_image_icd` | drop the query image / the demo images (text only) |
| `blank_image` / `blank_image_icd` | replace the query image / the demo images with `data/visual_reasoning/blank.png`, which you provide |
| `random_icd` | shuffle the answers among the demos |
| `corrupt_icd` | replace every demo answer with a random label of the task |
| `corrupt_icd_beginning` / `_middle` / `_end` | corrupt a quarter of the demo answers at that position (needs at least 4 demos) |

The `*_icd` and `corrupt_icd*` modes need `--icl`. The mode appears in the output leaf as `special_<mode>` ([OUTPUT.md](OUTPUT.md)).
