# Reproducing the figures and tables

Which experiment feeds which paper figure or table, and the command that draws it.

## Workflow

1. Run the experiments for the models, with `slurm/launch.sh <experiment> <model>` ([EXPERIMENTS.md](EXPERIMENTS.md)). Models are the keys of [../slurm/models.tsv](../slurm/models.tsv).
2. Run the figure commands from the repo root with `uv run python -m ...`.

Figure scripts read `--results_dir` (default `output`) and write to `output_vis/<name>/`. They draw whichever models are present in the results; colors, family order and display names are in [../figures/_plotting/models.py](../figures/_plotting/models.py). The lens experiments require a model with a `lens_index` in `slurm/models.tsv` ([LATENTLENS.md](LATENTLENS.md)).

## Figures and tables

| Output (`output_vis/...`) | Experiment | Figure command |
|---|---|---|
| `formal_icl_scale/` | `icl_scale` | `figures.fig_icl_scale` |
| `formal_icl_useful/` | `icl_scale` | `figures.fig_icl_useful` |
| `formal_vlm_biased/` | `icl_scale` | `figures.fig_vlm_biased` |
| `export_accuracy_table/` (`accuracy_table.csv`) | `icl_scale` | `figures.table_accuracy` |
| `formal_icl_stresstest/` | `icl_stresstest` (`experiments.icl_scale --preset stresstest`) | `figures.fig_icl_useful --preset stresstest` |
| `formal_icl_unlearn/` | `icl_unlearn` | `figures.fig_icl_unlearn` |
| `formal_icl_reasoning/` | `icl_reasoning`, compared against `icl_scale` | `figures.fig_icl_reasoning` |
| `formal_lens_visualize/` | `lens_visualize` | `figures.fig_lens_visualize --dataset visual_reasoning_lens` |
| `formal_lens_ablate_tokens/` | `lens_ablate_tokens` | `figures.fig_lens_ablate_tokens --dataset visual_reasoning_lens` |
| `formal_lens_divert/` | `lens_divert` | `figures.fig_lens_divert` |
| `formal_lens_rewrite/` | `lens_rewrite` | `figures.fig_lens_rewrite --dataset visual_reasoning_lens` |

Add `--results_dir <dir>` when the results are not in `output/`. The `lens_*` experiments and `fig_lens_visualize` run in the `lens` environment; everything else runs in `eval`.

The `icl_scale` results are the input of four outputs and the baseline of `fig_icl_reasoning` (`<results_dir>/formal_icl_scale/visual_reasoning`).
