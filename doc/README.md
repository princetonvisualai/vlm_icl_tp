# Documentation

Guides for installing, running and extending the code of
[*In-Context Learning Can Help Vision-Language Models Overcome Training Priors*](https://arxiv.org/abs/2610.12942)
(NeurIPS 2026).

## Getting started

| Guide | Contents |
|---|---|
| [SETUP.md](SETUP.md) | uv environments (`eval`, `lens`), API keys, model checkpoints, offline nodes, tests |
| [DATA.md](DATA.md) | datasets and Hub mirrors, `--data_root` / `--data_source`, file layout, schemas, loading |

## Running

| Guide | Contents |
|---|---|
| [EXPERIMENTS.md](EXPERIMENTS.md) | paper experiments, common flags, SLURM launcher, `slurm/models.tsv` |
| [REPRODUCE.md](REPRODUCE.md) | paper figures and tables: which experiment and figure command produces each |
| [ANALYSES.md](ANALYSES.md) | further analyses in `experiments/extra/`, `inference.py`, `evaluate.py`, `--special` ablations |
| [OUTPUT.md](OUTPUT.md) | output directory layout, leaf names and result files |

## Reference

| Guide | Contents |
|---|---|
| [ICL.md](ICL.md) | demo pools and selection, seeding, answer shuffling, prompt templates, query-GT demos |
| [LATENTLENS.md](LATENTLENS.md) | LatentLens indices, the lens experiments and the LatentLens API |
