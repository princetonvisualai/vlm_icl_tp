# Output Layout

Where each experiment writes its results and how the directory names are built.

Results go to

```
output/{analysis_type}/{dataset}/{task}/{model}/{leaf}/
```

`{model}` is the checkpoint directory name (the last component of `--model_path`). `--output_dir` replaces the root `output/`. Figures are written separately to `output_vis/<name>/` ([REPRODUCE.md](REPRODUCE.md)). The path code is [`vlm_icl/paths.py`](../vlm_icl/paths.py).

## Analysis types

The names are historical and unchanged, so existing results stay readable.

| `analysis_type` | Written by | Leaf |
|---|---|---|
| `formal_icl_scale` | `experiments.icl_scale` | `npl<n>_<split>` |
| `formal_icl_stresstest` | `experiments.icl_scale --preset stresstest` | `npl<n>_<split>` |
| `formal_icl_unlearn` | `experiments.icl_unlearn` | `npl<n>_<split>` |
| `formal_icl_reasoning` | `experiments.icl_reasoning` | `npl<n>_<split>` |
| `formal_lens_visualize` | `experiments.lens_visualize` | `[sps<n>_]npl<n>` or `npl<n>_<bias>`, plus `_logitlens` |
| `formal_lens_ablate_tokens` | `experiments.lens_ablate_tokens` | `npl<n>[_<bias>]` |
| `formal_lens_rewrite` | `experiments.lens_rewrite` | `npl<n>[_<bias>]` |
| `formal_lens_divert` | `experiments.lens_divert` | `npt<n>` |
| `evaluation` | `evaluate`, `variance_ablation` | extras grammar |
| `answer_dist` / `answer_shift` | `answer_distribution` / `answer_distribution_shift` | extras grammar |
| `attention_dist` | `visualize_attention_distribution` | extras grammar |
| `heat_maps` | `inference --image_attentionmap` | extras grammar |
| `control_accuracy`, `control_output_dist`, `control_hidden_emb` | the `control_*` scripts | `spl<n>_standard`; `control_output_dist` uses the extras prefix, e.g. `sps<n>_standard` |
| `hidden_embedding`, `linear_probe` | the scripts of the same name | `sps<n>_k<sets>` |
| `visualize_attention_token` | `visualize_attention_token` | `sps<n>_stride<s>/<split>` |
| `control_attention_dist` | `control_attention_distribution` | `{task}/{model}/{split}/` directly under `--output_dir` |

In the paper experiments `<n>` is `--n_per_label` (`npl`, `npt` for `lens_divert`), `<split>` is `simple`, `noprior` or `complex`, and `<bias>` is `--expected_bias`. Example: `output/formal_icl_scale/visual_reasoning/counting/Qwen3-VL-8B-Instruct/npl20_complex/`.

## Extras leaf grammar

`inference`, `evaluate` and the other extras that use `build_output_path` join these parts with `_`, in order:

1. `sps<N>` for `--sample_per_split`
2. `thinking` for `--enable_thinking`
3. the run config unless it is `default`: `cot`, `special_<mode>` or `cot_special_<mode>`
4. `icl<k>[_qgt_<position>]_<timestamp>` with `--icl` (`<timestamp>` is `YYYYMMDD_HHMMSS`), otherwise `standard`

`evaluate --num_eval_runs` and `variance_ablation` omit the timestamp (`icl<k>`), so repeated runs land in the same leaf.

| Options | Leaf |
|---|---|
| (none) | `standard` |
| `--icl --num_icl_examples 2` | `icl2_<timestamp>` |
| `--sample_per_split 10` | `sps10_standard` |
| `--special blank_image --icl --num_icl_examples 2` | `special_blank_image_icl2_<timestamp>` |
| `--enable_thinking` | `thinking_standard` |
| `--sample_per_split 10 --enable_thinking --cot --icl --num_icl_examples 2` | `sps10_thinking_cot_icl2_<timestamp>` |

[`tests/test_paths.py`](../tests/test_paths.py) pins this grammar.

## Files in a leaf

| Producer | Files |
|---|---|
| `icl_scale`, `icl_unlearn`, `icl_reasoning` | `run_config.json`; `shots_<n>/results.jsonl` and `shots_<n>/summary.json` per shot count |
| `evaluate` | `results.jsonl`, `metadata.json`, `metrics.json` (with `--num_eval_runs`: `metrics.json` only, with per-run metrics embedded) |
| `lens_visualize` | `formal_lens_visualize.json` |
| `lens_ablate_tokens`, `lens_rewrite` | `results.json`, `summary.json`, `run_config.json` |
| `lens_divert` | `summary.json`, `run_config.json`, `results.jsonl` |
| `answer_dist` | `run_config.json`; per split: `<split>/label_order.json`, `<split>/by_ground_truth.json`, `<split>/overall.json` |
| `answer_shift` | per split: `per_entry.jsonl`, `aggregated.json`, `label_token_mapping.json`, `fix_icl_demos.json`, `run_config.json` |
| `control_accuracy` | `run_config.json`, `control_demos.json`; `<condition>/k<k>/run<i>/accuracy.json`, `<condition>/k<k>/accuracy_summary.json` |
| `control_output_dist` | `run_config.json`, `control_demos.json`; `<condition>/k<k>/<split>/per_query.json` and the aggregated distributions |
| `control_hidden_emb`, `hidden_embedding`, `linear_probe` | `run_config.json`, `control_demos.json` (control only), `hidden_embedding.json` / `linear_probe.json` |
| `heat_maps`, `attention_dist` | `<split>/<image_id>/q<i>[/set<j>]/` with PNG heatmaps and `metadata.json` |

`metrics.json` holds `metadata` and `metrics` (`overall`, `by_task`, `by_task_split` and further breakdowns); `results.jsonl` has one record per query. Figure scripts read these files with `--results_dir` (default `output`).
