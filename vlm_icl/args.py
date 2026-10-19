"""Shared argument groups for CLI scripts."""

import argparse
from typing import Optional, Sequence

from .data import DATA_SOURCES

SPECIAL_MODES = [
    "remove_image", "remove_image_icd",
    "blank_image", "blank_image_icd",
    "random_icd",
    "corrupt_icd", "corrupt_icd_beginning", "corrupt_icd_middle", "corrupt_icd_end",
]


def add_model_args(
    parser: argparse.ArgumentParser,
    *,
    extra_backends: Optional[Sequence[str]] = None,
) -> None:
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the local model directory")
    backends = ["default", "vllm"]
    if extra_backends:
        backends.extend(extra_backends)
    parser.add_argument("--backend", type=str, default="default", choices=backends,
                        help="Inference backend (default: HF transformers, vllm: vLLM)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (vLLM backend only, default: 1)")
    parser.add_argument("--max_num_images", type=int, default=64,
                        help="Maximum images per prompt for vLLM (default: 64)")
    parser.add_argument("--max_model_len", type=int, default=32768,
                        help="Maximum context length for vLLM (default: 32768)")


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=str, default="visual_reasoning",
                        choices=["visual_reasoning", "visual_reasoning_many", "visual_reasoning_control", "blink", "visual_reasoning_stresstest", "visual_reasoning_lens"],
                        help="Dataset to use (default: visual_reasoning)")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Filter by tasks (or 'all')")
    parser.add_argument("--topics", type=str, nargs="+", default=None,
                        help="Filter by specific topics (optional)")
    parser.add_argument("--sample_per_split", type=int, default=None,
                        help="Sample N entries per (task, split) combination for stratified sampling")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Directory holding one subdirectory per dataset (default: the repo's data/)")
    parser.add_argument("--data_source", type=str, default="auto", choices=DATA_SOURCES,
                        help="auto (default): local files if the dataset directory exists, else the "
                             "Hugging Face Hub; local; hub")


def add_icl_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--icl", action="store_true",
                        help="Enable in-context learning mode")
    parser.add_argument("--num_icl_examples", type=int, default=2,
                        help="Number of ICL context examples (default: 2)")
    parser.add_argument("--num_icl_sample_sets", type=int, default=1,
                        help="Number of different ICL sample sets per query (default: 1)")
    parser.add_argument("--query_gt_demo", nargs='?', const='random', default=None,
                        choices=['random', 'end', 'beginning'],
                        help="Guarantee one ICL demo with query's GT. Optional position: random (default), end, beginning")


def add_run_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: None, i.e. non-deterministic)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save outputs")
    parser.add_argument("--k", type=int, default=1,
                        help="Number of samples per query for pass@k evaluation (default: 1)")
    parser.add_argument("--special", type=str, choices=SPECIAL_MODES, default=None,
                        help="Special ablation mode for image/answer manipulation")
    parser.add_argument("--cot", action="store_true",
                        help="Enable Chain-of-Thought prompting")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable model-level thinking/reasoning mode")


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num_eval_runs", nargs='?', const=10, type=int, default=1,
                        help="Number of evaluation runs with different seeds (absent: 1, flag without value: 10)")
