"""Output directory layout: run-config names, leaf names and analysis paths."""

from datetime import datetime
from pathlib import Path
from typing import List, Optional


def derive_run_config(special: Optional[str] = None, cot: bool = False) -> str:
    """Derive the run_config string from special mode and CoT flag."""
    parts = []
    if cot:
        parts.append("cot")
    if special:
        parts.append(f"special_{special}")
    return "_".join(parts) if parts else "default"


def build_analysis_path(
    base_dir: str,
    analysis_type: str,
    dataset: str,
    task_name: str,
    model_path: str,
    leaf: str,
) -> Path:
    """Build the analysis output directory path."""
    model_name = Path(model_path).name
    return Path(base_dir) / analysis_type / dataset / task_name / model_name / leaf


def leaf_parts_prefix(
    sample_per_split: Optional[int] = None,
    run_config: str = "default",
    enable_thinking: bool = False,
) -> List[str]:
    """Build the common prefix parts for an output leaf directory name."""
    parts = []
    if sample_per_split is not None:
        parts.append(f"sps{sample_per_split}")
    if enable_thinking:
        parts.append("thinking")
    if run_config != "default":
        parts.append(run_config)
    return parts


def build_output_path(
    base_dir: str,
    analysis_type: str,
    model_path: str,
    task_name: str,
    icl: bool = False,
    num_icl_examples: int = 0,
    run_config: str = "default",
    dataset: str = "visual_reasoning",
    query_gt_demo: Optional[str] = None,
    stable: bool = False,
    sample_per_split: Optional[int] = None,
    enable_thinking: bool = False,
) -> Path:
    """Build the full output directory path for an experiment run."""
    parts = leaf_parts_prefix(sample_per_split, run_config, enable_thinking)
    if icl:
        icl_part = f"icl{num_icl_examples}"
        if query_gt_demo is not None:
            icl_part += f"_qgt_{query_gt_demo}"
        if stable:
            parts.append(icl_part)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            parts.append(f"{icl_part}_{timestamp}")
    else:
        parts.append("standard")

    leaf = "_".join(parts)
    return build_analysis_path(base_dir, analysis_type, dataset, task_name, model_path, leaf)
