import re
from pathlib import Path

import pytest

from vlm_icl.paths import build_analysis_path, build_output_path, derive_run_config, leaf_parts_prefix

TS = r"\d{8}_\d{6}"
MODEL = "/models/Qwen3-VL-8B-Instruct"


@pytest.mark.parametrize("special,cot,expected", [
    (None, False, "default"),
    (None, True, "cot"),
    ("blank_image", False, "special_blank_image"),
    ("blank_image", True, "cot_special_blank_image"),
])
def test_derive_run_config(special, cot, expected):
    assert derive_run_config(special, cot) == expected


def leaf(**kwargs):
    path = build_output_path("out", "inference", MODEL, "counting", **kwargs)
    return path.name


@pytest.mark.parametrize("kwargs,pattern", [
    ({}, r"standard"),
    ({"icl": True, "num_icl_examples": 2}, rf"icl2_{TS}"),
    ({"sample_per_split": 10}, r"sps10_standard"),
    ({"icl": True, "num_icl_examples": 2, "run_config": "special_blank_image"}, rf"special_blank_image_icl2_{TS}"),
    ({"enable_thinking": True}, r"thinking_standard"),
    ({"enable_thinking": True, "icl": True, "num_icl_examples": 2}, rf"thinking_icl2_{TS}"),
    (
        {"sample_per_split": 10, "enable_thinking": True, "run_config": "cot", "icl": True, "num_icl_examples": 2},
        rf"sps10_thinking_cot_icl2_{TS}",
    ),
    ({"icl": True, "num_icl_examples": 8, "stable": True, "sample_per_split": 100}, r"sps100_icl8"),
    ({"icl": True, "num_icl_examples": 4, "query_gt_demo": "end", "stable": True}, r"icl4_qgt_end"),
])
def test_leaf_grammar(kwargs, pattern):
    assert re.fullmatch(pattern, leaf(**kwargs))


def test_full_path_layout():
    path = build_output_path("out", "evaluation", MODEL, "direction", dataset="visual_reasoning_lens")
    assert path == Path("out/evaluation/visual_reasoning_lens/direction/Qwen3-VL-8B-Instruct/standard")


def test_analysis_path_uses_model_basename():
    path = build_analysis_path("b", "answer_dist", "blink", "counting", "/a/b/Model/", "leaf")
    assert path == Path("b/answer_dist/blink/counting/Model/leaf")


def test_leaf_parts_prefix():
    assert leaf_parts_prefix() == []
    assert leaf_parts_prefix(5, "cot", True) == ["sps5", "thinking", "cot"]
