"""Model family detection and the (family, backend) → wrapper registry behind load_model()."""

from typing import Optional

from .base import VLModel
from .hf import Gemma4VLModel, InternVL35Model, Qwen25VLModel, Qwen35Model, Qwen3VLModel
from .api import GeminiVLModel, OpenAIVLModel
from .vllm import VLLMModel

# API models are named exactly (case-insensitive), so a local path never matches one.
API_FAMILIES = ("gemini", "gpt")

# Local families in lookup order: a path belongs to the first family with a substring in it.
LOCAL_FAMILIES = (
    ("internvl", ("internvl", "intern_vl")),
    ("gemma", ("gemma",)),
    ("qwen2.5", ("qwen2.5", "qwen2_5")),
    ("qwen3.5", ("qwen3.5", "qwen3_5")),
    ("qwen3", ("qwen",)),
)

REGISTRY = {
    ("gemini", "api"): GeminiVLModel,
    ("gpt", "api"): OpenAIVLModel,
    ("internvl", "hf"): InternVL35Model,
    ("gemma", "hf"): Gemma4VLModel,
    ("qwen2.5", "hf"): Qwen25VLModel,
    ("qwen3.5", "hf"): Qwen35Model,
    ("qwen3", "hf"): Qwen3VLModel,
}


def detect_family(model_path: str) -> Optional[str]:
    """Return the model family named by ``model_path``, or None if none matches."""
    name = model_path.strip().lower()
    if name in API_FAMILIES:
        return name
    path = model_path.lower()
    for family, substrings in LOCAL_FAMILIES:
        if any(s in path for s in substrings):
            return family
    return None


def load_model(
    model_path: str,
    backend: str = "default",
    enable_thinking: bool = False,
    **model_kwargs,
) -> VLModel:
    """Load the wrapper for ``model_path`` on ``backend`` (``default`` = HF Transformers, or ``vllm``).

    ``Gemini`` and ``GPT`` load the API clients. vLLM serves any local model through one
    family-agnostic wrapper.
    """
    family = detect_family(model_path)
    if family in API_FAMILIES:
        if backend == "vllm":
            raise ValueError("API models do not support the vLLM backend")
        cls = REGISTRY[family, "api"]
    elif backend == "vllm":
        cls = VLLMModel
    else:
        cls = REGISTRY.get((family, "hf"))
        if cls is None:
            raise ValueError(
                f"Cannot auto-detect model type from path: {model_path}. "
                "Supported models: InternVL-3.5, Qwen2.5-VL, Qwen3-VL, Qwen3.5-VL, Gemma 4, Gemini (API), GPT (API)."
            )
    return cls(model_path, enable_thinking=enable_thinking, **model_kwargs)
