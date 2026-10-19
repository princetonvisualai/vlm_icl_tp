"""vLLM backend for VLM inference (model-agnostic)."""

from pathlib import Path
from typing import Union, List, Dict, Any, FrozenSet, Optional
from PIL import Image

from ..images import pil_to_data_url
from ..prompts import build_conversation, get_default_generation_params, get_system_prompt
from .base import openai_chat_messages

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class VLLMModel:
    """Unified vLLM wrapper for VLM models (Qwen, Gemma, InternVL, etc.)."""

    capabilities: FrozenSet[str] = frozenset()

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        allowed_local_media_path: Optional[str] = None,
        gpu_memory_utilization: float = 0.9,
        max_num_images: int = 64,
        enable_thinking: bool = False,
        **kwargs
    ):
        """Initialize vLLM engine with tensor parallelism and multimodal support."""
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM backend requires vllm>=0.11.0. Install with:\n"
                "  pip install vllm>=0.11.0"
            )
        self.enable_thinking = enable_thinking
        self._is_gemma = 'gemma' in model_path.lower()

        # Build LLM kwargs
        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "limit_mm_per_prompt": {"image": max_num_images},
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if allowed_local_media_path is not None:
            llm_kwargs["allowed_local_media_path"] = allowed_local_media_path

        # Filter out non-vLLM kwargs
        vllm_keys = {
            "model", "tensor_parallel_size", "trust_remote_code", "dtype",
            "limit_mm_per_prompt", "gpu_memory_utilization", "max_model_len",
            "allowed_local_media_path", "allowed_media_domains",
            "pipeline_parallel_size", "max_num_seqs", "quantization",
            "enforce_eager", "enable_chunked_prefill"
        }
        for k, v in kwargs.items():
            if k in vllm_keys:
                llm_kwargs[k] = v

        self.llm = LLM(**llm_kwargs)
        self.model_path = model_path
        self.allowed_local_media_path = llm_kwargs.get("allowed_local_media_path")

    def generate(
        self,
        image: Optional[Union[str, Image.Image]],
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        cot: bool = False,
        image2: Optional[Union[str, Image.Image]] = None,
        image3: Optional[Union[str, Image.Image]] = None,
        image4: Optional[Union[str, Image.Image]] = None,
        **generation_kwargs
    ) -> str:
        """Run inference with optional ICL context. Returns answer string."""
        if not prompt:
            raise ValueError(
                "Prompt is empty or None — cannot send an empty decoder prompt to vLLM. "
                "Check that the dataset sample has a valid 'prompt' field."
            )
        messages = self.build_messages(image, prompt, context, system_prompt, cot=cot, image2=image2, image3=image3, image4=image4)
        sampling_params = self._build_sampling_params(**generation_kwargs)
        outputs = self.llm.chat(
            messages, sampling_params=sampling_params, use_tqdm=False,
            **self.chat_kwargs(),
        )
        return outputs[0].outputs[0].text

    def chat_kwargs(self) -> Dict[str, Any]:
        """Extra ``LLM.chat`` kwargs that switch thinking on or off.

        Gemma takes a ``<|think|>`` prefix in the system prompt instead (see build_messages).
        """
        if self._is_gemma:
            return {}
        return {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}

    def build_messages(
        self,
        image: Optional[Union[str, Image.Image]],
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        cot: bool = False,
        image2: Optional[Union[str, Image.Image]] = None,
        image3: Optional[Union[str, Image.Image]] = None,
        image4: Optional[Union[str, Image.Image]] = None,
    ) -> List[Dict]:
        """Build chat messages for vLLM in OpenAI format."""
        sys_prompt = system_prompt if system_prompt is not None else get_system_prompt(self.model_path)
        # Gemma 4 enables thinking via a <|think|> prefix in the system prompt
        if self._is_gemma and self.enable_thinking:
            sys_prompt = "<|think|>\n" + sys_prompt
        parts = build_conversation(prompt, image, context, cot=cot, image2=image2, image3=image3, image4=image4)
        return openai_chat_messages(parts, sys_prompt, self._to_image_url)

    def _to_image_url(self, image: Union[str, Image.Image]) -> str:
        """Convert image to a vLLM-supported URL (data: or file://)."""
        if isinstance(image, str):
            if image.startswith(("http://", "https://", "file://", "data:")):
                return image

            p = Path(image)
            if not p.is_file():
                raise FileNotFoundError(f"Image path not found: {image}")

            # Prefer file:// only when vLLM is explicitly allowed to load local files
            # from this directory; otherwise fall back to a data: URL to avoid
            # `Cannot load local files without --allowed-local-media-path`.
            if self.allowed_local_media_path:
                try:
                    allowed_root = Path(self.allowed_local_media_path).resolve()
                    resolved = p.resolve()
                    if resolved.is_relative_to(allowed_root):
                        return f"file://{resolved}"
                except Exception:
                    # If path resolution fails for any reason, fall back to data URL.
                    pass

            with Image.open(p) as img:
                return pil_to_data_url(img)

        return pil_to_data_url(image)

    def _build_sampling_params(self, **kwargs) -> "SamplingParams":
        """Build vLLM SamplingParams from generation kwargs using model-type defaults."""
        default_gen_kwargs = get_default_generation_params(self.model_path, backend="vllm", enable_thinking=self.enable_thinking)
        default_gen_kwargs.update(kwargs)

        # Extract max_tokens (vLLM uses max_tokens instead of max_new_tokens)
        max_tokens = default_gen_kwargs.pop("max_new_tokens", 512)

        # Remove do_sample as vLLM doesn't use it (sampling is controlled by temperature)
        default_gen_kwargs.pop("do_sample", None)

        return SamplingParams(max_tokens=max_tokens, **default_gen_kwargs)

