"""HuggingFace Transformers wrappers: Qwen3-VL, Qwen2.5-VL, Qwen3.5, Gemma 4 and InternVL 3.5."""

import re
from typing import Any, Dict, FrozenSet, List, Optional, Union
from PIL import Image
import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoModelForImageTextToText,
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
    GenerationConfig,
)
try:
    from transformers import AutoModelForMultimodalLM
except ImportError:
    AutoModelForMultimodalLM = None
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from ..prompts import ContentPart, build_conversation, get_default_generation_params, get_system_prompt
from .base import ATTENTIONS, HIDDEN_STATES, ISOLATION_MASK


# Helper functions

def _chat_template_has_var(processor, var_name: str) -> bool:
    """Check if a processor's chat template references a given variable name."""
    tmpl = getattr(processor, "chat_template", None)
    if isinstance(tmpl, dict):
        tmpl = tmpl.get("default", next(iter(tmpl.values()), None))
    return tmpl is not None and var_name in tmpl


def _get_default_model_kwargs(**overrides) -> dict:
    """Get default model loading kwargs."""
    default_kwargs = {
        'torch_dtype': torch.bfloat16,
        'device_map': 'auto',
        'trust_remote_code': True
    }
    default_kwargs.update(overrides)
    return default_kwargs


def get_decoder_layers(model):
    """Get decoder layers from a HuggingFace VLM model."""
    inner = model.model
    return getattr(inner, 'layers', None) or inner.language_model.layers


def _register_attention_hooks(model, layer_indices=None):
    """Register forward hooks on self_attn modules to capture attention weights to CPU.

    Each hook intercepts (attn_output, attn_weights) from self_attn.forward(),
    moves attn_weights to CPU, and replaces it with None to free GPU memory.
    Returns (captured_dict, hook_handles, num_layers).

    Args:
        layer_indices: If provided, only hook these layer indices (saves memory).
    """
    captured = {}
    hooks = []
    decoder_layers = get_decoder_layers(model)
    num_layers = len(decoder_layers)

    for i, layer in enumerate(decoder_layers):
        if layer_indices is not None and i not in layer_indices:
            continue
        def make_hook(idx):
            def hook_fn(module, input, output):
                captured[idx] = output[1].detach().cpu()
                return (output[0], None)
            return hook_fn
        hooks.append(layer.self_attn.register_forward_hook(make_hook(i)))

    return captured, hooks, num_layers


class CapturedAttentions:
    """Tuple-like wrapper around hook-captured attention weights (CPU tensors).

    Provides backward-compatible access: attentions[layer_idx][0] -> (heads, seq, seq),
    len(attentions) -> num_layers, iteration over all layers.
    """

    def __init__(self, captured: dict, num_layers: int):
        self._captured = captured
        self._num_layers = num_layers

    def __len__(self):
        return self._num_layers

    def __getitem__(self, index):
        if index < 0:
            index = self._num_layers + index
        return self._captured[index]

    def __iter__(self):
        for i in range(self._num_layers):
            yield self._captured[i]


# ============================================================================
# InternVL Image Preprocessing
# ============================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INTERNVL_IMAGE_SIZE = 448
INTERNVL_MAX_TILES = 12


def _internvl_build_transform():
    """Build the InternVL image transform pipeline."""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((INTERNVL_IMAGE_SIZE, INTERNVL_IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _internvl_find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _internvl_dynamic_preprocess(image, min_num=1, max_num=INTERNVL_MAX_TILES,
                                  image_size=INTERNVL_IMAGE_SIZE, use_thumbnail=True):
    """Split image into tiles using dynamic aspect ratio matching."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio = _internvl_find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = best_ratio[0] * image_size
    target_height = best_ratio[1] * image_size
    blocks = best_ratio[0] * best_ratio[1]

    resized = image.resize((target_width, target_height))
    processed = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed.append(resized.crop(box))

    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))

    return processed


def _internvl_load_image(image_input, max_num=INTERNVL_MAX_TILES):
    """Process a single image into pixel_values tensor for InternVL.

    Args:
        image_input: PIL Image or file path string.
    Returns:
        Tensor of shape [num_tiles, 3, 448, 448] in bfloat16.
    """
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert('RGB')
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    transform = _internvl_build_transform()
    tiles = _internvl_dynamic_preprocess(pil_img, max_num=max_num)
    return torch.stack([transform(t) for t in tiles]).to(torch.bfloat16)


def _hf_messages(parts: List[ContentPart], system_prompt: str) -> List[Dict]:
    """Render conversation parts as HF chat-template messages (images passed through)."""
    content = [
        {"type": "text", "text": value} if kind == "text" else {"type": "image", "image": value}
        for kind, value in parts
    ]
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]


class _HFChatModel:
    """Base for HF models driven by a processor chat template.

    Subclasses set ``model_class`` (the transformers class that loads the weights) and may
    set ``capabilities`` and ``strip_thinking`` (drop ``<think>...</think>`` from answers
    when thinking is on).
    """

    model_class = AutoModelForImageTextToText
    capabilities: FrozenSet[str] = frozenset()
    strip_thinking = False

    def __init__(self, model_path: str, enable_thinking: bool = False, **model_kwargs):
        self.model_path = model_path
        self.enable_thinking = enable_thinking
        default_kwargs = _get_default_model_kwargs(**model_kwargs)
        self.model = self.model_class.from_pretrained(model_path, **default_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self._supports_thinking_kwarg = _chat_template_has_var(self.processor, "enable_thinking")

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
        messages = self.build_messages(image, prompt, context, system_prompt, cot=cot, image2=image2, image3=image3, image4=image4)
        return self._process_and_generate(messages, **generation_kwargs)

    def forward_attentions(
        self,
        image: Optional[Union[str, Image.Image]],
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        cot: bool = False,
        image2: Optional[Union[str, Image.Image]] = None,
        image3: Optional[Union[str, Image.Image]] = None,
        image4: Optional[Union[str, Image.Image]] = None,
        layer_stride: int = 2,
    ) -> Dict[int, torch.Tensor]:
        """Run forward pass and return head-averaged attention maps from sampled layers.

        Returns Dict[layer_index, Tensor(seq_len, seq_len)] for every `layer_stride`-th layer.
        """
        messages = self.build_messages(image, prompt, context, system_prompt, cot=cot, image2=image2, image3=image3, image4=image4)
        inputs = self.prepare_inputs(messages)

        attentions = {}
        hooks = []
        decoder_layers = get_decoder_layers(self.model)
        for i, layer in enumerate(decoder_layers):
            if i % layer_stride != 0:
                continue
            def make_hook(idx):
                def hook_fn(module, input, output):
                    attentions[idx] = output[1][0].mean(dim=0).detach().cpu()
                    return (output[0], None)
                return hook_fn
            hooks.append(layer.self_attn.register_forward_hook(make_hook(i)))

        try:
            with torch.no_grad():
                self.model(**inputs, use_cache=False, return_dict=True)
        finally:
            for h in hooks:
                h.remove()

        torch.cuda.empty_cache()
        return attentions

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
        """Build the chat-template messages for one query."""
        sys_prompt = system_prompt if system_prompt is not None else get_system_prompt(self.model_path)
        parts = build_conversation(prompt, image, context, cot=cot, image2=image2, image3=image3, image4=image4)
        return _hf_messages(parts, sys_prompt)

    def prepare_inputs(self, messages):
        """Tokenize chat messages and move to model device."""
        kwargs = dict(tokenize=True, add_generation_prompt=True,
                      return_dict=True, return_tensors="pt")
        if self._supports_thinking_kwarg:
            kwargs["enable_thinking"] = self.enable_thinking
        inputs = self.processor.apply_chat_template(messages, **kwargs)
        return inputs.to(self.model.device)

    def _process_and_generate(self, messages: List[Dict], **generation_kwargs) -> str:
        """Process messages through chat template and generate decoded response."""
        default_gen_kwargs = get_default_generation_params(self.model_path, backend="huggingface", enable_thinking=self.enable_thinking)
        default_gen_kwargs.update(generation_kwargs)

        inputs = self.prepare_inputs(messages)
        generated_ids = self.model.generate(**inputs, **default_gen_kwargs)

        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in
            zip(inputs.input_ids, generated_ids)
        ]

        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        if self.strip_thinking and self.enable_thinking:
            answer = re.sub(r'<think>.*?</think>\s*', '', answer, flags=re.DOTALL).strip()

        return answer


class Qwen3VLModel(_HFChatModel):
    """Qwen3-VL. Thinking mode loads the matching ``-Thinking`` checkpoint."""

    model_class = Qwen3VLForConditionalGeneration
    capabilities = frozenset({ATTENTIONS, HIDDEN_STATES, ISOLATION_MASK})
    strip_thinking = True

    def __init__(self, model_path: str, enable_thinking: bool = False, **model_kwargs):
        """Load model and processor. Use attn_implementation='eager' for attention visualization."""
        if enable_thinking:
            model_path = model_path.replace("-Instruct", "-Thinking")
        super().__init__(model_path, enable_thinking=enable_thinking, **model_kwargs)

    def set_isolation_mask(self, mask_fn=None, or_mask_fn=None, decode_only=False):
        """Set mask functions for the causal mask during generation.

        Args:
            mask_fn: Optional (b, h, q, kv) -> bool closure AND'd with causal mask.
            or_mask_fn: Optional (b, h, q, kv) -> bool closure OR'd with causal mask
                        (True = unmask beyond causality).
            decode_only: If True, only apply masks during decode (not prefill).
        """
        import transformers.models.qwen3_vl.modeling_qwen3_vl as mod
        if not hasattr(mod, '_original_create_causal_mask'):
            mod._original_create_causal_mask = mod.create_causal_mask
            original = mod._original_create_causal_mask

            def _patched(**kwargs):
                and_fn = getattr(mod, '_active_and_mask_function', None)
                or_fn = getattr(mod, '_active_or_mask_function', None)
                _decode_only = getattr(mod, '_active_decode_only', False)
                if _decode_only:
                    pkv = kwargs.get('past_key_values')
                    is_decode = (
                        pkv is not None
                        and hasattr(pkv, 'get_seq_length')
                        and pkv.get_seq_length() > 0
                    )
                    if not is_decode:
                        return original(**kwargs)
                if and_fn is not None:
                    kwargs['and_mask_function'] = and_fn
                if or_fn is not None:
                    kwargs['or_mask_function'] = or_fn
                return original(**kwargs)

            mod.create_causal_mask = _patched
        mod._active_and_mask_function = mask_fn
        mod._active_or_mask_function = or_mask_fn
        mod._active_decode_only = decode_only

    def clear_isolation_mask(self):
        """Clear the active mask functions."""
        import transformers.models.qwen3_vl.modeling_qwen3_vl as mod
        mod._active_and_mask_function = None
        mod._active_or_mask_function = None
        mod._active_decode_only = False

    def generate_with_attentions(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        cot: bool = False,
        image2: Optional[Union[str, Image.Image]] = None,
        image3: Optional[Union[str, Image.Image]] = None,
        image4: Optional[Union[str, Image.Image]] = None,
        layer_indices: Optional[set] = None,
        **generation_kwargs
    ) -> tuple[str, Dict[str, Any]]:
        """Run inference and return (answer, attention_data) for visualization.

        Args:
            layer_indices: If provided, only capture attention from these layer indices.
        """
        messages = self.build_messages(image, prompt, context, system_prompt, cot=cot, image2=image2, image3=image3, image4=image4)

        # Get default generation kwargs based on model type (HuggingFace backend)
        default_gen_kwargs = get_default_generation_params(self.model_path, backend="huggingface", enable_thinking=self.enable_thinking)
        # User-provided kwargs override defaults
        default_gen_kwargs.update(generation_kwargs)

        inputs = self.prepare_inputs(messages)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **default_gen_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in
            zip(inputs.input_ids, generated_ids)
        ]

        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        generated_ids_trimmed = torch.stack(generated_ids_trimmed, dim=0)
        full_input_ids = torch.cat([inputs.input_ids, generated_ids_trimmed], dim=-1)
        full_attention_mask = torch.cat(
            [inputs.attention_mask, torch.ones_like(generated_ids_trimmed)],
            dim=-1
        )

        model_inputs = dict(inputs)
        model_inputs["input_ids"] = full_input_ids
        model_inputs["attention_mask"] = full_attention_mask

        captured, hooks, num_layers = _register_attention_hooks(self.model, layer_indices=layer_indices)
        try:
            with torch.no_grad():
                self.model(**model_inputs, use_cache=False, return_dict=True)
        finally:
            for h in hooks:
                h.remove()

        attentions = CapturedAttentions(captured, num_layers)

        attention_data = {
            "tokenizer": self.processor.tokenizer,
            "input_ids": inputs.input_ids,
            "generated_ids_trimmed": generated_ids_trimmed,
            "full_input_ids": full_input_ids,
            "attentions": attentions,
            "num_layers": num_layers,
            "image_grid_thw": model_inputs.get("image_grid_thw"),
            "answer_text": answer
        }
        return answer, attention_data

    def forward_hidden_states(
        self,
        image: Optional[Union[str, Image.Image]],
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        cot: bool = False,
        image2: Optional[Union[str, Image.Image]] = None,
        image3: Optional[Union[str, Image.Image]] = None,
        image4: Optional[Union[str, Image.Image]] = None,
    ) -> Dict[int, torch.Tensor]:
        """Run forward pass and return last-token hidden states from every layer using hooks."""
        messages = self.build_messages(image, prompt, context, system_prompt, cot=cot, image2=image2, image3=image3, image4=image4)
        inputs = self.prepare_inputs(messages)

        hidden_states = {}
        hooks = []

        decoder_layers = get_decoder_layers(self.model)
        for i, layer in enumerate(decoder_layers):
            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    h = output[0]
                    if h.dim() == 3:
                        hidden_states[layer_idx] = h[0, -1, :].detach().cpu()
                    else:
                        hidden_states[layer_idx] = h[-1, :].detach().cpu()
                return hook_fn
            hooks.append(layer.register_forward_hook(make_hook(i)))

        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        return hidden_states


class Qwen25VLModel(_HFChatModel):
    """Qwen2.5-VL (no thinking mode)."""

    def __init__(self, model_path: str, enable_thinking: bool = False, **model_kwargs):
        if enable_thinking:
            raise ValueError(
                "Qwen2.5-VL does not support --enable_thinking. "
                "Use a Qwen3.5-VL model instead."
            )
        super().__init__(model_path, enable_thinking=False, **model_kwargs)


class Qwen35Model(_HFChatModel):
    """Qwen3.5. Its chat template defaults to thinking on, so enable_thinking is always passed."""


class Gemma4VLModel(_HFChatModel):
    """Gemma 4."""

    model_class = AutoModelForMultimodalLM


class InternVL35Model:
    """Wrapper for InternVL 3.5 model inference (standard and ICL).

    Uses AutoModel + AutoTokenizer with trust_remote_code and the model's
    built-in chat() method, which differs from the Qwen AutoProcessor pipeline.
    """

    capabilities: FrozenSet[str] = frozenset()

    THINKING_SYSTEM_PROMPT = (
        "You are an AI assistant that rigorously follows this response protocol:\n"
        "1. First, conduct a detailed analysis of the question. Consider different angles, "
        "potential solutions, and reason through the problem step-by-step. Enclose this entire "
        "thinking process within <think> and </think> tags.\n"
        "2. After the thinking section, provide a clear, concise, and direct answer to the user's "
        "question. Separate the answer from the think section with a newline. Ensure that the "
        "thinking process is thorough but remains focused on the query.\n\n"
        "The final answer should be standalone and not reference the thinking section."
    )

    def __init__(self, model_path: str, enable_thinking: bool = False, **model_kwargs):
        self.enable_thinking = enable_thinking
        self.model_path = model_path

        default_kwargs = _get_default_model_kwargs(**model_kwargs)
        self.model = AutoModel.from_pretrained(model_path, **default_kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if self.enable_thinking:
            self.model.system_message = self.THINKING_SYSTEM_PROMPT

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
        question, pixel_values = self._build_query(
            image, prompt, context, cot=cot,
            image2=image2, image3=image3, image4=image4,
        )

        default_gen_kwargs = get_default_generation_params(
            self.model_path, backend="huggingface", enable_thinking=self.enable_thinking
        )
        default_gen_kwargs.update(generation_kwargs)
        gen_config = GenerationConfig(**default_gen_kwargs)

        if pixel_values is not None:
            pixel_values = pixel_values.to(self.model.device)

        response = self.model.chat(self.tokenizer, pixel_values, question, gen_config)
        if self.enable_thinking:
            response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL).strip()
        return response

    def _build_query(
        self,
        image: Optional[Union[str, Image.Image]],
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        cot: bool = False,
        image2: Optional[Union[str, Image.Image]] = None,
        image3: Optional[Union[str, Image.Image]] = None,
        image4: Optional[Union[str, Image.Image]] = None,
    ) -> tuple:
        """Build text query with <image> placeholders and concatenated pixel_values.

        Returns (question_str, pixel_values_tensor_or_None).
        """
        parts = build_conversation(prompt, image, context, cot=cot, image2=image2, image3=image3, image4=image4)
        all_pixel_values = [_internvl_load_image(value) for kind, value in parts if kind == "image"]
        question = '\n'.join('<image>' if kind == "image" else value for kind, value in parts)
        pixel_values = torch.cat(all_pixel_values, dim=0) if all_pixel_values else None
        return question, pixel_values
