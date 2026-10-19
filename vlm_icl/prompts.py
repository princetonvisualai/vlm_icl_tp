"""ICL prompt templates, system and chain-of-thought prompts, and generation parameters."""

from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# Generation Parameters
# ============================================================================

# Standard generation parameters for regular VLM models
STANDARD_GENERATION_PARAMS = {
    'temperature': 0.7,
    'top_p': 0.8,
    'top_k': 20,
    'presence_penalty': 1.5,
    'repetition_penalty': 1.0,
    'max_new_tokens': 512,
    'do_sample': True  # Required for temperature/top_p/top_k to take effect
}


# Google-recommended generation parameters for Gemma 4 models
GEMMA4_GENERATION_PARAMS = {
    'temperature': 1.0,
    'top_p': 0.95,
    'top_k': 64,
    'presence_penalty': 0.0,
    'repetition_penalty': 1.0,
    'max_new_tokens': 512,
    'do_sample': True  # Required for temperature/top_p/top_k to take effect
}


# Generation parameters for "thinking" models (e.g., QwQ models)
THINKING_GENERATION_PARAMS = {
    'temperature': 1.0,
    'top_p': 0.95,
    'top_k': 20,
    'presence_penalty': 0.0,
    'repetition_penalty': 1.0,
    'max_new_tokens': 1024,
    'do_sample': True  # Required for temperature/top_p/top_k to take effect
}


# Parameters supported by HuggingFace Transformers (presence_penalty is NOT supported)
HF_SUPPORTED_PARAMS = {
    'temperature', 'top_p', 'top_k', 'repetition_penalty',
    'max_new_tokens', 'do_sample', 'max_length', 'min_length',
    'num_beams', 'early_stopping', 'length_penalty'
}


# Parameters supported by vLLM (supports presence_penalty)
VLLM_SUPPORTED_PARAMS = {
    'temperature', 'top_p', 'top_k', 'repetition_penalty',
    'presence_penalty', 'frequency_penalty', 'max_tokens',
    'max_new_tokens', 'min_tokens', 'stop', 'n', 'best_of',
    'logprobs', 'skip_special_tokens'
}


# API backends use their provider's parameter names. Gemini's thinking_budget becomes a ThinkingConfig.
OPENAI_GENERATION_PARAMS = {
    'temperature': 1.0,
    'top_p': 0.95,
    'max_completion_tokens': 256,
    'reasoning_effort': 'none',
}


OPENAI_THINKING_GENERATION_PARAMS = {
    'max_completion_tokens': 1024,
    'reasoning_effort': 'medium',
}


GEMINI_GENERATION_PARAMS = {
    'temperature': 1.0,
    'top_p': 0.95,
    'max_output_tokens': 1024,
    'thinking_budget': 0,
}


def get_default_generation_params(model_path: str, backend: str = "huggingface", enable_thinking: bool = False) -> dict:
    """Return the default generation parameters for a model on the given backend.

    ``backend`` is ``huggingface`` (or ``hf``/``transformers``/``default``), ``vllm``, ``openai``
    or ``gemini``. Local backends get the model's parameters filtered to the supported set.
    """
    if backend.lower() == "openai":
        return dict(OPENAI_THINKING_GENERATION_PARAMS if enable_thinking else OPENAI_GENERATION_PARAMS)
    if backend.lower() == "gemini":
        return {**GEMINI_GENERATION_PARAMS, 'thinking_budget': 512 if enable_thinking else 0}

    # Get base parameters: model-specific overrides first, then thinking mode
    model_path_lower = model_path.lower()
    if 'gemma' in model_path_lower:
        params = GEMMA4_GENERATION_PARAMS.copy()
        if enable_thinking:
            params['max_new_tokens'] = 1024
    elif enable_thinking:
        params = THINKING_GENERATION_PARAMS.copy()
    else:
        params = STANDARD_GENERATION_PARAMS.copy()

    # Filter parameters based on backend
    if backend.lower() in ["huggingface", "hf", "transformers", "default"]:
        # Remove parameters not supported by HuggingFace Transformers
        params = {k: v for k, v in params.items() if k in HF_SUPPORTED_PARAMS}
    elif backend.lower() == "vllm":
        # Keep all vLLM-supported parameters
        params = {k: v for k, v in params.items() if k in VLLM_SUPPORTED_PARAMS}

    return params


# ============================================================================
# System Prompts and ICL Templates
# ============================================================================

DEFAULT_SYSTEM_PROMPTS = {
    'qwen3vl': "You are a helpful assistant.",
    'internvl': "You are a helpful assistant.",
}


ICL_DEMO_TEMPLATE = """Given the image above, answer the following question.
Question: {question}
Answer: {answer}"""


ICL_QUERY_TEMPLATE = """Given the image above, answer the following question.
Question: {question}"""


ICL_DEMO_TEMPLATE_MULTI = """Given the images above, answer the following question.
Question: {question}
Answer: {answer}"""


ICL_QUERY_TEMPLATE_MULTI = """Given the images above, answer the following question.
Question: {question}"""


def get_icl_templates(multi_image: bool = False):
    """Return the demo and query ICL templates for single- or multi-image samples."""
    if multi_image:
        return ICL_DEMO_TEMPLATE_MULTI, ICL_QUERY_TEMPLATE_MULTI
    return ICL_DEMO_TEMPLATE, ICL_QUERY_TEMPLATE


COT_PROMPT_ZEROSHOT = "Let's think step by step. Give reasoning rationales before providing the final answer."


COT_PROMPT_ICL = "Let's first think step by step and analyze the relationship between the given few-shot question-answer pairs. Give reasoning rationales before providing the final answer."


def get_cot_prompt(icl: bool = False) -> str:
    """Return the chain-of-thought prompt for zero-shot or ICL mode."""
    return COT_PROMPT_ICL if icl else COT_PROMPT_ZEROSHOT


def get_system_prompt(model_path: str) -> str:
    """Return the default system prompt for the given model."""
    model_path_lower = model_path.lower()
    if 'internvl' in model_path_lower or 'intern_vl' in model_path_lower:
        return DEFAULT_SYSTEM_PROMPTS['internvl']
    return DEFAULT_SYSTEM_PROMPTS['qwen3vl']


def _prompt_needs_curly_braces(prompt: Optional[str]) -> bool:
    if not prompt:
        return False
    lowered = prompt.lower()
    return (
        "curly bracket" in lowered
        or "curly brackets" in lowered
        or ("{" in prompt and "}" in prompt)
    )


def format_icl_answer(prompt: Optional[str], answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return answer
    if not _prompt_needs_curly_braces(prompt):
        return answer
    stripped = answer.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    return f"{{{stripped}}}"


ContentPart = Tuple[str, Any]


def build_conversation(
    prompt: str,
    image: Any = None,
    context: Optional[List[Dict[str, Any]]] = None,
    cot: bool = False,
    image2: Any = None,
    image3: Any = None,
    image4: Any = None,
) -> List[ContentPart]:
    """Build the user turn as backend-neutral ``("text", str)`` and ``("image", obj)`` parts.

    The order is the CoT prompt (if ``cot``), then each demo's images and its answered
    question, then the query's images and its question. Images are passed through as given;
    each backend converts them to its own image format.
    """
    demo_template, query_template = get_icl_templates(image2 is not None)
    parts: List[ContentPart] = []
    if cot:
        parts.append(("text", get_cot_prompt(icl=bool(context))))
    for ctx in context or []:
        for key in ("image", "image2", "image3", "image4"):
            if ctx.get(key) is not None:
                parts.append(("image", ctx[key]))
        parts.append(("text", demo_template.format(question=ctx["prompt"], answer=ctx["answer"])))
    for img in (image, image2, image3, image4):
        if img is not None:
            parts.append(("image", img))
    parts.append(("text", query_template.format(question=prompt)))
    return parts
