"""API-based VLM backends (Google Gemini, OpenAI GPT) with same generate() interface."""

import os
import time
from typing import Any, Dict, FrozenSet, List, Optional, Union

from PIL import Image

from ..images import pil_to_data_url, to_pil
from ..prompts import build_conversation, get_default_generation_params, get_system_prompt
from .base import openai_chat_messages


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

class GeminiVLModel:
    """Google Gemini API backend. Requires GOOGLE_API_KEY env var."""

    capabilities: FrozenSet[str] = frozenset()

    def __init__(self, model_path: str, api_model: str = "gemini-3-flash-preview", enable_thinking: bool = False, **kwargs):
        from google import genai

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY environment variable is required for the Gemini backend."
            )

        self.model_path = model_path
        self.enable_thinking = enable_thinking
        self._client = genai.Client(api_key=api_key)
        self._types = genai.types
        self._api_model = api_model

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
        **generation_kwargs,
    ) -> str:
        """Run inference via Gemini API. Returns answer string."""
        sys_prompt = system_prompt if system_prompt is not None else get_system_prompt(self.model_path)
        parts = build_conversation(prompt, image, context, cot=cot, image2=image2, image3=image3, image4=image4)
        # The google-genai SDK takes text and PIL images directly
        content = [value if kind == "text" else to_pil(value) for kind, value in parts]

        gen_config = get_default_generation_params(self.model_path, backend="gemini", enable_thinking=self.enable_thinking)
        gen_config.update(generation_kwargs)
        thinking_budget = gen_config.pop("thinking_budget")
        gen_config["thinking_config"] = self._types.ThinkingConfig(thinking_budget=thinking_budget)

        config = self._types.GenerateContentConfig(
            system_instruction=sys_prompt,
            **gen_config,
        )

        max_retries = 8
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._api_model,
                    contents=content,
                    config=config,
                )
                return response.text
            except Exception as e:
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < max_retries - 1:
                    wait = min(60, 5 * (2 ** attempt))  # 5, 10, 20, 40, 60, 60, 60 seconds (~255s total)
                    print(f"[Gemini] Rate limited (attempt {attempt + 1}/{max_retries}), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise


# ---------------------------------------------------------------------------
# OpenAI GPT
# ---------------------------------------------------------------------------

class OpenAIVLModel:
    """OpenAI GPT API backend. Requires OPENAI_API_KEY env var."""

    capabilities: FrozenSet[str] = frozenset()

    def __init__(self, model_path: str, api_model: str = "gpt-5.2", enable_thinking: bool = False, **kwargs):
        from openai import OpenAI

        self.model_path = model_path
        self.enable_thinking = enable_thinking
        self._client = OpenAI()  # uses OPENAI_API_KEY automatically
        self._api_model = api_model
        self._data_url_cache: Dict[int, str] = {}

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
        **generation_kwargs,
    ) -> str:
        """Run inference via OpenAI API. Returns answer string."""
        sys_prompt = system_prompt if system_prompt is not None else get_system_prompt(self.model_path)
        parts = build_conversation(prompt, image, context, cot=cot, image2=image2, image3=image3, image4=image4)
        messages = openai_chat_messages(parts, sys_prompt, self._to_data_url)

        params = get_default_generation_params(self.model_path, backend="openai", enable_thinking=self.enable_thinking)
        params.update(generation_kwargs)

        response = self._client.chat.completions.create(
            model=self._api_model,
            messages=messages,
            **params,
        )
        return response.choices[0].message.content

    def _to_data_url(self, image_input) -> str:
        """Convert image input to a cached base64 data URL."""
        # Cache hit for same PIL object (useful for pass@k repeated calls)
        obj_id = id(image_input)
        if obj_id in self._data_url_cache:
            return self._data_url_cache[obj_id]

        pil_img = to_pil(image_input)
        data_url = pil_to_data_url(pil_img)
        self._data_url_cache[obj_id] = data_url
        return data_url
