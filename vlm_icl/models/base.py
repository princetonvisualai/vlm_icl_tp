"""The interface every model wrapper implements, and the OpenAI-format message adapter."""

from typing import Any, Callable, Dict, FrozenSet, List, Optional, Protocol

from ..prompts import ContentPart

# Optional analysis hooks a wrapper may expose, named in its ``capabilities``:
#   attentions      generate_with_attentions() and forward_attentions()
#   hidden_states   forward_hidden_states()
#   isolation_mask  set_isolation_mask() and clear_isolation_mask()
ATTENTIONS = "attentions"
HIDDEN_STATES = "hidden_states"
ISOLATION_MASK = "isolation_mask"


class VLModel(Protocol):
    """A vision-language model that answers a question about one or more images."""

    model_path: str
    enable_thinking: bool
    capabilities: FrozenSet[str]

    def generate(
        self,
        image: Any,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        cot: bool = False,
        image2: Any = None,
        image3: Any = None,
        image4: Any = None,
        **generation_kwargs,
    ) -> str:
        """Answer ``prompt`` about the query image(s), after the ICL demos in ``context``."""
        ...


def require_capability(model: VLModel, capability: str) -> None:
    """Raise ValueError unless ``model`` exposes ``capability``."""
    if capability not in model.capabilities:
        raise ValueError(
            f"{type(model).__name__} does not support '{capability}'; "
            "this analysis needs a Qwen3-VL model on the HF backend."
        )


def openai_chat_messages(
    parts: List[ContentPart],
    system_prompt: str,
    image_url: Callable[[Any], str],
) -> List[Dict[str, Any]]:
    """Render conversation parts as OpenAI-format chat messages (used by vLLM and GPT)."""
    content = [
        {"type": "text", "text": value} if kind == "text"
        else {"type": "image_url", "image_url": {"url": image_url(value)}}
        for kind, value in parts
    ]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
