import pytest

from vlm_icl.models.api import GeminiVLModel, OpenAIVLModel
from vlm_icl.models.base import ATTENTIONS, HIDDEN_STATES, ISOLATION_MASK, openai_chat_messages, require_capability
from vlm_icl.models.hf import Gemma4VLModel, InternVL35Model, Qwen25VLModel, Qwen35Model, Qwen3VLModel
from vlm_icl.models.registry import detect_family, load_model
from vlm_icl.models.vllm import VLLMModel
from vlm_icl.prompts import build_conversation, get_cot_prompt


@pytest.mark.parametrize("name,family", [
    ("GPT", "gpt"),
    ("gpt", "gpt"),
    ("  Gemini ", "gemini"),
    ("GEMINI", "gemini"),
    ("/m/InternVL3_5-8B", "internvl"),
    ("/m/gemma-4-E4B-it", "gemma"),
    ("/m/Qwen2.5-VL-7B", "qwen2.5"),
    ("/m/Qwen3.5-4B", "qwen3.5"),
    ("/m/Qwen3-VL-8B-Instruct", "qwen3"),
    ("/m/llava-1.5", None),
    ("/m/gpt-oss", None),
])
def test_detect_family(name, family):
    assert detect_family(name) == family


def test_load_model_rejects_before_loading():
    with pytest.raises(ValueError, match="vLLM"):
        load_model("GPT", backend="vllm")
    with pytest.raises(ValueError, match="auto-detect"):
        load_model("/m/llava", backend="default")


def test_class_capabilities():
    assert Qwen3VLModel.capabilities == frozenset({ATTENTIONS, HIDDEN_STATES, ISOLATION_MASK})
    for cls in (Qwen35Model, Gemma4VLModel, Qwen25VLModel, InternVL35Model, VLLMModel, GeminiVLModel, OpenAIVLModel):
        assert cls.capabilities == frozenset(), cls.__name__


class Stub:
    def __init__(self, capabilities):
        self.capabilities = frozenset(capabilities)


def test_require_capability():
    require_capability(Stub({ATTENTIONS}), ATTENTIONS)
    with pytest.raises(ValueError, match=ATTENTIONS):
        require_capability(Stub(set()), ATTENTIONS)
    with pytest.raises(ValueError):
        require_capability(Stub({HIDDEN_STATES}), ISOLATION_MASK)


def test_conversation_zero_shot():
    parts = build_conversation("Q?", "img")
    assert [k for k, _ in parts] == ["image", "text"]
    assert parts[0][1] == "img" and "Question: Q?" in parts[1][1]
    assert "Answer:" not in parts[1][1]


def test_conversation_icl_order():
    context = [
        {"image": "d1", "prompt": "P1", "answer": "A1"},
        {"image": "d2", "image2": "d2b", "prompt": "P2", "answer": "A2"},
    ]
    parts = build_conversation("Q?", "q", context=context, cot=True)
    kinds = [k for k, _ in parts]
    assert kinds == ["text", "image", "text", "image", "image", "text", "image", "text"]
    assert parts[0][1] == get_cot_prompt(icl=True)
    assert [v for k, v in parts if k == "image"] == ["d1", "d2", "d2b", "q"]
    assert "Question: P1" in parts[2][1] and parts[2][1].endswith("Answer: A1")
    assert parts[5][1].endswith("Answer: A2")
    assert "Question: Q?" in parts[7][1] and "Answer" not in parts[7][1]


def test_conversation_cot_zero_shot_and_multi_image():
    parts = build_conversation("Q?", "a", cot=True, image2="b")
    assert parts[0] == ("text", get_cot_prompt(icl=False))
    assert [v for k, v in parts if k == "image"] == ["a", "b"]
    assert parts[-1][1].startswith("Given the images above")


def test_openai_chat_messages():
    parts = [("text", "t0"), ("image", "i1"), ("image", "i2"), ("text", "t1")]
    messages = openai_chat_messages(parts, "sys", lambda x: f"url:{x}")
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == [
        {"type": "text", "text": "t0"},
        {"type": "image_url", "image_url": {"url": "url:i1"}},
        {"type": "image_url", "image_url": {"url": "url:i2"}},
        {"type": "text", "text": "t1"},
    ]
