"""Answer scoring: generate a response and read candidate-label logprobs at the answer brace."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Literal, Mapping, Optional, Protocol, Sequence, Tuple, Union

import torch
from PIL import Image

try:
    from vllm import SamplingParams
except ImportError:  # pragma: no cover
    SamplingParams = None

from .metrics import normalize_answer
from .models import load_model
from .models.vllm import VLLMModel


ImageInput = Optional[Union[str, Image.Image]]
PartitionKey = Tuple[str, str]  # (task, split)
BraceVariant = Literal["{", "{ "]


def _find_last_brace_token_pos(token_ids, tokenizer) -> Optional[int]:
    """Find the index of the last token whose decoded string contains '{'."""
    last_pos = None
    for i, tid in enumerate(token_ids):
        decoded = tokenizer.decode([tid])
        if '{' in decoded:
            last_pos = i
    return last_pos


def tokenize_label_after_brace(tokenizer, *, label: str, brace_variant: BraceVariant) -> List[int]:
    """Tokenize the label as it would appear after the scoring brace prefix."""
    full = tokenizer.encode(f"{brace_variant}{label}", add_special_tokens=False)
    prefix = tokenizer.encode(brace_variant, add_special_tokens=False)
    label_ids = full[len(prefix) :]
    if not label_ids:
        raise ValueError(f"Label tokenization produced empty ids for label={label!r}")
    return label_ids


class AnswerScorer(Protocol):
    """Backend-agnostic interface for scoring answers."""

    def score_answers(
        self,
        *,
        image: ImageInput,
        prompt: str,
        labels: Sequence[str],
        context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Generate response, extract answer, and return label logprobs at answer position."""
        ...


class HFAnswerScorer:
    """HuggingFace-backed scorer."""

    def __init__(
        self,
        *,
        model_path: str,
        cot: bool = False,
        enable_thinking: bool = False,
    ) -> None:
        self.model_wrapper = load_model(model_path, enable_thinking=enable_thinking)
        self.model = self.model_wrapper.model
        self.processor = self.model_wrapper.processor
        self.cot = cot

    def score_answers(
        self,
        *,
        image: ImageInput,
        prompt: str,
        labels: Sequence[str],
        context: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """Generate response, extract answer from {}, and extract label distribution at answer position."""
        messages = self.model_wrapper.build_messages(image, prompt, context, system_prompt=None, cot=self.cot)
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        # 1. Generate full response, capturing logits at every position
        with torch.inference_mode():
            gen_output = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=max_tokens,
                return_dict_in_generate=True, output_logits=True,
            )
        generated_only = gen_output.sequences[0, input_len:]
        tokenizer = self.processor.tokenizer
        generated_text = tokenizer.decode(generated_only, skip_special_tokens=True)
        all_logits = gen_output.logits  # tuple of [1, vocab] tensors, one per step

        # 2. Extract predicted answer from last {...}
        predicted_label = normalize_answer(generated_text)

        # 3. Find last '{' token and extract label logprobs at brace+1 position
        label_logprobs = {l: float('-inf') for l in labels}
        top_k_at_answer: List[Tuple[int, str, float]] = []
        mass_in_topk = 0.0
        brace_pos = _find_last_brace_token_pos(generated_only.tolist(), tokenizer)
        if brace_pos is not None and brace_pos + 1 < len(all_logits):
            logits = all_logits[brace_pos + 1].squeeze(0)  # [vocab]
            log_probs = torch.log_softmax(logits, dim=0)
            probs = torch.softmax(logits, dim=0)
            for label in labels:
                token_ids = tokenize_label_after_brace(tokenizer, label=label, brace_variant="{")
                if token_ids:
                    label_logprobs[label] = log_probs[token_ids[0]].item()
            # Extract top-k tokens at answer position
            k = min(20, probs.shape[0])
            top_vals, top_ids = torch.topk(probs, k)
            for val, tid in zip(top_vals.tolist(), top_ids.tolist()):
                top_k_at_answer.append((tid, tokenizer.decode([tid]), val))
            mass_in_topk = float(top_vals.sum())

        return {
            "predicted_label": predicted_label,
            "generated_text": generated_text,
            "label_logprobs": label_logprobs,
            "top_k_at_answer": top_k_at_answer,
            "mass_in_topk": mass_in_topk,
        }


class VLLMAnswerScorer:
    """vLLM-backed scorer."""

    def __init__(
        self,
        *,
        model_path: str,
        tensor_parallel_size: int = 1,
        max_num_images: int = 64,
        max_model_len: int = 32768,
        cot: bool = False,
        enable_thinking: bool = False,
        # TODO: [vllm config] Add gpu_memory_utilization, enforce_eager, etc. if needed.
    ) -> None:
        """Initialize the vLLM engine."""
        if SamplingParams is None:
            raise ImportError("vLLM is not installed; install vllm to use backend='vllm'")
        self.model_wrapper = VLLMModel(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_num_images=max_num_images,
            max_model_len=max_model_len,
            enable_thinking=enable_thinking,
        )
        self.llm = self.model_wrapper.llm
        self.cot = cot
        
        # Get tokenizer for label tokenization
        self.tokenizer = self.llm.get_tokenizer()


    def score_answers(
        self,
        *,
        image: ImageInput,
        prompt: str,
        labels: Sequence[str],
        context: Optional[List[Dict[str, Any]]] = None,
        logprobs_k: int = 20,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """Generate response, extract answer from {}, and extract label distribution at answer position."""
        # 1. Generate full response with logprobs at every position
        messages = self.model_wrapper.build_messages(
            image=image, prompt=prompt, context=context, cot=self.cot,
        )
        sampling_params = SamplingParams(
            max_tokens=max_tokens, temperature=0.0, logprobs=logprobs_k,
        )
        outputs = self.llm.chat(
            messages, sampling_params=sampling_params, use_tqdm=False, **self.model_wrapper.chat_kwargs(),
        )
        output_obj = outputs[0].outputs[0]
        generated_text = output_obj.text

        # 2. Extract predicted answer from last {...}
        predicted_label = normalize_answer(generated_text)

        # 3. Find last '{' token and extract label logprobs at brace+1 position
        label_logprobs = {l: float('-inf') for l in labels}
        top_k_at_answer: List[Tuple[int, str, float]] = []
        mass_in_topk = 0.0
        brace_pos = _find_last_brace_token_pos(output_obj.token_ids, self.tokenizer)
        if brace_pos is not None and brace_pos < len(output_obj.token_ids) - 1:
            logprobs_data = output_obj.logprobs[brace_pos + 1]
            for label in labels:
                token_ids = tokenize_label_after_brace(self.tokenizer, label=label, brace_variant="{")
                if token_ids and token_ids[0] in logprobs_data:
                    label_logprobs[label] = logprobs_data[token_ids[0]].logprob
            # Extract top-k tokens at answer position
            for tid, lp_obj in sorted(logprobs_data.items(), key=lambda x: x[1].logprob, reverse=True):
                prob = math.exp(lp_obj.logprob)
                top_k_at_answer.append((tid, lp_obj.decoded_token, prob))
            mass_in_topk = sum(p for _, _, p in top_k_at_answer)

        return {
            "predicted_label": predicted_label,
            "generated_text": generated_text,
            "label_logprobs": label_logprobs,
            "top_k_at_answer": top_k_at_answer,
            "mass_in_topk": mass_in_topk,
        }


def build_label_vocab_by_partition(
    data: Sequence[Mapping[str, Any]],
    *,
    include_expected_bias: bool = True,
) -> Dict[PartitionKey, List[str]]:
    """Build candidate label vocabularies per (task, split)."""
    # Collect labels per partition
    labels_by_partition = defaultdict(set)
    
    for entry in data:
        task = entry["task"]
        split = entry["split"]
        partition_key = (task, split)
        
        for question in entry.get("questions", []):
            # Add normalized ground truth
            gt = normalize_answer(question["ground_truth"])
            labels_by_partition[partition_key].add(gt)
            
            # Optionally add expected bias
            if include_expected_bias and question.get("expected_bias"):
                bias = normalize_answer(question["expected_bias"])
                labels_by_partition[partition_key].add(bias)
    
    # Convert to sorted lists for stable ordering
    result = {
        key: sorted(labels)
        for key, labels in labels_by_partition.items()
    }

    return result
