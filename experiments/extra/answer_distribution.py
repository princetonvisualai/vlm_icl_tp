"""Answer distribution analysis at the {answer} location per (task, split) partition."""

from __future__ import annotations

import argparse
import json
import random
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Literal,
)

import torch

from vlm_icl.data import prepare_data
from vlm_icl.images import resolve_image
from vlm_icl.prompts import format_icl_answer
from vlm_icl.icl import (
    apply_special_mode,
    build_demo_pool,
    build_icl_context_pools,
    build_label_space,
    corrupt_icl_answers,
    process_icl_context_images,
    select_from_demo_pool,
    shuffle_icl_answers,
    validate_special_mode,
)
from vlm_icl.paths import build_output_path, derive_run_config
from vlm_icl.metrics import normalize_answer
from vlm_icl.scoring import (
    BraceVariant,
    HFAnswerScorer,
    ImageInput,
    PartitionKey,
    VLLMAnswerScorer,
    build_label_vocab_by_partition,
)

if TYPE_CHECKING:
    import numpy as np



def _infer_brace_variant(prompt_prefix: str) -> BraceVariant:
    if prompt_prefix.endswith("{ "):
        return "{ "
    return "{"


def _strip_score_prompt_suffix(prompt_prefix: str) -> str:
    """Recover the original prompt by stripping the trailing brace suffix."""
    for suffix in ["\n{ ", "\n{"]:
        if prompt_prefix.endswith(suffix):
            return prompt_prefix[:-len(suffix)]
    return prompt_prefix


def _extract_last_logits(logits: torch.Tensor) -> torch.Tensor:
    # Support either [B, V] or [B, T, V] shapes.
    if logits.ndim == 2:
        return logits
    if logits.ndim == 3:
        return logits[:, -1, :]
    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")


@dataclass(frozen=True)
class ScoringQuery:
    """Single scoring query for candidate label scoring."""

    image_id: str
    task: str
    topic: Optional[str]
    split: str
    question_type: str
    prompt: str
    ground_truth: str
    expected_bias: Optional[str]
    image: ImageInput
    context: Optional[List[Dict[str, Any]]]
    score_prompt: str
    partition_key: PartitionKey


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for answer distribution analysis."""

    model_path: str
    backend: Literal["default", "vllm"] = "default"

    tensor_parallel_size: int = 1
    max_num_images: int = 64
    max_model_len: int = 32768

    seed: int = 42
    output_dir: Optional[str] = None

    tasks: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    verified_only: Optional[bool] = None
    sample_per_split: Optional[int] = None

    icl: bool = False
    num_icl_examples: int = 2
    num_icl_sample_sets: int = 1

    special: Optional[str] = None
    cot: bool = False
    enable_thinking: bool = False
    query_gt_demo: Optional[str] = None
    max_samples: Optional[int] = None

    include_expected_bias: bool = True
    brace_variant: BraceVariant = "{"
    vllm_logprobs_k: int = 20
    data_root: Optional[str] = None
    data_source: str = "auto"


def make_score_prompt_prefix(prompt: str, *, variant: BraceVariant = "{") -> str:
    """Build the scoring prompt prefix ending at the answer location."""
    return f"{prompt.rstrip()}\n{variant}"


def iter_queries(
    *,
    data: Sequence[Mapping[str, Any]],
    label_vocab_by_partition: Mapping[PartitionKey, Sequence[str]],
    icl: bool,
    num_icl_examples: int,
    num_icl_sample_sets: int,
    seed: int,
    special: Optional[str],
    cot: bool,
    brace_variant: BraceVariant,
    prefer_pil: bool,
    max_samples: Optional[int] = None,
    fixed_icl_contexts: Optional[Mapping[PartitionKey, List[Dict[str, Any]]]] = None,
    query_gt_demo: Optional[str] = None,
    demo_pool: Optional[Dict] = None,
) -> Iterator[ScoringQuery]:
    """Yield ScoringQuery objects for each data entry, optionally with ICL context.

    ICL contexts come from ``fixed_icl_contexts`` when given, otherwise they are drawn
    per query from ``demo_pool``.
    """
    use_fixed = icl and fixed_icl_contexts is not None
    if icl and not use_fixed and demo_pool is None:
        raise ValueError("ICL without fixed_icl_contexts needs a demo_pool")
    answer_rng = random.Random(seed + 1)
    if special and special.startswith("corrupt_icd") and demo_pool:
        label_spaces = build_label_space(demo_pool)
    else:
        label_spaces = None
    count = 0

    for entry in data:
        if max_samples and count >= max_samples:
            return

        image_obj = apply_special_mode(entry["image"], special, is_icd=False)
        image = resolve_image(image_obj, prefer_pil=prefer_pil) if image_obj is not None else None

        for question in entry.get("questions", []):
            partition_key = (entry["task"], entry["split"])
            if partition_key not in label_vocab_by_partition:
                continue

            score_prompt = make_score_prompt_prefix(question["prompt"], variant=brace_variant)

            if use_fixed:
                # Use pre-built fixed ICL context
                if partition_key not in fixed_icl_contexts:
                    continue
                if max_samples and count >= max_samples:
                    return

                yield ScoringQuery(
                    image_id=entry["image_id"],
                    task=entry["task"],
                    topic=entry.get("topic"),
                    split=entry["split"],
                    question_type=question["type"],
                    prompt=question["prompt"],
                    ground_truth=question["ground_truth"],
                    expected_bias=question.get("expected_bias"),
                    image=image,
                    context=fixed_icl_contexts[partition_key],
                    score_prompt=score_prompt,
                    partition_key=partition_key,
                )
                count += 1

            elif icl:
                context_samples = select_from_demo_pool(
                    entry, question, demo_pool, num_icl_examples, seed,
                    query_gt_position=query_gt_demo,
                    ensure_query_gt=(query_gt_demo is not None),
                )
                if context_samples is None:
                    continue

                if label_spaces is not None:
                    key = (entry["task"], entry["split"])
                    shuffled_answers = corrupt_icl_answers(context_samples, special, answer_rng, label_spaces[key])
                else:
                    shuffled_answers = shuffle_icl_answers(context_samples, special, answer_rng)
                icl_context = process_icl_context_images(
                    context_samples,
                    shuffled_answers,
                    special,
                    format_icl_answer,
                    prefer_pil=prefer_pil,
                )

                for icl_set_idx in range(num_icl_sample_sets):
                    if max_samples and count >= max_samples:
                        return

                    if icl_set_idx > 0:
                        set_seed = seed + icl_set_idx if seed is not None else None
                        context_samples = select_from_demo_pool(
                            entry, question, demo_pool, num_icl_examples, set_seed,
                            query_gt_position=query_gt_demo,
                            ensure_query_gt=(query_gt_demo is not None),
                        )
                        if context_samples is None:
                            continue
                        if label_spaces is not None:
                            key = (entry["task"], entry["split"])
                            shuffled_answers = corrupt_icl_answers(context_samples, special, answer_rng, label_spaces[key])
                        else:
                            shuffled_answers = shuffle_icl_answers(context_samples, special, answer_rng)
                        icl_context = process_icl_context_images(
                            context_samples,
                            shuffled_answers,
                            special,
                            format_icl_answer,
                            prefer_pil=prefer_pil,
                        )

                    yield ScoringQuery(
                        image_id=entry["image_id"],
                        task=entry["task"],
                        topic=entry.get("topic"),
                        split=entry["split"],
                        question_type=question["type"],
                        prompt=question["prompt"],
                        ground_truth=question["ground_truth"],
                        expected_bias=question.get("expected_bias"),
                        image=image,
                        context=icl_context,
                        score_prompt=score_prompt,
                        partition_key=partition_key,
                    )
                    count += 1
            else:
                if max_samples and count >= max_samples:
                    return

                yield ScoringQuery(
                    image_id=entry["image_id"],
                    task=entry["task"],
                    topic=entry.get("topic"),
                    split=entry["split"],
                    question_type=question["type"],
                    prompt=question["prompt"],
                    ground_truth=question["ground_truth"],
                    expected_bias=question.get("expected_bias"),
                    image=image,
                    context=None,
                    score_prompt=score_prompt,
                    partition_key=partition_key,
                )
                count += 1

@dataclass
class AggregationState:
    """Streaming aggregation of distributions per ground-truth label."""

    label_order: List[str]
    sum_by_gt: Dict[str, np.ndarray] = field(default_factory=dict)
    count_by_gt: Dict[str, int] = field(default_factory=dict)
    
    def update(self, ground_truth: str, distribution: np.ndarray) -> None:
        """Add a sample's distribution to the running totals."""
        if ground_truth not in self.sum_by_gt:
            self.sum_by_gt[ground_truth] = np.zeros_like(distribution, dtype=np.float32)
            self.count_by_gt[ground_truth] = 0
        self.sum_by_gt[ground_truth] += distribution
        self.count_by_gt[ground_truth] += 1
    
    def compute_mean_by_gt(self) -> Dict[str, np.ndarray]:
        """Compute mean distribution for each ground truth."""
        return {gt: self.sum_by_gt[gt] / self.count_by_gt[gt] 
                for gt in self.sum_by_gt}
    
    def compute_macro_average(self) -> np.ndarray:
        """Compute macro-averaged overall distribution."""
        mean_by_gt = self.compute_mean_by_gt()
        if not mean_by_gt:
            return np.zeros((len(self.label_order),), dtype=np.float32)
        return np.mean(list(mean_by_gt.values()), axis=0)


def analyze_answer_distributions(config: AnalysisConfig) -> Dict[str, Any]:
    """Analyze answer distributions for a given analysis configuration."""
    
    # Validate
    validate_special_mode(config.special, config.icl, config.num_icl_examples)
    
    # Load data
    data = prepare_data(
        tasks=config.tasks, topics=config.topics,
        verified_only=config.verified_only,
        sample_per_split=config.sample_per_split,
        seed=config.seed,
        data_root=config.data_root,
        data_source=config.data_source,
    )

    # When sample_per_split is used with ICL, load full dataset for ICL pools
    icl_pool_data = None
    if config.icl and config.sample_per_split is not None:
        icl_pool_data = prepare_data(
            tasks=config.tasks, topics=config.topics,
            verified_only=config.verified_only,
            seed=config.seed,
            data_root=config.data_root,
            data_source=config.data_source,
        )

    # Build demo pool for ICL
    demo_pool = None
    if config.icl:
        pool_source = icl_pool_data if icl_pool_data is not None else data
        context_pools = build_icl_context_pools(pool_source)
        demo_pool = build_demo_pool(context_pools, seed=config.seed)

    # Build label vocabularies per partition
    label_vocab_by_partition = build_label_vocab_by_partition(
        data, include_expected_bias=config.include_expected_bias
    )

    # Create scorer
    scorer_kwargs = {"model_path": config.model_path, "cot": config.cot, "enable_thinking": config.enable_thinking}
    engine_kwargs = {"tensor_parallel_size": config.tensor_parallel_size,
                     "max_num_images": config.max_num_images, "max_model_len": config.max_model_len}
    if config.backend == "vllm":
        scorer = VLLMAnswerScorer(**scorer_kwargs, **engine_kwargs)
    else:
        scorer = HFAnswerScorer(**scorer_kwargs)

    # Aggregate by partition
    aggregates = {pk: AggregationState(label_order=list(labels))
                  for pk, labels in label_vocab_by_partition.items()}

    # Score queries
    prefer_pil = config.backend != "vllm"
    queries = iter_queries(data=data, label_vocab_by_partition=label_vocab_by_partition,
                           icl=config.icl, num_icl_examples=config.num_icl_examples,
                           num_icl_sample_sets=config.num_icl_sample_sets, seed=config.seed,
                           special=config.special, cot=config.cot, brace_variant=config.brace_variant,
                           prefer_pil=prefer_pil, max_samples=config.max_samples,
                           query_gt_demo=config.query_gt_demo,
                           demo_pool=demo_pool)
    for query in tqdm(queries, desc="Scoring samples", total=config.max_samples):
        
        agg = aggregates[query.partition_key]
        label_order = agg.label_order
        
        # Score answers
        result = scorer.score_answers(
            image=query.image,
            prompt=query.prompt,
            labels=label_order,
            context=query.context,
            **({"logprobs_k": config.vllm_logprobs_k} if isinstance(scorer, VLLMAnswerScorer) else {}),
        )

        # Convert to distribution
        logp_vec = np.array([result["label_logprobs"].get(l, float('-inf')) for l in label_order], dtype=np.float32)
        if np.isneginf(logp_vec).all():
            continue
        dist = softmax(logp_vec)
        
        # Update aggregates
        gt_norm = normalize_answer(query.ground_truth)
        agg.update(gt_norm, dist)
    
    # Save artifacts per task
    output_dirs = []
    for pk, agg in aggregates.items():
        task_name, split = pk
        output_dir = build_analysis_output_dir(config, task_name=task_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "run_config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)

        partition_dir = output_dir / split
        mean_by_gt = agg.compute_mean_by_gt()
        total_samples = int(sum(agg.count_by_gt.values()))

        save_label_order(partition_dir, label_order=agg.label_order)
        save_by_ground_truth(
            partition_dir,
            by_ground_truth={
                gt: {
                    "count": int(agg.count_by_gt[gt]),
                    "mean_distribution": mean_by_gt[gt].tolist(),
                }
                for gt in agg.sum_by_gt
            },
        )
        save_overall(
            partition_dir,
            overall={
                "macro_mean_distribution": agg.compute_macro_average().tolist(),
                "n_labels": len(agg.label_order),
                "n_samples": total_samples,
                "partition": {"task": task_name, "split": split},
                "config": asdict(config),
            },
        )
        output_dirs.append(str(output_dir))

    return {"output_dirs": output_dirs}

   


def build_analysis_output_dir(config: AnalysisConfig, task_name: str) -> Path:
    """Build output directory for this analysis run."""
    return build_output_path(
        base_dir=config.output_dir or "./output",
        analysis_type="answer_dist",
        model_path=config.model_path,
        task_name=task_name,
        icl=config.icl,
        num_icl_examples=config.num_icl_examples,
        run_config=derive_run_config(config.special, config.cot),
        query_gt_demo=config.query_gt_demo,
        enable_thinking=config.enable_thinking,
    )


def save_label_order(output_dir: Path, *, label_order: Sequence[str]) -> None:
    """Save label_order.json to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "label_order.json", "w") as f:
        json.dump(list(label_order), f, indent=2)


def save_by_ground_truth(
    output_dir: Path,
    *,
    by_ground_truth: Mapping[str, Mapping[str, Any]],
) -> None:
    """Save by_ground_truth.json to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "by_ground_truth.json", "w") as f:
        json.dump(dict(by_ground_truth), f, indent=2)


def save_overall(output_dir: Path, *, overall: Mapping[str, Any]) -> None:
    """Save overall.json to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "overall.json", "w") as f:
        json.dump(dict(overall), f, indent=2)


def parse_args(argv: Optional[Sequence[str]] = None) -> AnalysisConfig:
    """Parse CLI args into AnalysisConfig."""
    from vlm_icl.args import add_model_args, add_data_args, add_icl_args, add_run_config_args

    parser = argparse.ArgumentParser(description="Analyze answer distributions at the '{answer}' location.")
    add_model_args(parser)
    add_data_args(parser)
    add_icl_args(parser)
    add_run_config_args(parser)

    # Script-specific
    parser.add_argument("--verified_only", action="store_true", help="Use only verified samples")
    parser.add_argument("--unverified_only", action="store_true", help="Use only unverified samples")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap number of scored queries (debug)")
    parser.add_argument(
        "--include_expected_bias",
        action="store_true",
        help="Include expected_bias labels in candidate label set (default: enabled)",
    )
    parser.add_argument(
        "--no_include_expected_bias",
        action="store_true",
        help="Disable including expected_bias labels",
    )
    parser.add_argument(
        "--brace_variant",
        type=str,
        default="{",
        choices=["{", "{ "],
        help="Use '{' or '{ ' as scoring prefix suffix",
    )
    parser.add_argument("--vllm_logprobs_k", type=int, default=20, help="vLLM top-K logprobs (first token)")

    args = parser.parse_args(argv)

    if args.verified_only and args.unverified_only:
        raise ValueError("Choose at most one of --verified_only / --unverified_only")

    verified_only: Optional[bool]
    if args.verified_only:
        verified_only = True
    elif args.unverified_only:
        verified_only = False
    else:
        verified_only = None

    include_expected_bias = True
    if args.no_include_expected_bias:
        include_expected_bias = False
    elif args.include_expected_bias:
        include_expected_bias = True

    return AnalysisConfig(
        model_path=args.model_path,
        backend=args.backend,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_images=args.max_num_images,
        max_model_len=args.max_model_len,
        seed=args.seed,
        output_dir=args.output_dir,
        tasks=args.tasks,
        topics=args.topics,
        verified_only=verified_only,
        sample_per_split=args.sample_per_split,
        icl=args.icl,
        num_icl_examples=args.num_icl_examples,
        num_icl_sample_sets=args.num_icl_sample_sets,
        special=args.special,
        cot=args.cot,
        enable_thinking=args.enable_thinking,
        query_gt_demo=args.query_gt_demo,
        max_samples=args.max_samples,
        include_expected_bias=include_expected_bias,
        brace_variant=args.brace_variant,
        vllm_logprobs_k=args.vllm_logprobs_k,
        data_root=args.data_root,
        data_source=args.data_source,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint: parse args, run analysis, print output path."""
    config = parse_args(argv)
    validate_special_mode(config.special, config.icl, config.num_icl_examples)
    result = analyze_answer_distributions(config)
    for d in result['output_dirs']:
        print(f"Saved answer distribution artifacts to: {d}")


if __name__ == "__main__":
    main()
