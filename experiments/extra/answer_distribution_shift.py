"""Compare top-k token distributions at the answer position between zero-shot and ICL settings."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from tqdm import tqdm

from experiments.extra.answer_distribution import (
    AnalysisConfig,
    ScoringQuery,
    iter_queries,
)
from vlm_icl.scoring import (
    BraceVariant,
    HFAnswerScorer,
    PartitionKey,
    VLLMAnswerScorer,
    build_label_vocab_by_partition,
    tokenize_label_after_brace,
)
from vlm_icl.data import prepare_data
from vlm_icl.prompts import format_icl_answer
from vlm_icl.icl import (
    build_demo_pool,
    build_icl_context_pools,
    process_icl_context_images,
    shuffle_icl_answers,
    validate_special_mode,
)
from vlm_icl.paths import build_output_path, derive_run_config
from vlm_icl.metrics import normalize_answer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShiftConfig:
    """Configuration for answer distribution shift analysis."""

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

    num_icl_examples: int = 2
    num_icl_sample_sets: int = 1

    special: Optional[str] = None
    cot: bool = False
    enable_thinking: bool = False
    max_samples: Optional[int] = None

    include_expected_bias: bool = True
    brace_variant: BraceVariant = "{"
    vllm_logprobs_k: int = 20

    top_k: int = 20
    data_root: Optional[str] = None
    data_source: str = "auto"


# ---------------------------------------------------------------------------
# Label-to-first-token mapping
# ---------------------------------------------------------------------------

def build_label_to_first_token(
    label_vocab_by_partition: Mapping[PartitionKey, Sequence[str]],
    tokenizer: Any,
    brace_variant: BraceVariant,
) -> Dict[PartitionKey, Dict[str, Tuple[int, str]]]:
    """Map each label to its first token (token_id, token_str) per partition."""
    result: Dict[PartitionKey, Dict[str, Tuple[int, str]]] = {}
    for pk, labels in label_vocab_by_partition.items():
        mapping: Dict[str, Tuple[int, str]] = {}
        for label in labels:
            token_ids = tokenize_label_after_brace(tokenizer, label=label, brace_variant=brace_variant)
            first_id = token_ids[0]
            first_str = tokenizer.decode([first_id])
            mapping[label] = (first_id, first_str)
        result[pk] = mapping
    return result


# ---------------------------------------------------------------------------
# Fixed ICL demo sampling
# ---------------------------------------------------------------------------

def sample_fixed_icl_demos(
    demo_pool: Dict[Tuple, List[Dict]],
    num_sets: int,
    num_examples: int,
    base_seed: int,
) -> Dict[PartitionKey, List[List[Dict]]]:
    """Pre-sample fixed ICL demo sets per partition for reproducible comparisons."""
    result: Dict[PartitionKey, List[List[Dict]]] = {}

    for pk, pool in demo_pool.items():
        sets: List[List[Dict]] = []

        # Group by ground_truth for balanced sampling
        gt_groups: Dict[str, List[Dict]] = defaultdict(list)
        for item in pool:
            gt_groups[item["question"]["ground_truth"]].append(item)

        for i in range(num_sets):
            rng = random.Random(base_seed + i)

            unique_gts = list(gt_groups.keys())
            rng.shuffle(unique_gts)

            n_unique = len(unique_gts)
            base = num_examples // n_unique
            remainder = num_examples % n_unique

            selected: List[Dict] = []
            for j, gt in enumerate(unique_gts):
                group = gt_groups[gt]
                n = min(base + (1 if j < remainder else 0), len(group))
                # Use rng.sample for reproducibility
                selected.extend(rng.sample(group, n))

            # Fallback: fill deficit from remaining candidates
            if len(selected) < num_examples:
                selected_ids = {id(x) for x in selected}
                remaining = [c for c in pool if id(c) not in selected_ids]
                deficit = num_examples - len(selected)
                if len(remaining) >= deficit:
                    selected.extend(rng.sample(remaining, deficit))

            rng.shuffle(selected)
            sets.append(selected[:num_examples])

        result[pk] = sets

    return result


# ---------------------------------------------------------------------------
# Token classification
# ---------------------------------------------------------------------------

def classify_zs(
    top_k: List[Tuple[int, str, float]],
    gt_prob: Optional[float],
    mass_in_topk: float,
    gt_token_id: Optional[int],
    answer_token_ids: Set[int],
) -> Dict[str, float]:
    """Classify zero-shot top-k tokens into semantic groups (gt, other_answer, non_answer, residual)."""
    gt_p = gt_prob if gt_prob is not None else 0.0

    other_answer = 0.0
    non_answer = 0.0
    for tid, _, prob in top_k:
        if gt_token_id is not None and tid == gt_token_id:
            # gt probability is accounted for separately via gt_prob
            continue
        if tid in answer_token_ids:
            other_answer += prob
        else:
            non_answer += prob

    residual = max(0.0, 1.0 - mass_in_topk)

    return {
        "gt": gt_p,
        "other_answer": other_answer,
        "non_answer": non_answer,
        "residual": residual,
    }


def classify_icl(
    top_k: List[Tuple[int, str, float]],
    gt_prob: Optional[float],
    mass_in_topk: float,
    gt_token_id: Optional[int],
    answer_token_ids: Set[int],
    demo_answer_token_ids: Set[int],
) -> Dict[str, float]:
    """Classify ICL top-k tokens into semantic groups (gt, icl_demo_answer, other_answer, non_answer, residual)."""
    gt_p = gt_prob if gt_prob is not None else 0.0

    icl_demo_answer = 0.0
    other_answer = 0.0
    non_answer = 0.0
    for tid, _, prob in top_k:
        if gt_token_id is not None and tid == gt_token_id:
            continue
        if tid in demo_answer_token_ids:
            icl_demo_answer += prob
        elif tid in answer_token_ids:
            other_answer += prob
        else:
            non_answer += prob

    residual = max(0.0, 1.0 - mass_in_topk)

    return {
        "gt": gt_p,
        "icl_demo_answer": icl_demo_answer,
        "other_answer": other_answer,
        "non_answer": non_answer,
        "residual": residual,
    }


# ---------------------------------------------------------------------------
# Bucket aggregator
# ---------------------------------------------------------------------------

class BucketAggregator:
    """Streaming sum/count per bucket for computing means."""

    def __init__(self, bucket_names: Sequence[str]) -> None:
        self.bucket_names = list(bucket_names)
        self.sums: Dict[str, float] = {b: 0.0 for b in self.bucket_names}
        self.count: int = 0

    def update(self, classified: Dict[str, float]) -> None:
        for b in self.bucket_names:
            self.sums[b] += classified.get(b, 0.0)
        self.count += 1

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"count": self.count}
        for b in self.bucket_names:
            result[f"mean_{b}"] = self.sums[b] / self.count if self.count > 0 else 0.0
        return result


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def _get_tokenizer(scorer: Any) -> Any:
    """Extract tokenizer from either HF or vLLM scorer."""
    if isinstance(scorer, HFAnswerScorer):
        return scorer.processor.tokenizer
    return scorer.tokenizer


def _serialize_demo_set(demo_set: List[Dict], seed: int) -> Dict[str, Any]:
    """Serialize a demo set to lightweight JSON-safe references."""
    demos = []
    for item in demo_set:
        entry = item["entry"]
        question = item["question"]
        demos.append({
            "image_id": entry["image_id"],
            "task": entry["task"],
            "split": entry["split"],
            "topic": entry.get("topic"),
            "prompt": question["prompt"],
            "ground_truth": question["ground_truth"],
        })
    return {"demos": demos, "seed": seed}


def run_shift_analysis(config: ShiftConfig) -> Dict[str, Any]:
    """Run the full shift analysis: zero-shot + ICL passes."""

    validate_special_mode(config.special, True, config.num_icl_examples)

    # Step 1-2: Load data
    data = prepare_data(
        tasks=config.tasks,
        topics=config.topics,
        verified_only=config.verified_only,
        sample_per_split=config.sample_per_split,
        seed=config.seed,
        data_root=config.data_root,
        data_source=config.data_source,
    )

    # When sample_per_split is used, load full dataset for ICL context pools
    icl_pool_data = None
    if config.sample_per_split is not None:
        icl_pool_data = prepare_data(
            tasks=config.tasks,
            topics=config.topics,
            verified_only=config.verified_only,
            seed=config.seed,
            data_root=config.data_root,
            data_source=config.data_source,
        )

    # Step 3: Build label vocab per partition
    label_vocab_by_partition = build_label_vocab_by_partition(
        data, include_expected_bias=config.include_expected_bias
    )

    # Step 4: Create scorer
    scorer_kwargs = {
        "model_path": config.model_path,
        "cot": config.cot,
        "enable_thinking": config.enable_thinking,
    }
    if config.backend == "vllm":
        scorer = VLLMAnswerScorer(
            **scorer_kwargs,
            tensor_parallel_size=config.tensor_parallel_size,
            max_num_images=config.max_num_images,
            max_model_len=config.max_model_len,
        )
    else:
        scorer = HFAnswerScorer(**scorer_kwargs)

    tokenizer = _get_tokenizer(scorer)

    # Step 5: Build label-to-first-token mapping
    label_to_first_token = build_label_to_first_token(
        label_vocab_by_partition, tokenizer, config.brace_variant
    )

    # Build answer_token_ids set per partition (all answer label first-tokens)
    answer_token_ids_by_pk: Dict[PartitionKey, Set[int]] = {}
    for pk, mapping in label_to_first_token.items():
        answer_token_ids_by_pk[pk] = {tid for tid, _ in mapping.values()}

    # Step 6: Build ICL context pools and demo pool (from full dataset if subsampled)
    context_pools = build_icl_context_pools(icl_pool_data if icl_pool_data is not None else data)
    demo_pool = build_demo_pool(context_pools, seed=config.seed)

    # Step 7: Sample fixed demos
    fixed_demos = sample_fixed_icl_demos(
        demo_pool, config.num_icl_sample_sets, config.num_icl_examples, config.seed
    )

    # Step 8: Preprocess each demo set's ICL context + precompute demo answer sets
    prefer_pil = config.backend != "vllm"
    answer_rng = random.Random(config.seed + 1)

    # fixed_contexts[pk][i] = preprocessed ICL context list
    fixed_contexts: Dict[PartitionKey, List[List[Dict[str, Any]]]] = {}
    # demo_answer_labels[pk][i] = set of normalized ground_truth labels in demos
    demo_answer_labels: Dict[PartitionKey, List[Set[str]]] = {}
    # demo_image_ids[pk][i] = set of image_ids in demos (for query_in_demo check)
    demo_image_ids: Dict[PartitionKey, List[Set[str]]] = {}

    for pk, demo_sets in fixed_demos.items():
        fixed_contexts[pk] = []
        demo_answer_labels[pk] = []
        demo_image_ids[pk] = []

        for demo_set in demo_sets:
            shuffled_answers = shuffle_icl_answers(demo_set, config.special, answer_rng)
            icl_context = process_icl_context_images(
                demo_set,
                shuffled_answers,
                config.special,
                format_icl_answer,
                prefer_pil=prefer_pil,
            )
            fixed_contexts[pk].append(icl_context)

            # Precompute demo answer label set
            labels = set()
            img_ids = set()
            for item in demo_set:
                labels.add(normalize_answer(item["question"]["ground_truth"]))
                img_ids.add(item["entry"]["image_id"])
            demo_answer_labels[pk].append(labels)
            demo_image_ids[pk].append(img_ids)

    # Build output directory (use ICL config for timestamped path)
    output_dirs: List[str] = []
    # Get unique task names from data
    task_names = sorted({entry["task"] for entry in data})

    run_config = derive_run_config(config.special, config.cot)
    for task_name in task_names:
        out_dir = build_output_path(
            base_dir=config.output_dir or "./output",
            analysis_type="answer_shift",
            model_path=config.model_path,
            task_name=task_name,
            icl=True,
            num_icl_examples=config.num_icl_examples,
            run_config=run_config,
            enable_thinking=config.enable_thinking,
        )
        output_dirs.append(str(out_dir))

    # Helper to get output dir for a task
    def _out_dir_for_task(task: str) -> Path:
        idx = task_names.index(task)
        return Path(output_dirs[idx])

    # -----------------------------------------------------------------------
    # Zero-shot pass (Step 9)
    # -----------------------------------------------------------------------
    ZS_BUCKETS = ["gt", "other_answer", "non_answer", "residual"]
    zs_agg: Dict[PartitionKey, BucketAggregator] = {}

    # Initialize aggregators
    for pk in label_to_first_token:
        zs_agg[pk] = BucketAggregator(ZS_BUCKETS)

    # Open JSONL writers per partition
    zs_writers: Dict[PartitionKey, Path] = {}
    for pk in label_to_first_token:
        task, split = pk
        out = _out_dir_for_task(task) / split / "zero_shot"
        out.mkdir(parents=True, exist_ok=True)
        zs_writers[pk] = out / "per_entry.jsonl"
        # Truncate existing file
        zs_writers[pk].write_text("")

    print("=" * 60)
    print("Zero-shot pass")
    print("=" * 60)

    queries = iter_queries(
        data=data,
        label_vocab_by_partition=label_vocab_by_partition,
        icl=False,
        num_icl_examples=0,
        num_icl_sample_sets=1,
        seed=config.seed,
        special=config.special,
        cot=config.cot,
        brace_variant=config.brace_variant,
        prefer_pil=prefer_pil,
        max_samples=config.max_samples,
    )

    for query in tqdm(queries, desc="Zero-shot scoring", total=config.max_samples):
        pk = query.partition_key
        mapping = label_to_first_token[pk]
        all_answer_tids = answer_token_ids_by_pk[pk]

        gt_norm = normalize_answer(query.ground_truth)
        gt_token_id = mapping[gt_norm][0] if gt_norm in mapping else None

        # Score answers and extract top-k at answer position
        result = scorer.score_answers(
            image=query.image,
            prompt=query.prompt,
            labels=list(label_vocab_by_partition.get(pk, [])),
            context=query.context,
            **({"logprobs_k": config.vllm_logprobs_k} if isinstance(scorer, VLLMAnswerScorer) else {}),
        )
        top_k = result["top_k_at_answer"][:config.top_k]
        gt_prob = next((p for tid, _, p in top_k if tid == gt_token_id), None) if gt_token_id is not None else None
        mass_in_topk = result["mass_in_topk"]

        classified = classify_zs(
            top_k, gt_prob, mass_in_topk, gt_token_id, all_answer_tids
        )
        zs_agg[pk].update(classified)

        # Model output = argmax token string
        model_output = top_k[0][1] if top_k else ""

        # Per-entry record
        record = {
            "image_id": query.image_id,
            "task": query.task,
            "split": query.split,
            "question_type": query.question_type,
            "ground_truth": query.ground_truth,
            "prompt": query.prompt,
            "model_output": model_output,
            "top_k_tokens": [(tid, tok, round(p, 6)) for tid, tok, p in top_k],
            "classified_probs": {k: round(v, 6) for k, v in classified.items()},
            "gt_prob": round(gt_prob, 6) if gt_prob is not None else None,
            "mass_in_topk": round(mass_in_topk, 6),
        }
        if "generated_text" in result:
            record["generated_text"] = result["generated_text"]
        with open(zs_writers[pk], "a") as f:
            f.write(json.dumps(record) + "\n")

    # Save zero-shot aggregated
    for pk, agg in zs_agg.items():
        task, split = pk
        out = _out_dir_for_task(task) / split / "zero_shot"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "aggregated.json", "w") as f:
            json.dump(agg.to_dict(), f, indent=2)

    # -----------------------------------------------------------------------
    # ICL passes (Step 10)
    # -----------------------------------------------------------------------
    ICL_BUCKETS = ["gt", "icl_demo_answer", "other_answer", "non_answer", "residual"]

    for set_idx in range(config.num_icl_sample_sets):
        print(f"\n{'=' * 60}")
        print(f"ICL pass — set {set_idx}")
        print(f"{'=' * 60}")

        # Build fixed_icl_contexts for this set
        icl_ctx_for_set: Dict[PartitionKey, List[Dict[str, Any]]] = {}
        for pk in fixed_contexts:
            if set_idx < len(fixed_contexts[pk]):
                icl_ctx_for_set[pk] = fixed_contexts[pk][set_idx]

        # Aggregators per category
        icl_agg_in: Dict[PartitionKey, BucketAggregator] = {}
        icl_agg_out: Dict[PartitionKey, BucketAggregator] = {}

        for pk in label_to_first_token:
            icl_agg_in[pk] = BucketAggregator(ICL_BUCKETS)
            icl_agg_out[pk] = BucketAggregator(ICL_BUCKETS)

        # Open JSONL writers per partition per category
        writers_in: Dict[PartitionKey, Path] = {}
        writers_out: Dict[PartitionKey, Path] = {}
        for pk in label_to_first_token:
            task, split = pk
            for cat, writers in [("gt_in_demo", writers_in), ("gt_not_in_demo", writers_out)]:
                out = _out_dir_for_task(task) / split / f"icl_set_{set_idx}" / cat
                out.mkdir(parents=True, exist_ok=True)
                p = out / "per_entry.jsonl"
                p.write_text("")
                writers[pk] = p

        queries = iter_queries(
            data=data,
            label_vocab_by_partition=label_vocab_by_partition,
            icl=True,
            num_icl_examples=config.num_icl_examples,
            num_icl_sample_sets=1,
            seed=config.seed,
            special=config.special,
            cot=config.cot,
            brace_variant=config.brace_variant,
            prefer_pil=prefer_pil,
            max_samples=config.max_samples,
            fixed_icl_contexts=icl_ctx_for_set,
        )

        for query in tqdm(queries, desc=f"ICL set {set_idx} scoring", total=config.max_samples):
            pk = query.partition_key
            mapping = label_to_first_token[pk]
            all_answer_tids = answer_token_ids_by_pk[pk]

            gt_norm = normalize_answer(query.ground_truth)
            gt_token_id = mapping[gt_norm][0] if gt_norm in mapping else None

            # Determine category: gt_in_demo or gt_not_in_demo
            demo_labels = demo_answer_labels.get(pk, [{}] * (set_idx + 1))[set_idx]
            gt_in_demo = gt_norm in demo_labels

            # query_in_demo flag
            d_img_ids = demo_image_ids.get(pk, [set()] * (set_idx + 1))[set_idx]
            query_in_demo = query.image_id in d_img_ids

            # Build demo_answer_token_ids (excluding gt)
            demo_ans_tids: Set[int] = set()
            for label in demo_labels:
                if label != gt_norm and label in mapping:
                    demo_ans_tids.add(mapping[label][0])

            # Score answers and extract top-k at answer position
            result = scorer.score_answers(
                image=query.image,
                prompt=query.prompt,
                labels=list(label_vocab_by_partition.get(pk, [])),
                context=query.context,
                **({"logprobs_k": config.vllm_logprobs_k} if isinstance(scorer, VLLMAnswerScorer) else {}),
            )
            top_k = result["top_k_at_answer"][:config.top_k]
            gt_prob = next((p for tid, _, p in top_k if tid == gt_token_id), None) if gt_token_id is not None else None
            mass_in_topk = result["mass_in_topk"]

            classified = classify_icl(
                top_k, gt_prob, mass_in_topk, gt_token_id, all_answer_tids, demo_ans_tids
            )

            # Update correct aggregator
            if gt_in_demo:
                icl_agg_in[pk].update(classified)
            else:
                icl_agg_out[pk].update(classified)

            model_output = top_k[0][1] if top_k else ""

            record = {
                "image_id": query.image_id,
                "task": query.task,
                "split": query.split,
                "question_type": query.question_type,
                "ground_truth": query.ground_truth,
                "prompt": query.prompt,
                "model_output": model_output,
                "top_k_tokens": [(tid, tok, round(p, 6)) for tid, tok, p in top_k],
                "classified_probs": {k: round(v, 6) for k, v in classified.items()},
                "gt_prob": round(gt_prob, 6) if gt_prob is not None else None,
                "mass_in_topk": round(mass_in_topk, 6),
                "gt_in_demo": gt_in_demo,
                "query_in_demo": query_in_demo,
            }
            if "generated_text" in result:
                record["generated_text"] = result["generated_text"]

            writer = writers_in[pk] if gt_in_demo else writers_out[pk]
            with open(writer, "a") as f:
                f.write(json.dumps(record) + "\n")

        # Save ICL aggregated
        for pk in label_to_first_token:
            task, split = pk
            for cat, agg_dict in [("gt_in_demo", icl_agg_in), ("gt_not_in_demo", icl_agg_out)]:
                out = _out_dir_for_task(task) / split / f"icl_set_{set_idx}" / cat
                out.mkdir(parents=True, exist_ok=True)
                with open(out / "aggregated.json", "w") as f:
                    json.dump(agg_dict[pk].to_dict(), f, indent=2)

    # -----------------------------------------------------------------------
    # Save shared artifacts (Step 11)
    # -----------------------------------------------------------------------
    for pk in label_to_first_token:
        task, split = pk
        split_dir = _out_dir_for_task(task) / split

        # Save label_token_mapping.json
        mapping = label_to_first_token[pk]
        mapping_json = {
            label: {"token_id": tid, "token_str": tstr}
            for label, (tid, tstr) in mapping.items()
        }
        with open(split_dir / "label_token_mapping.json", "w") as f:
            json.dump(mapping_json, f, indent=2)

        # Save fix_icl_demos.json
        demo_sets = fixed_demos.get(pk, [])
        demo_json = [
            _serialize_demo_set(ds, config.seed + i)
            for i, ds in enumerate(demo_sets)
        ]
        with open(split_dir / "fix_icl_demos.json", "w") as f:
            json.dump(demo_json, f, indent=2)

    # Save run_config.json per split dir
    for pk in label_to_first_token:
        task, split = pk
        split_dir = _out_dir_for_task(task) / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with open(split_dir / "run_config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)

    return {"output_dirs": output_dirs}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> ShiftConfig:
    """Parse CLI args into ShiftConfig."""
    from vlm_icl.args import add_model_args, add_data_args, add_icl_args, add_run_config_args

    parser = argparse.ArgumentParser(
        description="Answer distribution shift analysis: zero-shot vs ICL top-k distributions."
    )
    add_model_args(parser)
    add_data_args(parser)
    add_icl_args(parser)
    add_run_config_args(parser)

    # Script-specific
    parser.add_argument("--verified_only", action="store_true", help="Use only verified samples")
    parser.add_argument("--unverified_only", action="store_true", help="Use only unverified samples")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap number of scored queries (debug)")
    parser.add_argument(
        "--include_expected_bias", action="store_true",
        help="Include expected_bias labels in candidate label set (default: enabled)",
    )
    parser.add_argument(
        "--no_include_expected_bias", action="store_true",
        help="Disable including expected_bias labels",
    )
    parser.add_argument(
        "--brace_variant", type=str, default="{",
        choices=["{", "{ "],
        help="Use '{' or '{ ' as scoring prefix suffix",
    )
    parser.add_argument("--vllm_logprobs_k", type=int, default=20, help="vLLM top-K logprobs (default: 20, max: 20)")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top tokens to extract (default: 20)")

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

    if args.backend == "vllm" and args.vllm_logprobs_k < args.top_k:
        raise ValueError(
            f"--vllm_logprobs_k ({args.vllm_logprobs_k}) must be >= --top_k ({args.top_k})"
        )

    return ShiftConfig(
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
        num_icl_examples=args.num_icl_examples,
        num_icl_sample_sets=args.num_icl_sample_sets,
        special=args.special,
        cot=args.cot,
        enable_thinking=args.enable_thinking,
        max_samples=args.max_samples,
        include_expected_bias=include_expected_bias,
        brace_variant=args.brace_variant,
        vllm_logprobs_k=args.vllm_logprobs_k,
        top_k=args.top_k,
        data_root=args.data_root,
        data_source=args.data_source,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint: parse args, run shift analysis, print output paths."""
    config = parse_args(argv)
    result = run_shift_analysis(config)
    for d in result["output_dirs"]:
        print(f"Saved shift analysis artifacts to: {d}")


if __name__ == "__main__":
    main()
