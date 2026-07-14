from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from examples.benchmark_common import (
    DEFAULT_VLLM_BASE_URL,
    BenchmarkSpec,
    RunnerConfig,
    ScoreResult,
    ValidatedSnapshot,
    run_benchmark,
)

CANONICAL_CONTEXT_LENGTH = 32768
PAIR_PATTERN = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
QUESTION_INSTRUCTION = (
    "The context contains general-knowledge questions, one per line, each with "
    "a User ID. Infer the TREC coarse category of each question and answer the "
    "pairwise aggregate question. Return every matching pair in the form "
    "(id1, id2), with id1 < id2."
)


def parse_pairs(prediction: str) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for match in PAIR_PATTERN.finditer(prediction):
        first, second = int(match.group(1)), int(match.group(2))
        if first == second:
            continue
        pairs.add((min(first, second), max(first, second)))
    return pairs


def score_pairs(
    gold_pairs: set[tuple[int, int]],
    prediction: str,
) -> ScoreResult:
    predicted_pairs = parse_pairs(prediction)
    if not gold_pairs and not predicted_pairs:
        return ScoreResult(
            score=1.0,
            precision=1.0,
            recall=1.0,
            parse_status="empty",
        )
    if not gold_pairs or not predicted_pairs:
        return ScoreResult(
            score=0.0,
            precision=0.0,
            recall=0.0,
            parse_status="parsed" if predicted_pairs else "no_valid_pairs",
        )
    true_positives = len(gold_pairs & predicted_pairs)
    precision = true_positives / len(predicted_pairs)
    recall = true_positives / len(gold_pairs)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return ScoreResult(
        score=f1,
        precision=precision,
        recall=recall,
        parse_status="parsed" if predicted_pairs else "no_valid_pairs",
    )


def build_root_prompt(question: str) -> str:
    return f"{QUESTION_INSTRUCTION}\n\nQuestion: {question}"


def build_plain_prompt(example: dict[str, Any]) -> str:
    return f"{build_root_prompt(str(example['question']))}\n\nContext:\n{example['context']}"


def build_rlm_inputs(example: dict[str, Any]) -> tuple[str, str]:
    return str(example["context"]), build_root_prompt(str(example["question"]))


def load_examples(
    snapshot: ValidatedSnapshot,
    filters: Mapping[str, Any],
) -> list[dict[str, Any]]:
    context_length = int(filters["context_length"])
    contexts_by_id: dict[str, dict[str, Any]] = {}
    for context in snapshot.contexts:
        context_id = str(context["id"])
        if context_id in contexts_by_id:
            raise ValueError(f"OOLONG-Pairs snapshot contains duplicate context ID: {context_id}")
        contexts_by_id[context_id] = context

    loaded: list[dict[str, Any]] = []
    for example in snapshot.iter_examples():
        example_context_length = int(example.get("context_len", -1))
        if example_context_length > context_length:
            break
        if example_context_length != context_length:
            continue
        context_id = str(example.get("context_id", ""))
        context = contexts_by_id.get(context_id)
        if context is None:
            raise ValueError(
                f"OOLONG-Pairs example {example.get('id')} references missing context {context_id}"
            )
        gold_value = example.get("gold_pairs")
        if not isinstance(gold_value, list):
            raise ValueError(f"OOLONG-Pairs example {example.get('id')} has invalid gold_pairs")
        gold_pairs: set[tuple[int, int]] = set()
        for pair in gold_value:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(value, int) for value in pair)
                or pair[0] >= pair[1]
            ):
                raise ValueError(
                    f"OOLONG-Pairs example {example.get('id')} has invalid gold pair {pair!r}"
                )
            gold_pairs.add((pair[0], pair[1]))
        loaded.append(
            {
                **example,
                "gold_pair_set": gold_pairs,
                "context": str(context["context"]),
            }
        )
    return loaded


def score_prediction(example: dict[str, Any], prediction: str) -> ScoreResult:
    return score_pairs(example["gold_pair_set"], prediction)


def gold_answer(example: dict[str, Any]) -> str:
    return json.dumps(example["gold_pairs"], separators=(",", ":"))


def is_canonical(filters: Mapping[str, Any]) -> bool:
    return filters.get("context_length") == CANONICAL_CONTEXT_LENGTH


SPEC = BenchmarkSpec(
    benchmark="oolong_pairs",
    dataset_name="OOLONG-Pairs",
    scoring_method="set precision/recall/F1 over canonical integer pairs",
    load_examples=load_examples,
    build_plain_prompt=build_plain_prompt,
    build_rlm_inputs=build_rlm_inputs,
    score=score_prediction,
    gold_answer=gold_answer,
    is_canonical=is_canonical,
    canonical_example_count=20,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local OOLONG-Pairs paper slice against local vLLM."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=DEFAULT_VLLM_BASE_URL)
    parser.add_argument("--data-dir", type=Path, default=Path("./data/benchmarks"))
    parser.add_argument("--log-dir", type=Path, default=Path("./logs"))
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-length", type=int, default=CANONICAL_CONTEXT_LENGTH)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-tokens", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_benchmark(
        SPEC,
        RunnerConfig(
            model=args.model,
            base_url=args.base_url,
            data_dir=args.data_dir,
            log_dir=args.log_dir,
            num_examples=args.num_examples,
            seed=args.seed,
            max_depth=args.max_depth,
            max_iterations=args.max_iterations,
            max_tokens=args.max_tokens,
            filters={"context_length": args.context_length},
        ),
    )
    print(f"plain: {summary['scores']['plain']:.4f}")
    print(f"rlm: {summary['scores']['rlm']:.4f}")
    print(f"CSV: {summary['csv_path']}")
    print(f"JSON: {Path(summary['csv_path']).with_suffix('.json')}")


if __name__ == "__main__":
    main()
