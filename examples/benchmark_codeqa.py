from __future__ import annotations

import argparse
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

FINAL_ANSWER_PATTERN = re.compile(
    r"\bfinal\s+answer\s*(?:is\s*)?[:=-]?\s*[\[(]?\s*([A-D])\s*[\])]?",
    re.IGNORECASE,
)
ANSWER_PATTERN = re.compile(
    r"\banswer\s*(?:is\s*)?[:=-]?\s*[\[(]?\s*([A-D])\s*[\])]?",
    re.IGNORECASE,
)
CHOICE_TOKEN_PATTERN = re.compile(r"\b([A-D])\b", re.IGNORECASE)
STANDALONE_PATTERN = re.compile(
    r"\s*(?:\*\*)?[\[(]?\s*([A-D])\s*[\])]?(?:\*\*)?[.!]?\s*",
    re.IGNORECASE,
)
QUESTION_INSTRUCTION = (
    "Answer the multiple-choice code repository question using the supplied "
    "context. Return one final choice from A, B, C, or D in the form "
    "'Final answer: X'."
)


def unique_explicit_choices(pattern: re.Pattern[str], prediction: str) -> set[str]:
    return {match.group(1).upper() for match in pattern.finditer(prediction)}


def parse_choice(prediction: str) -> tuple[str | None, str]:
    final_choices = unique_explicit_choices(FINAL_ANSWER_PATTERN, prediction)
    if len(final_choices) == 1:
        return next(iter(final_choices)), "explicit"
    if len(final_choices) > 1:
        return None, "ambiguous"

    answer_choices = unique_explicit_choices(ANSWER_PATTERN, prediction)
    if len(answer_choices) == 1:
        return next(iter(answer_choices)), "explicit"
    if len(answer_choices) > 1:
        return None, "ambiguous"

    standalone = STANDALONE_PATTERN.fullmatch(prediction)
    if standalone is not None:
        return standalone.group(1).upper(), "standalone"
    token_choices = {match.group(1).upper() for match in CHOICE_TOKEN_PATTERN.finditer(prediction)}
    if len(token_choices) > 1:
        return None, "ambiguous"
    return None, "unparseable"


def score_choice(gold_choice: str, prediction: str) -> ScoreResult:
    choice, parse_status = parse_choice(prediction)
    return ScoreResult(
        score=1.0 if choice == gold_choice else 0.0,
        parse_status=parse_status,
    )


def choices_text(example: Mapping[str, Any]) -> str:
    choices = example["choices"]
    return "\n".join(f"{choice}. {choices[choice]}" for choice in ("A", "B", "C", "D"))


def build_root_prompt(example: Mapping[str, Any]) -> str:
    return (
        f"{QUESTION_INSTRUCTION}\n\nQuestion: {example['question']}\n\n"
        f"Choices:\n{choices_text(example)}"
    )


def build_plain_prompt(example: dict[str, Any]) -> str:
    return f"{build_root_prompt(example)}\n\nContext:\n{example['context']}"


def build_rlm_inputs(example: dict[str, Any]) -> tuple[str, str]:
    return str(example["context"]), build_root_prompt(example)


def load_examples(
    snapshot: ValidatedSnapshot,
    filters: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if filters:
        raise ValueError(f"CodeQA does not accept benchmark filters: {dict(filters)!r}")
    required = {
        "id",
        "question",
        "choices",
        "gold_choice",
        "context",
        "difficulty",
        "length",
        "domain",
        "sub_domain",
    }
    loaded: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for example in snapshot.iter_examples():
        missing = sorted(required - example.keys())
        if missing:
            raise ValueError(f"CodeQA example is missing fields: {', '.join(missing)}")
        example_id = str(example["id"])
        if example_id in seen_ids:
            raise ValueError(f"CodeQA snapshot contains duplicate example ID: {example_id}")
        seen_ids.add(example_id)
        choices = example["choices"]
        if not isinstance(choices, dict) or set(choices) != {"A", "B", "C", "D"}:
            raise ValueError(f"CodeQA example {example_id} must contain choices A-D")
        if example["gold_choice"] not in {"A", "B", "C", "D"}:
            raise ValueError(f"CodeQA example {example_id} has invalid gold choice")
        loaded.append(dict(example))
    return loaded


def score_prediction(example: dict[str, Any], prediction: str) -> ScoreResult:
    return score_choice(str(example["gold_choice"]), prediction)


def gold_answer(example: dict[str, Any]) -> str:
    return str(example["gold_choice"])


def is_canonical(filters: Mapping[str, Any]) -> bool:
    return not filters


SPEC = BenchmarkSpec(
    benchmark="codeqa",
    dataset_name="LongBench-v2 CodeQA",
    scoring_method="strict parsed multiple-choice accuracy",
    load_examples=load_examples,
    build_plain_prompt=build_plain_prompt,
    build_rlm_inputs=build_rlm_inputs,
    score=score_prediction,
    gold_answer=gold_answer,
    is_canonical=is_canonical,
    canonical_example_count=50,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local LongBench-v2 CodeQA paper slice against local vLLM."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=DEFAULT_VLLM_BASE_URL)
    parser.add_argument("--data-dir", type=Path, default=Path("./data/benchmarks"))
    parser.add_argument("--log-dir", type=Path, default=Path("./logs"))
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--seed", type=int, default=42)
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
            filters={},
        ),
    )
    print(f"plain: {summary['scores']['plain']:.4f}")
    print(f"rlm: {summary['scores']['rlm']:.4f}")
    print(f"CSV: {summary['csv_path']}")
    print(f"JSON: {Path(summary['csv_path']).with_suffix('.json')}")


if __name__ == "__main__":
    main()
