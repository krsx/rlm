from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
from datetime import datetime
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

COMPARISON_PHRASES = ("more common than", "less common than", "same frequency as")
QUESTION_INSTRUCTION = (
    "The context contains thousands of general-knowledge questions, one per "
    "line. Each line has a User ID and a question, and each question's answer "
    "falls into one of 6 categories: 'numeric value', 'entity', 'location', "
    "'description and abstract concept', 'abbreviation', 'human being'. "
    "Answer the following aggregate question."
)
CANONICAL_CONTEXT_LENGTH = 131072


def find_comparison_phrase(output: str) -> str | None:
    output_lower = output.lower()
    hits = [
        (output_lower.rfind(phrase), phrase)
        for phrase in COMPARISON_PHRASES
        if phrase in output_lower
    ]
    return max(hits)[1] if hits else None


def attempt_answer_parse(answer: str) -> tuple[str, str]:
    comparison_phrase = find_comparison_phrase(answer)
    if comparison_phrase is not None:
        return comparison_phrase, "high"
    if ":" not in answer:
        if len(answer) < 20:
            return answer, "low"
        return answer.split()[-1], "low"
    candidate = answer.split(":")[-1].strip().replace("*", "").replace("[", "").replace("]", "")
    if len(candidate) < 20:
        return candidate, "vhigh"
    return candidate, "med"


def synth_score(datapoint: Mapping[str, Any], output: str) -> float:
    answer = str(datapoint.get("answer", ""))
    try:
        if "datetime" in answer:
            gold: Any = datetime.strptime(answer, "[datetime.date(%Y, %m, %d)]")
        else:
            gold = ast.literal_eval(answer)[0]
    except (ValueError, SyntaxError, TypeError, IndexError):
        gold = answer

    trimmed, _confidence = attempt_answer_parse(output)
    gold_string = str(gold)
    if trimmed == gold_string or trimmed.lower() == gold_string.lower():
        return 1.0

    answer_type = datapoint.get("answer_type", "")
    if answer_type == "ANSWER_TYPE.NUMERIC":
        try:
            return 0.75 ** abs(int(gold) - int(trimmed))
        except (TypeError, ValueError):
            return 0.0
    if answer_type == "ANSWER_TYPE.DATE":
        try:
            import dateutil.parser

            return 1.0 if dateutil.parser.parse(trimmed) == gold else 0.0
        except (TypeError, ValueError, OverflowError):
            return 0.0

    if (
        gold_string
        and gold_string.lower() not in [phrase.lower() for phrase in COMPARISON_PHRASES]
        and gold_string.lower() in output.lower()
    ):
        return 1.0
    return 0.0


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
    exclude_numeric = bool(filters["exclude_numeric"])
    contexts_by_id: dict[str, dict[str, Any]] = {}
    for context in snapshot.contexts:
        context_id = str(context["id"])
        if context_id in contexts_by_id:
            raise ValueError(f"OOLONG snapshot contains duplicate context ID: {context_id}")
        contexts_by_id[context_id] = context

    loaded: list[dict[str, Any]] = []
    for example in snapshot.iter_examples():
        if (
            example.get("dataset") != "trec_coarse"
            or int(example.get("context_len", -1)) != context_length
            or (exclude_numeric and example.get("answer_type") == "ANSWER_TYPE.NUMERIC")
        ):
            continue
        context_id = str(example.get("context_id", ""))
        context = contexts_by_id.get(context_id)
        if context is None:
            raise ValueError(
                f"OOLONG example {example.get('id')} references missing context {context_id}"
            )
        loaded.append({**example, "context": str(context["context"])})
    return loaded


def score_prediction(example: dict[str, Any], prediction: str) -> ScoreResult:
    return ScoreResult(score=synth_score(example, prediction))


def gold_answer(example: dict[str, Any]) -> str:
    return str(example["answer"])


def is_canonical(filters: Mapping[str, Any]) -> bool:
    return (
        filters.get("context_length") == CANONICAL_CONTEXT_LENGTH
        and filters.get("exclude_numeric") is False
    )


SPEC = BenchmarkSpec(
    benchmark="oolong",
    dataset_name="trec_coarse",
    scoring_method="OOLONG synthetic answer score",
    load_examples=load_examples,
    build_plain_prompt=build_plain_prompt,
    build_rlm_inputs=build_rlm_inputs,
    score=score_prediction,
    gold_answer=gold_answer,
    is_canonical=is_canonical,
    canonical_example_count=50,
)


def require_dateutil() -> None:
    try:
        import dateutil.parser  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Missing OOLONG scoring dependency. Run through "
            "make benchmark-oolong MODEL=... or add --with python-dateutil."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local OOLONG paper slice against local vLLM."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=DEFAULT_VLLM_BASE_URL)
    parser.add_argument("--data-dir", type=Path, default=Path("./data/benchmarks"))
    parser.add_argument("--log-dir", type=Path, default=Path("./logs"))
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-length", type=int, default=CANONICAL_CONTEXT_LENGTH)
    parser.add_argument("--exclude-numeric", action="store_true")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-tokens", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_dateutil()
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
            filters={
                "context_length": args.context_length,
                "exclude_numeric": args.exclude_numeric,
            },
        ),
    )
    print(f"plain: {summary['scores']['plain']:.4f}")
    print(f"rlm: {summary['scores']['rlm']:.4f}")
    print(f"CSV: {summary['csv_path']}")
    print(f"JSON: {Path(summary['csv_path']).with_suffix('.json')}")


if __name__ == "__main__":
    main()
