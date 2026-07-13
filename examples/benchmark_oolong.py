# Run an OOLONG benchmark against local vLLM:
#   uv run --with datasets --with python-dateutil python -m examples.benchmark_oolong --model Qwen/Qwen3-0.6B

from __future__ import annotations

import argparse
import ast
import csv
import random
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from rlm import RLM
from rlm.clients import get_client
from rlm.core.types import UsageSummary
from rlm.logger import RLMLogger

LOCAL_VLLM_BASE_URL = "http://localhost:8000/v1"
COMPARISON_PHRASES = ("more common than", "less common than", "same frequency as")
QUESTION_INSTRUCTION = (
    "The context contains thousands of general-knowledge questions, one per "
    "line. Each line has a User ID and a question, and each question's answer "
    "falls into one of 6 categories: 'numeric value', 'entity', 'location', "
    "'description and abstract concept', 'abbreviation', 'human being'. "
    "Answer the following aggregate question."
)
CSV_COLUMNS = [
    "example_id",
    "dataset_name",
    "mode",
    "model",
    "question",
    "gold_answer",
    "prediction",
    "score",
    "latency_sec",
    "prompt_tokens",
    "completion_tokens",
    "total_calls",
    "log_file",
    "error",
]


def find_comparison_phrase(output: str) -> str | None:
    out_low = output.lower()
    hits = [
        (out_low.rfind(phrase), phrase)
        for phrase in COMPARISON_PHRASES
        if phrase in out_low
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
    candidate = (
        answer.split(":")[-1].strip().replace("*", "").replace("[", "").replace("]", "")
    )
    if len(candidate) < 20:
        return candidate, "vhigh"
    return candidate, "med"


def synth_score(datapoint: dict[str, Any], output: str) -> float:
    answer = str(datapoint.get("answer", ""))
    try:
        if "datetime" in answer:
            gold: Any = datetime.strptime(answer, "[datetime.date(%Y, %m, %d)]")
        else:
            gold = ast.literal_eval(answer)[0]
    except Exception:
        gold = answer

    trimmed, _confidence = attempt_answer_parse(output)
    gold_s = str(gold)

    if trimmed == gold_s or trimmed.lower() == gold_s.lower():
        return 1.0

    answer_type = datapoint.get("answer_type", "")
    if answer_type == "ANSWER_TYPE.NUMERIC":
        try:
            return 0.75 ** abs(int(gold) - int(trimmed))
        except Exception:
            return 0.0
    if answer_type == "ANSWER_TYPE.DATE":
        try:
            import dateutil.parser

            return 1.0 if dateutil.parser.parse(trimmed) == gold else 0.0
        except Exception:
            return 0.0

    if gold_s and gold_s.lower() not in [
        phrase.lower() for phrase in COMPARISON_PHRASES
    ]:
        if gold_s.lower() in output.lower():
            return 1.0

    return 0.0


def build_root_prompt(question: str) -> str:
    return f"{QUESTION_INSTRUCTION}\n\nQuestion: {question}"


def build_plain_prompt(*, question: str, context: str) -> str:
    return f"{build_root_prompt(question)}\n\nContext:\n{context}"


def build_rlm_inputs(*, question: str, context: str) -> tuple[str, str]:
    return context, build_root_prompt(question)


def extract_context(example: dict[str, Any]) -> str:
    return str(example.get("context_window_text", example.get("context", "")))


def filter_examples(
    examples: Iterable[dict[str, Any]],
    *,
    dataset_name: str,
    min_ctx: int,
    max_ctx: int,
    num_examples: int,
    seed: int,
    exclude_numeric: bool,
) -> list[dict[str, Any]]:
    kept = [
        example
        for example in examples
        if example.get("dataset") == dataset_name
        and min_ctx <= example.get("context_len", 0) <= max_ctx
        and not (
            exclude_numeric and example.get("answer_type") == "ANSWER_TYPE.NUMERIC"
        )
    ]
    random.Random(seed).shuffle(kept)
    return kept[:num_examples] if num_examples > 0 else kept


def load_oolong_examples(
    *,
    dataset_name: str,
    min_ctx: int,
    max_ctx: int,
    num_examples: int,
    seed: int,
    exclude_numeric: bool,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing benchmark dependency: run with `uv run --with datasets --with python-dateutil ...`"
        ) from exc

    if num_examples > 0:
        stream = load_dataset(
            "oolongbench/oolong-synth", split="validation", streaming=True
        )
        stream = stream.shuffle(seed=seed, buffer_size=10_000)
        samples: list[dict[str, Any]] = []
        for example in stream:
            if (
                example.get("dataset") == dataset_name
                and min_ctx <= example.get("context_len", 0) <= max_ctx
                and not (
                    exclude_numeric
                    and example.get("answer_type") == "ANSWER_TYPE.NUMERIC"
                )
            ):
                samples.append(example)
                if len(samples) >= num_examples:
                    break
        return samples

    dataset = load_dataset("oolongbench/oolong-synth", split="validation")
    return filter_examples(
        dataset,
        dataset_name=dataset_name,
        min_ctx=min_ctx,
        max_ctx=max_ctx,
        num_examples=num_examples,
        seed=seed,
        exclude_numeric=exclude_numeric,
    )


def make_csv_row(
    *,
    example_id: int | str,
    dataset_name: str,
    mode: str,
    model: str,
    question: str,
    gold_answer: str,
    prediction: str,
    score: float | str,
    latency_sec: float | str,
    usage_summary: UsageSummary | None,
    log_file: str | None,
    error: str,
) -> dict[str, Any]:
    usage = list(usage_summary.model_usage_summaries.values()) if usage_summary else []
    return {
        "example_id": example_id,
        "dataset_name": dataset_name,
        "mode": mode,
        "model": model,
        "question": question,
        "gold_answer": gold_answer,
        "prediction": prediction,
        "score": score,
        "latency_sec": latency_sec,
        "prompt_tokens": (
            sum(item.total_input_tokens for item in usage) if usage else ""
        ),
        "completion_tokens": (
            sum(item.total_output_tokens for item in usage) if usage else ""
        ),
        "total_calls": sum(item.total_calls for item in usage) if usage else "",
        "log_file": log_file or "",
        "error": error,
    }


def require_vllm_server() -> None:
    response = requests.get(f"{LOCAL_VLLM_BASE_URL}/models", timeout=5)
    response.raise_for_status()


def run_plain(
    example: dict[str, Any], *, model: str, max_tokens: int | None
) -> tuple[str, UsageSummary]:
    client = get_client(
        "vllm",
        {
            "model_name": model,
            "base_url": LOCAL_VLLM_BASE_URL,
            "api_key": "dummy",
            "sampling_args": (
                {"max_tokens": max_tokens} if max_tokens is not None else None
            ),
        },
    )
    prediction = client.completion(
        build_plain_prompt(
            question=str(example["question"]), context=extract_context(example)
        )
    )
    return prediction, client.get_usage_summary()


def run_rlm(
    example: dict[str, Any],
    *,
    model: str,
    max_depth: int,
    max_iterations: int,
    max_tokens: int | None,
    log_dir: str,
) -> tuple[str, UsageSummary, str | None]:
    logger = RLMLogger(log_dir=log_dir)
    sampling_args = {"max_tokens": max_tokens} if max_tokens is not None else None
    rlm = RLM(
        backend="vllm",
        backend_kwargs={
            "model_name": model,
            "base_url": LOCAL_VLLM_BASE_URL,
            "api_key": "dummy",
        },
        max_depth=max_depth,
        max_iterations=max_iterations,
        sampling_args=sampling_args,
        sub_sampling_args=sampling_args,
        verbose=True,
        logger=logger,
    )
    prompt, root_prompt = build_rlm_inputs(
        question=str(example["question"]), context=extract_context(example)
    )
    result = rlm.completion(prompt, root_prompt=root_prompt)
    return result.response, result.usage_summary, logger.log_file_path


def append_row(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal OOLONG benchmark against local vLLM."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-name", default="trec_coarse")
    parser.add_argument("--min-ctx", type=int, default=1024)
    parser.add_argument("--max-ctx", type=int, default=4096)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-numeric", action="store_true")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--output", default="oolong_benchmark.csv")
    parser.add_argument("--log-dir", default="./logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_vllm_server()
    examples = load_oolong_examples(
        dataset_name=args.dataset_name,
        min_ctx=args.min_ctx,
        max_ctx=args.max_ctx,
        num_examples=args.num_examples,
        seed=args.seed,
        exclude_numeric=args.exclude_numeric,
    )

    output_path = Path(args.output)
    scores = {"plain": [], "rlm": []}
    for index, example in enumerate(examples):
        for mode in ("plain", "rlm"):
            start = time.perf_counter()
            prediction = ""
            score: float | str = ""
            usage_summary = None
            log_file = None
            error = ""
            try:
                if mode == "plain":
                    prediction, usage_summary = run_plain(
                        example, model=args.model, max_tokens=args.max_tokens
                    )
                else:
                    prediction, usage_summary, log_file = run_rlm(
                        example,
                        model=args.model,
                        max_depth=args.max_depth,
                        max_iterations=args.max_iterations,
                        max_tokens=args.max_tokens,
                        log_dir=args.log_dir,
                    )
                score = synth_score(example, prediction)
                scores[mode].append(score)
            except Exception as exc:
                error = str(exc)

            row = make_csv_row(
                example_id=example.get("id", index),
                dataset_name=args.dataset_name,
                mode=mode,
                model=args.model,
                question=str(example.get("question", "")),
                gold_answer=str(example.get("answer", "")),
                prediction=prediction,
                score=score,
                latency_sec=round(time.perf_counter() - start, 6),
                usage_summary=usage_summary,
                log_file=log_file,
                error=error,
            )
            append_row(output_path, row)

    for mode, mode_scores in scores.items():
        average = sum(mode_scores) / len(mode_scores) if mode_scores else 0.0
        print(f"{mode}: n={len(mode_scores)} avg_score={average:.4f}")
    print(f"CSV: {output_path}")


if __name__ == "__main__":
    main()
