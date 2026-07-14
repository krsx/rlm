from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from examples.benchmark_common import ValidatedSnapshot
from examples.benchmark_oolong import (
    build_parser,
    build_plain_prompt,
    build_rlm_inputs,
    is_canonical,
    load_examples,
    synth_score,
)


def snapshot(
    *,
    contexts: list[dict[str, object]] | None = None,
    examples: list[dict[str, object]] | None = None,
) -> ValidatedSnapshot:
    return ValidatedSnapshot(
        benchmark="oolong",
        directory=Path("unused"),
        manifest={
            "sources": [],
            "created_at": "2026-07-14T00:00:00Z",
            "canonical_filter": {"dataset": "trec_coarse", "context_len": 131072},
        },
        contexts=contexts or [{"id": "ctx", "context_len": 131072, "context": "User 1: sample"}],
        examples=examples
        or [
            {
                "id": "1",
                "dataset": "trec_coarse",
                "context_len": 131072,
                "context_id": "ctx",
                "question": "Which category wins?",
                "answer": "['entity']",
                "answer_type": "ANSWER_TYPE.ENTITY",
            }
        ],
        manifest_sha256="a" * 64,
    )


def test_plain_prompt_contains_instruction_question_and_context() -> None:
    prompt = build_plain_prompt({"question": "Which category wins?", "context": "User 1: sample"})

    assert "Answer the following aggregate question." in prompt
    assert "Question: Which category wins?" in prompt
    assert "Context:\nUser 1: sample" in prompt


def test_rlm_inputs_keep_context_separate_from_root_prompt() -> None:
    prompt, root_prompt = build_rlm_inputs(
        {"question": "Which category wins?", "context": "long context"}
    )

    assert prompt == "long context"
    assert "Question: Which category wins?" in root_prompt
    assert "long context" not in root_prompt


def test_synth_score_matches_exact_literal_answer() -> None:
    datapoint = {"answer": "['Paris']", "answer_type": "ANSWER_TYPE.ENTITY"}

    assert synth_score(datapoint, "Answer: Paris") == 1.0


def test_synth_score_uses_last_comparison_phrase() -> None:
    datapoint = {
        "answer": "['less common than']",
        "answer_type": "ANSWER_TYPE.COMPARISON",
    }

    assert synth_score(datapoint, "Maybe more common than. Final: less common than") == 1.0


def test_synth_score_decays_numeric_distance() -> None:
    datapoint = {"answer": "[10]", "answer_type": "ANSWER_TYPE.NUMERIC"}

    assert synth_score(datapoint, "Answer: 12") == 0.75**2


def test_synth_score_parses_date_answer() -> None:
    datapoint = {
        "answer": "[datetime.date(2026, 7, 14)]",
        "answer_type": "ANSWER_TYPE.DATE",
    }

    assert synth_score(datapoint, "Answer: July 14, 2026") == 1.0


def test_synth_score_returns_zero_for_unparseable_numeric_answer() -> None:
    datapoint = {"answer": "[10]", "answer_type": "ANSWER_TYPE.NUMERIC"}

    assert synth_score(datapoint, "I cannot tell") == 0.0


def test_local_loader_joins_context_and_preserves_source_order() -> None:
    loaded = load_examples(
        snapshot(
            examples=[
                {
                    "id": "b",
                    "dataset": "trec_coarse",
                    "context_len": 131072,
                    "context_id": "ctx",
                    "question": "b",
                    "answer": "[1]",
                    "answer_type": "ANSWER_TYPE.NUMERIC",
                },
                {
                    "id": "a",
                    "dataset": "trec_coarse",
                    "context_len": 131072,
                    "context_id": "ctx",
                    "question": "a",
                    "answer": "['A']",
                    "answer_type": "ANSWER_TYPE.ENTITY",
                },
            ]
        ),
        {"context_length": 131072, "exclude_numeric": False},
    )

    assert [example["id"] for example in loaded] == ["b", "a"]
    assert all(example["context"] == "User 1: sample" for example in loaded)


def test_local_loader_can_exclude_numeric_for_selective_check() -> None:
    loaded = load_examples(
        snapshot(
            examples=[
                {
                    "id": "numeric",
                    "dataset": "trec_coarse",
                    "context_len": 131072,
                    "context_id": "ctx",
                    "question": "n",
                    "answer": "[1]",
                    "answer_type": "ANSWER_TYPE.NUMERIC",
                },
                {
                    "id": "entity",
                    "dataset": "trec_coarse",
                    "context_len": 131072,
                    "context_id": "ctx",
                    "question": "e",
                    "answer": "['A']",
                    "answer_type": "ANSWER_TYPE.ENTITY",
                },
            ]
        ),
        {"context_length": 131072, "exclude_numeric": True},
    )

    assert [example["id"] for example in loaded] == ["entity"]


def test_local_loader_rejects_missing_context_reference() -> None:
    with pytest.raises(ValueError, match="missing context"):
        load_examples(
            snapshot(
                examples=[
                    {
                        "id": "1",
                        "dataset": "trec_coarse",
                        "context_len": 131072,
                        "context_id": "absent",
                        "question": "q",
                        "answer": "['A']",
                        "answer_type": "ANSWER_TYPE.ENTITY",
                    }
                ]
            ),
            {"context_length": 131072, "exclude_numeric": False},
        )


def test_cli_defaults_to_full_canonical_slice() -> None:
    args = build_parser().parse_args(["--model", "model"])

    assert args.context_length == 131072
    assert args.num_examples is None
    assert args.seed == 42
    assert args.exclude_numeric is False
    assert not hasattr(args, "output")


def test_only_canonical_context_without_exclusion_is_full() -> None:
    assert is_canonical({"context_length": 131072, "exclude_numeric": False})
    assert not is_canonical({"context_length": 32768, "exclude_numeric": False})
    assert not is_canonical({"context_length": 131072, "exclude_numeric": True})


def test_runner_source_has_no_runtime_dataset_download() -> None:
    import examples.benchmark_oolong as benchmark_oolong

    source = inspect.getsource(benchmark_oolong)
    assert "load_dataset" not in source
    assert "from datasets" not in source


def test_makefile_exposes_independent_local_benchmark_targets() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    for target in (
        "fetch-benchmarks",
        "benchmark-oolong",
        "benchmark-oolong-pairs",
        "benchmark-codeqa",
    ):
        assert re.search(rf"^{target}:", makefile, re.MULTILINE)
    assert "BENCHMARK_DATA_DIR ?= ./data/benchmarks" in makefile
    assert "LOG_DIR ?= ./logs" in makefile
    assert "FETCH_ARGS ?=" in makefile
    assert "BENCHMARK_ARGS ?=" in makefile
    assert "--with datasets" in makefile
    assert "--with huggingface_hub" in makefile
    assert "--with ijson" in makefile
    assert "benchmark-all:" not in makefile


def test_benchmark_make_targets_do_not_depend_on_fetch() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    for target in ("benchmark-oolong", "benchmark-oolong-pairs", "benchmark-codeqa"):
        declaration = re.search(rf"^{target}:(.*)$", makefile, re.MULTILINE)
        assert declaration is not None
        assert "fetch-benchmarks" not in declaration.group(1)


def test_benchmark_data_directory_is_gitignored() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert "data/benchmarks/" in gitignore.splitlines()
