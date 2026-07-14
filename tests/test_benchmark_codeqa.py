from __future__ import annotations

from pathlib import Path

import pytest

from examples.benchmark_codeqa import (
    build_parser,
    build_plain_prompt,
    build_rlm_inputs,
    load_examples,
    parse_choice,
    score_choice,
)
from examples.benchmark_common import ValidatedSnapshot


@pytest.mark.parametrize(
    ("prediction", "choice", "status"),
    [
        ("Final answer: C", "C", "explicit"),
        ("After reviewing it, the final answer is (B).", "B", "explicit"),
        ("Answer: D", "D", "explicit"),
        ("C", "C", "standalone"),
        ("**A**", "A", "standalone"),
        ("A or B", None, "ambiguous"),
        ("Final answer: A. Final answer: B.", None, "ambiguous"),
        ("I lean toward C", None, "unparseable"),
        ("unknown", None, "unparseable"),
    ],
)
def test_parse_choice_is_strict_and_deterministic(
    prediction: str,
    choice: str | None,
    status: str,
) -> None:
    assert parse_choice(prediction) == (choice, status)


def test_score_choice_uses_exact_choice_and_retains_parse_status() -> None:
    correct = score_choice("B", "Final answer: B")
    wrong = score_choice("A", "B")
    unparseable = score_choice("A", "I do not know")

    assert (correct.score, correct.parse_status) == (1.0, "explicit")
    assert (wrong.score, wrong.parse_status) == (0.0, "standalone")
    assert (unparseable.score, unparseable.parse_status) == (0.0, "unparseable")


def example() -> dict[str, object]:
    return {
        "id": "code-1",
        "question": "What does the function return?",
        "choices": {
            "A": "one",
            "B": "two",
            "C": "three",
            "D": "four",
        },
        "gold_choice": "B",
        "context": "def f(): return 2",
        "difficulty": "hard",
        "length": "long",
        "domain": "Code Repository Understanding",
        "sub_domain": "Code repo QA",
    }


def snapshot(rows: list[dict[str, object]] | None = None) -> ValidatedSnapshot:
    return ValidatedSnapshot(
        benchmark="codeqa",
        directory=Path("unused"),
        manifest={
            "sources": [],
            "created_at": "2026-07-14T00:00:00Z",
            "canonical_filter": {
                "domain": "Code Repository Understanding",
                "sub_domain": "Code repo QA",
            },
        },
        examples=rows or [example()],
        contexts=[],
        manifest_sha256="a" * 64,
    )


def test_local_loader_preserves_all_stored_rows_in_source_order() -> None:
    first = example()
    second = {**example(), "id": "code-2"}

    loaded = load_examples(snapshot([first, second]), {})

    assert [row["id"] for row in loaded] == ["code-1", "code-2"]


def test_local_loader_rejects_invalid_choice_schema() -> None:
    invalid = example()
    invalid["choices"] = {"A": "one"}

    with pytest.raises(ValueError, match="choices A-D"):
        load_examples(snapshot([invalid]), {})


def test_prompts_include_choices_and_rlm_keeps_context_separate() -> None:
    item = example()

    plain = build_plain_prompt(item)
    prompt, root_prompt = build_rlm_inputs(item)

    assert "A. one" in plain
    assert "D. four" in plain
    assert "def f(): return 2" in plain
    assert prompt == "def f(): return 2"
    assert "What does the function return?" in root_prompt
    assert "def f(): return 2" not in root_prompt


def test_cli_defaults_to_full_stored_codeqa_slice() -> None:
    args = build_parser().parse_args(["--model", "model"])

    assert args.num_examples is None
    assert args.seed == 42
    assert args.data_dir == Path("./data/benchmarks")
