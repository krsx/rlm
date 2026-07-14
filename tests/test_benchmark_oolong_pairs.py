from __future__ import annotations

from pathlib import Path

import pytest

from examples.benchmark_common import ValidatedSnapshot
from examples.benchmark_oolong_pairs import (
    build_parser,
    build_plain_prompt,
    build_rlm_inputs,
    is_canonical,
    load_examples,
    parse_pairs,
    score_pairs,
)


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ("(9, 2), (2, 9), junk", {(2, 9)}),
        ("pairs: (1,2) and (3, 4)", {(1, 2), (3, 4)}),
        ("(4, 4)", set()),
        ("none", set()),
        ("(1, x), [2, 3]", set()),
    ],
)
def test_parse_pairs_canonicalizes_duplicates_and_ignores_malformed_fragments(
    prediction: str,
    expected: set[tuple[int, int]],
) -> None:
    assert parse_pairs(prediction) == expected


def test_pair_f1_tracks_precision_and_recall() -> None:
    result = score_pairs({(1, 2), (3, 4)}, "(2,1), (8,9)")

    assert result.score == pytest.approx(0.5)
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("gold", "prediction", "expected"),
    [
        (set(), "none", 1.0),
        ({(1, 2)}, "none", 0.0),
        (set(), "(1,2)", 0.0),
    ],
)
def test_pair_f1_empty_set_rules(
    gold: set[tuple[int, int]],
    prediction: str,
    expected: float,
) -> None:
    assert score_pairs(gold, prediction).score == expected


def snapshot() -> ValidatedSnapshot:
    return ValidatedSnapshot(
        benchmark="oolong_pairs",
        directory=Path("unused"),
        manifest={
            "sources": [],
            "created_at": "2026-07-14T00:00:00Z",
            "canonical_filter": {"context_len": 32768},
        },
        contexts=[
            {"id": "small", "context_len": 1024, "context": "small context"},
            {"id": "canonical", "context_len": 32768, "context": "long context"},
        ],
        examples=[
            {
                "id": "1024:1",
                "task_id": "1",
                "dataset": "oolong_pairs",
                "context_len": 1024,
                "context_id": "small",
                "question": "small question",
                "gold_pairs": [],
                "answer_type": "list_of_answers",
            },
            {
                "id": "32768:1",
                "task_id": "1",
                "dataset": "oolong_pairs",
                "context_len": 32768,
                "context_id": "canonical",
                "question": "canonical question",
                "gold_pairs": [[2, 9]],
                "answer_type": "list_of_answers",
            },
        ],
        manifest_sha256="a" * 64,
    )


def test_local_loader_selects_context_length_and_joins_context() -> None:
    loaded = load_examples(snapshot(), {"context_length": 32768})

    assert [example["id"] for example in loaded] == ["32768:1"]
    assert loaded[0]["context"] == "long context"


def test_local_loader_rejects_missing_context_reference() -> None:
    invalid = snapshot()
    invalid.examples[1]["context_id"] = "absent"

    with pytest.raises(ValueError, match="missing context"):
        load_examples(invalid, {"context_length": 32768})


def test_local_loader_stops_after_ordered_target_slice() -> None:
    class OrderedSnapshot:
        contexts = [{"id": "canonical", "context_len": 32768, "context": "long context"}]

        @staticmethod
        def iter_examples():
            yield {
                "id": "32768:1",
                "task_id": "1",
                "dataset": "oolong_pairs",
                "context_len": 32768,
                "context_id": "canonical",
                "question": "canonical question",
                "gold_pairs": [],
                "answer_type": "list_of_answers",
            }
            yield {"context_len": 65536}
            raise AssertionError("loader scanned beyond the selected ordered slice")

    loaded = load_examples(OrderedSnapshot(), {"context_length": 32768})

    assert [example["id"] for example in loaded] == ["32768:1"]


def test_prompts_request_pairs_and_rlm_keeps_context_separate() -> None:
    example = {
        "question": "List matching users.",
        "context": "very long context",
    }

    plain = build_plain_prompt(example)
    prompt, root_prompt = build_rlm_inputs(example)

    assert "List matching users." in plain
    assert "(id1, id2)" in plain
    assert "very long context" in plain
    assert prompt == "very long context"
    assert "List matching users." in root_prompt
    assert "very long context" not in root_prompt


def test_cli_defaults_to_all_twenty_canonical_tasks() -> None:
    args = build_parser().parse_args(["--model", "model"])

    assert args.context_length == 32768
    assert args.num_examples is None
    assert args.seed == 42


def test_only_default_context_is_canonical() -> None:
    assert is_canonical({"context_length": 32768})
    assert not is_canonical({"context_length": 131072})
