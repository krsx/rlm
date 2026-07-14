from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.benchmark_common import validate_snapshot
from scripts.fetch_benchmarks import (
    CODEQA_REVISION,
    OOLONG_PAIRS_REVISION,
    OOLONG_REVISION,
    PAIR_CONTEXT_LENGTHS,
    fetch_oolong_pairs,
    normalize_codeqa,
    normalize_oolong,
    normalize_oolong_pairs,
    publish_snapshot,
    stream_pair_examples,
    write_pairs_snapshot,
    write_snapshot,
)


def oolong_row(
    *,
    row_id: int = 1,
    dataset: str = "trec_coarse",
    context_len: int = 131072,
    context: str = "User 7: question",
    labelled: str = "User 7: question [entity]",
    context_window_id: int = 7,
) -> dict[str, object]:
    return {
        "id": row_id,
        "context_len": context_len,
        "dataset": dataset,
        "context_window_text": context,
        "context_window_text_with_labels": labelled,
        "question": f"question {row_id}",
        "task_group": "group",
        "task": "task",
        "answer": "['entity']",
        "answer_type": "ANSWER_TYPE.ENTITY",
        "input_subset": "subset",
        "num_labels": 6,
        "context_window_id": context_window_id,
    }


def pair_tasks() -> list[dict[str, object]]:
    return [
        {
            "id": str(task_id),
            "question": f"pair question {task_id}",
            "answer": ["(9, 2)", "(2, 9)"] if task_id == 1 else [],
            "type": "list_of_answers",
        }
        for task_id in range(1, 21)
    ]


def codeqa_row(
    *,
    row_id: str = "code-1",
    domain: str = "Code Repository Understanding",
    sub_domain: str = "Code repo QA",
    answer: str = "A",
) -> dict[str, str]:
    return {
        "_id": row_id,
        "domain": domain,
        "sub_domain": sub_domain,
        "difficulty": "hard",
        "length": "long",
        "question": "What does this code do?",
        "choice_A": "A thing",
        "choice_B": "B thing",
        "choice_C": "C thing",
        "choice_D": "D thing",
        "answer": answer,
        "context": "def example(): pass",
    }


def test_source_revisions_are_full_commit_shas() -> None:
    assert OOLONG_REVISION == "f0d59eaf0febf130664cfceb710436c8e3216b2b"
    assert OOLONG_PAIRS_REVISION == "d1e1522b86ac0c169bbc890b0471408aaa29e8fa"
    assert CODEQA_REVISION == "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"


def test_normalize_oolong_filters_and_deduplicates_no_label_contexts() -> None:
    rows = [
        oolong_row(row_id=1),
        oolong_row(row_id=2),
        oolong_row(row_id=3, dataset="other"),
    ]

    contexts, examples = normalize_oolong(rows)

    assert contexts == [
        {
            "id": "trec_coarse-131072-7",
            "context_len": 131072,
            "context": "User 7: question",
        }
    ]
    assert [example["id"] for example in examples] == ["1", "2"]
    assert all(example["context_id"] == contexts[0]["id"] for example in examples)
    serialized = json.dumps({"contexts": contexts, "examples": examples})
    assert "context_window_text_with_labels" not in serialized
    assert "[entity]" not in serialized


def test_normalize_oolong_rejects_conflicting_context_text() -> None:
    rows = [oolong_row(row_id=1), oolong_row(row_id=2, context="different")]

    with pytest.raises(ValueError, match="Conflicting context"):
        normalize_oolong(rows)


def test_normalize_pairs_canonicalizes_and_deduplicates_gold_pairs() -> None:
    contexts = [
        {
            "id": "trec_coarse-32768-7",
            "context_len": 32768,
            "context": "context",
        }
    ]

    pair_contexts, examples = normalize_oolong_pairs({32768: pair_tasks()}, contexts)

    assert pair_contexts == contexts
    assert len(examples) == 20
    assert examples[0]["gold_pairs"] == [[2, 9]]
    assert examples[0]["id"] == "32768:1"


def test_normalize_pairs_rejects_missing_official_task() -> None:
    contexts = [{"id": "trec_coarse-32768-7", "context_len": 32768, "context": "context"}]

    with pytest.raises(ValueError, match="task IDs"):
        normalize_oolong_pairs({32768: pair_tasks()[:-1]}, contexts)


def test_normalize_pairs_rejects_self_pair() -> None:
    tasks = pair_tasks()
    tasks[0]["answer"] = ["(4, 4)"]
    contexts = [{"id": "trec_coarse-32768-7", "context_len": 32768, "context": "context"}]

    with pytest.raises(ValueError, match="different IDs"):
        normalize_oolong_pairs({32768: tasks}, contexts)


def test_stream_pair_examples_normalizes_without_materializing_file(
    tmp_path: Path,
) -> None:
    pytest.importorskip("ijson")
    raw_path = tmp_path / "pairs.json"
    raw_path.write_text(
        json.dumps(
            [
                {
                    "id": "1",
                    "question": "question one",
                    "answer": ["(9, 2)", "(2, 9)"],
                    "type": "list_of_answers",
                },
                {
                    "id": "2",
                    "question": "question two",
                    "answer": [],
                    "type": "list_of_answers",
                },
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "normalized.jsonl"

    with output_path.open("w+", encoding="utf-8") as output:
        task_ids = stream_pair_examples(
            raw_path,
            output,
            context_len=32768,
            context_id="context",
        )
        output.seek(0)
        rows = [json.loads(line) for line in output]

    assert task_ids == {"1", "2"}
    assert rows[0]["gold_pairs"] == [[2, 9]]
    assert rows[0]["id"] == "32768:1"
    assert rows[1]["gold_pairs"] == []


def test_fetch_oolong_pairs_picks_first_context_window_per_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("ijson")
    data_dir = tmp_path / "data"
    contexts: list[dict[str, object]] = []
    examples: list[dict[str, object]] = []
    for context_len in PAIR_CONTEXT_LENGTHS:
        # oolong-synth samples two independent trec_coarse context windows per
        # length (see fetch_oolong_pairs); only the first one in row order is
        # the one oolong-pairs' gold answers were computed against.
        contexts.append(
            {
                "id": f"trec_coarse-{context_len}-0",
                "context_len": context_len,
                "context": f"correct context {context_len}",
            }
        )
        contexts.append(
            {
                "id": f"trec_coarse-{context_len}-1",
                "context_len": context_len,
                "context": f"other context {context_len}",
            }
        )
        examples.append(
            {
                "id": f"ex-{context_len}",
                "context_id": f"trec_coarse-{context_len}-0",
                "context_len": context_len,
            }
        )
    write_snapshot(
        data_dir / "oolong",
        benchmark="oolong",
        sources=[
            {
                "repository": "oolongbench/oolong-synth",
                "revision": OOLONG_REVISION,
                "split": "validation",
            }
        ],
        transformations=["fixture"],
        canonical_filter={},
        rows_by_name={"contexts": contexts, "examples": examples},
        created_at="2026-07-14T00:00:00Z",
    )

    def fake_hf_hub_download(
        *, repo_id: str, filename: str, repo_type: str, revision: str, cache_dir: str
    ) -> str:
        context_len = filename.removeprefix("data/oolong-pairs-").removesuffix(".json")
        raw_path = tmp_path / f"raw-{context_len}.json"
        raw_path.write_text(json.dumps(pair_tasks()), encoding="utf-8")
        return str(raw_path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    assert fetch_oolong_pairs(data_dir, tmp_path / "cache", force=False)

    snapshot = validate_snapshot(data_dir, "oolong_pairs")
    assert snapshot.context_count == len(PAIR_CONTEXT_LENGTHS)
    assert {context["context"] for context in snapshot.contexts} == {
        f"correct context {context_len}" for context_len in PAIR_CONTEXT_LENGTHS
    }


def test_write_pairs_snapshot_streams_all_lengths_and_writes_valid_manifest(
    tmp_path: Path,
) -> None:
    pytest.importorskip("ijson")
    raw_paths: dict[int, Path] = {}
    contexts: list[dict[str, object]] = []
    for context_len in PAIR_CONTEXT_LENGTHS:
        raw_path = tmp_path / f"pairs-{context_len}.json"
        raw_path.write_text(json.dumps(pair_tasks()), encoding="utf-8")
        raw_paths[context_len] = raw_path
        contexts.append(
            {
                "id": f"context-{context_len}",
                "context_len": context_len,
                "context": f"context {context_len}",
            }
        )
    snapshot_dir = tmp_path / "snapshots" / "oolong_pairs"

    write_pairs_snapshot(
        snapshot_dir,
        raw_paths=raw_paths,
        contexts=contexts,
        created_at="2026-07-14T00:00:00Z",
    )

    validated = validate_snapshot(snapshot_dir.parent, "oolong_pairs")
    assert validated.example_count == 220
    assert validated.context_count == len(PAIR_CONTEXT_LENGTHS)
    assert len(list(validated.iter_examples())) == 220


def test_normalize_codeqa_keeps_only_code_repo_qa() -> None:
    examples = normalize_codeqa([codeqa_row(), codeqa_row(row_id="other", sub_domain="Other")])

    assert len(examples) == 1
    assert examples[0] == {
        "id": "code-1",
        "question": "What does this code do?",
        "choices": {"A": "A thing", "B": "B thing", "C": "C thing", "D": "D thing"},
        "gold_choice": "A",
        "context": "def example(): pass",
        "difficulty": "hard",
        "length": "long",
        "domain": "Code Repository Understanding",
        "sub_domain": "Code repo QA",
    }


def test_normalize_codeqa_rejects_invalid_gold_choice() -> None:
    with pytest.raises(ValueError, match="gold answer"):
        normalize_codeqa([codeqa_row(answer="E")])


def test_write_snapshot_records_files_counts_and_checksums(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "codeqa"
    write_snapshot(
        snapshot_dir,
        benchmark="codeqa",
        sources=[
            {
                "repository": "zai-org/LongBench-v2",
                "revision": CODEQA_REVISION,
                "split": "train",
            }
        ],
        transformations=["fixture"],
        canonical_filter={"domain": "Code Repository Understanding"},
        rows_by_name={"examples": [codeqa_row()]},
        created_at="2026-07-14T00:00:00Z",
    )

    snapshot = validate_snapshot(tmp_path, "codeqa")

    assert snapshot.example_count == 1
    assert snapshot.manifest["files"]["examples"]["byte_size"] > 0
    assert len(snapshot.manifest["files"]["examples"]["sha256"]) == 64


def test_publish_failure_preserves_existing_valid_snapshot(tmp_path: Path) -> None:
    target = tmp_path / "codeqa"
    write_snapshot(
        target,
        benchmark="codeqa",
        sources=[
            {
                "repository": "zai-org/LongBench-v2",
                "revision": CODEQA_REVISION,
                "split": "train",
            }
        ],
        transformations=["old"],
        canonical_filter={},
        rows_by_name={"examples": [{"id": "old"}]},
        created_at="2026-07-14T00:00:00Z",
    )

    def invalid_builder(staging: Path) -> None:
        staging.mkdir()
        (staging / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="missing fields"):
        publish_snapshot(tmp_path, "codeqa", invalid_builder, force=True)

    assert list(validate_snapshot(tmp_path, "codeqa").iter_examples()) == [{"id": "old"}]


def test_publish_skips_valid_snapshot_without_force(tmp_path: Path) -> None:
    target = tmp_path / "codeqa"
    write_snapshot(
        target,
        benchmark="codeqa",
        sources=[
            {
                "repository": "zai-org/LongBench-v2",
                "revision": CODEQA_REVISION,
                "split": "train",
            }
        ],
        transformations=["old"],
        canonical_filter={},
        rows_by_name={"examples": [{"id": "old"}]},
    )

    def unexpected_builder(staging: Path) -> None:
        raise AssertionError(f"builder unexpectedly called for {staging}")

    assert (
        publish_snapshot(
            tmp_path,
            "codeqa",
            unexpected_builder,
            force=False,
        )
        is False
    )


def test_publish_force_replaces_only_after_new_snapshot_validates(tmp_path: Path) -> None:
    target = tmp_path / "codeqa"
    source = {
        "repository": "zai-org/LongBench-v2",
        "revision": CODEQA_REVISION,
        "split": "train",
    }
    write_snapshot(
        target,
        benchmark="codeqa",
        sources=[source],
        transformations=["old"],
        canonical_filter={},
        rows_by_name={"examples": [{"id": "old"}]},
    )

    def replacement_builder(staging: Path) -> None:
        write_snapshot(
            staging,
            benchmark="codeqa",
            sources=[source],
            transformations=["new"],
            canonical_filter={},
            rows_by_name={"examples": [{"id": "new"}]},
        )

    assert publish_snapshot(tmp_path, "codeqa", replacement_builder, force=True)
    assert list(validate_snapshot(tmp_path, "codeqa").iter_examples()) == [{"id": "new"}]
