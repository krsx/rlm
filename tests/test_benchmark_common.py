from __future__ import annotations

import gzip
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from examples.benchmark_common import (
    ArtifactPaths,
    BenchmarkSpec,
    CallResult,
    RunnerConfig,
    ScoreResult,
    append_csv_row,
    create_artifact_paths,
    model_slug,
    read_jsonl_gz,
    run_benchmark,
    select_examples,
    validate_snapshot,
    write_summary_atomic,
)
from rlm.core.types import ModelUsageSummary, UsageSummary


def write_jsonl_gz(path: Path, rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True))
            file.write("\n")


def file_metadata(path: Path, record_count: int) -> dict[str, object]:
    return {
        "name": path.name,
        "byte_size": path.stat().st_size,
        "record_count": record_count,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def create_snapshot(
    root: Path,
    benchmark: str,
    examples: list[dict[str, object]],
    *,
    schema_version: int = 1,
) -> Path:
    snapshot = root / benchmark
    snapshot.mkdir(parents=True)
    examples_path = snapshot / "examples.jsonl.gz"
    write_jsonl_gz(examples_path, examples)
    manifest = {
        "schema_version": schema_version,
        "benchmark": benchmark,
        "created_at": "2026-07-14T00:00:00Z",
        "sources": [
            {
                "repository": "example/source",
                "revision": "a" * 40,
                "split": "train",
            }
        ],
        "transformations": ["fixture normalization"],
        "canonical_filter": {},
        "files": {"examples": file_metadata(examples_path, len(examples))},
        "example_count": len(examples),
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return snapshot


def test_validate_snapshot_returns_manifest_and_rows(tmp_path: Path) -> None:
    create_snapshot(tmp_path, "codeqa", [{"id": "1"}, {"id": "2"}])

    snapshot = validate_snapshot(tmp_path, "codeqa")

    assert snapshot.benchmark == "codeqa"
    assert snapshot.example_count == 2
    assert snapshot.examples == []
    assert [row["id"] for row in snapshot.iter_examples()] == ["1", "2"]
    assert snapshot.source_revisions == {
        "example/source": "a" * 40,
    }


def test_read_jsonl_gz_preserves_source_order(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl.gz"
    write_jsonl_gz(path, [{"id": "b"}, {"id": "a"}])

    assert read_jsonl_gz(path) == [{"id": "b"}, {"id": "a"}]


def test_validate_snapshot_missing_data_names_fetch_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make fetch-benchmarks"):
        validate_snapshot(tmp_path, "oolong")


def test_validate_snapshot_rejects_unsupported_schema(tmp_path: Path) -> None:
    create_snapshot(tmp_path, "codeqa", [{"id": "1"}], schema_version=2)

    with pytest.raises(ValueError, match="schema version"):
        validate_snapshot(tmp_path, "codeqa")


def test_validate_snapshot_rejects_checksum_mismatch(tmp_path: Path) -> None:
    snapshot = create_snapshot(tmp_path, "oolong", [{"id": "1"}])
    with gzip.open(snapshot / "examples.jsonl.gz", "at", encoding="utf-8") as file:
        file.write(json.dumps({"id": "2"}) + "\n")

    with pytest.raises(ValueError, match="checksum"):
        validate_snapshot(tmp_path, "oolong")


def test_validate_snapshot_rejects_record_count_mismatch(tmp_path: Path) -> None:
    snapshot = create_snapshot(tmp_path, "oolong", [{"id": "1"}])
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["examples"]["record_count"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="record count"):
        validate_snapshot(tmp_path, "oolong")


def test_select_examples_without_cap_preserves_complete_source_order() -> None:
    examples = [{"id": str(index)} for index in range(5)]

    assert select_examples(examples, num_examples=None, seed=999) == examples


def test_select_examples_with_cap_is_exact_and_deterministic() -> None:
    examples = [{"id": str(index)} for index in range(10)]

    first = select_examples(examples, num_examples=3, seed=42)
    second = select_examples(examples, num_examples=3, seed=42)

    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) == 3


@pytest.mark.parametrize("cap", [0, -1, 6, True])
def test_select_examples_rejects_invalid_caps(cap: object) -> None:
    with pytest.raises(ValueError, match="num-examples"):
        select_examples([{"id": str(index)} for index in range(5)], num_examples=cap, seed=42)


def test_model_slug_sanitizes_repository_style_model_name() -> None:
    assert model_slug("Qwen/Qwen3-1.7B") == "qwen-qwen3-1.7b"


def test_artifacts_share_stem_and_refuse_overwrite(tmp_path: Path) -> None:
    started_at = datetime(2026, 7, 14, 14, 30, 12, tzinfo=UTC)
    paths = create_artifact_paths(tmp_path, "codeqa", "Qwen/Qwen3-1.7B", started_at)

    assert paths.csv.stem == paths.summary.stem
    assert paths.csv.name == "codeqa_qwen-qwen3-1.7b_20260714T143012Z.csv"
    paths.csv.parent.mkdir(parents=True, exist_ok=True)
    paths.csv.touch()

    with pytest.raises(FileExistsError):
        create_artifact_paths(tmp_path, "codeqa", "Qwen/Qwen3-1.7B", started_at)


def test_csv_rows_append_incrementally_with_one_header(tmp_path: Path) -> None:
    path = tmp_path / "run.csv"
    row = {
        "run_id": "run",
        "benchmark": "codeqa",
        "example_id": "1",
        "dataset_name": "CodeQA",
        "mode": "plain",
        "model": "model",
        "question": "question",
        "gold_answer": "A",
        "prediction": "A",
        "score": 1.0,
        "score_precision": "",
        "score_recall": "",
        "parse_status": "explicit",
        "latency_sec": 0.1,
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
        "total_calls": 1,
        "trajectory_log_path": "",
        "error": "",
    }

    append_csv_row(path, row)
    append_csv_row(path, {**row, "mode": "rlm"})

    assert path.read_text(encoding="utf-8").count("run_id,benchmark") == 1
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_summary_is_created_atomically_without_temp_file(tmp_path: Path) -> None:
    paths = ArtifactPaths(
        run_id="run",
        csv=tmp_path / "run.csv",
        summary=tmp_path / "run.json",
    )

    write_summary_atomic(paths.summary, {"schema_version": 1, "run_id": "run"})

    assert json.loads(paths.summary.read_text(encoding="utf-8"))["run_id"] == "run"
    assert list(tmp_path.glob("*.tmp")) == []


def test_runner_pairs_same_examples_and_counts_inference_failures_as_zero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    create_snapshot(
        tmp_path / "data",
        "codeqa",
        [{"id": str(index), "question": f"q{index}", "gold": "1"} for index in range(4)],
    )
    calls: list[tuple[str, str]] = []
    usage = UsageSummary(
        {
            "model": ModelUsageSummary(
                total_calls=1,
                total_input_tokens=2,
                total_output_tokens=1,
            )
        }
    )

    def plain_call(prompt: str, config: RunnerConfig) -> CallResult:
        calls.append(("plain", prompt))
        return CallResult(prediction="1", usage_summary=usage)

    def rlm_call(prompt: str, root_prompt: str, config: RunnerConfig) -> CallResult:
        calls.append(("rlm", root_prompt))
        raise RuntimeError("inference\nfailed")

    spec = BenchmarkSpec(
        benchmark="codeqa",
        dataset_name="CodeQA",
        scoring_method="fixture accuracy",
        load_examples=lambda snapshot, filters: list(snapshot.iter_examples()),
        build_plain_prompt=lambda example: str(example["id"]),
        build_rlm_inputs=lambda example: ("context", str(example["id"])),
        score=lambda example, prediction: ScoreResult(float(prediction), parse_status="ok"),
        gold_answer=lambda example: str(example["gold"]),
        is_canonical=lambda filters: True,
    )
    config = RunnerConfig(
        model="Qwen/Qwen3-1.7B",
        data_dir=tmp_path / "data",
        log_dir=tmp_path / "logs",
        num_examples=2,
        seed=42,
        max_depth=1,
        max_iterations=2,
        max_tokens=16,
        filters={},
    )

    def fixed_now() -> datetime:
        return datetime(2026, 7, 14, 14, 30, 12, tzinfo=UTC)

    health_urls: list[str] = []
    summary = run_benchmark(
        spec,
        config,
        health_check=health_urls.append,
        plain_call=plain_call,
        rlm_call=rlm_call,
        now=fixed_now,
    )

    assert health_urls == ["http://localhost:8000/v1"]
    selected = summary["selection"]["selected_example_ids"]
    assert calls == [
        ("plain", selected[0]),
        ("rlm", selected[0]),
        ("plain", selected[1]),
        ("rlm", selected[1]),
    ]
    assert summary["selection"]["is_full_evaluation"] is False
    assert summary["counts"] == {"eligible": 4, "selected": 2, "succeeded": 2, "failed": 2}
    assert summary["scores"] == {"plain": 1.0, "rlm": 0.0, "rlm_minus_plain": -1.0}
    assert summary["usage"]["plain"]["total_tokens"] == 6
    assert summary["usage"]["rlm"]["total_tokens"] == 0
    assert Path(summary["csv_path"]).exists()
    assert Path(summary["csv_path"]).with_suffix(".json").exists()
    csv_text = Path(summary["csv_path"]).read_text(encoding="utf-8")
    assert csv_text.count("inference\nfailed") == 2

    progress_lines = capsys.readouterr().out.splitlines()
    assert len(progress_lines) == 8
    for example_index, example_id in enumerate(selected, start=1):
        offset = (example_index - 1) * 4
        common = (
            rf"\d{{2}}:\d{{2}}:\d{{2}} \| Qwen3-1\.7B \| codeqa \| "
            rf"example {example_index}/2 \| id={example_id} \|"
        )
        assert re.fullmatch(rf"{common} plain \| running", progress_lines[offset])
        assert re.fullmatch(
            rf"{common} plain \| done \| \d+\.\d+s \| score=1\.000 \| tokens=3",
            progress_lines[offset + 1],
        )
        assert re.fullmatch(rf"{common} rlm \| running", progress_lines[offset + 2])
        assert re.fullmatch(
            rf"{common} rlm \| error \| \d+\.\d+s \| "
            r"RuntimeError: inference failed",
            progress_lines[offset + 3],
        )


def test_startup_health_failure_creates_no_result_artifacts(tmp_path: Path) -> None:
    create_snapshot(
        tmp_path / "data",
        "codeqa",
        [{"id": "1", "question": "q", "gold": "A"}],
    )
    spec = BenchmarkSpec(
        benchmark="codeqa",
        dataset_name="CodeQA",
        scoring_method="accuracy",
        load_examples=lambda snapshot, filters: list(snapshot.iter_examples()),
        build_plain_prompt=lambda example: "plain",
        build_rlm_inputs=lambda example: ("context", "root"),
        score=lambda example, prediction: ScoreResult(1.0),
        gold_answer=lambda example: str(example["gold"]),
        is_canonical=lambda filters: True,
        canonical_example_count=1,
    )
    config = RunnerConfig(
        model="model",
        data_dir=tmp_path / "data",
        log_dir=tmp_path / "logs",
        num_examples=None,
        seed=42,
        max_depth=1,
        max_iterations=2,
        max_tokens=None,
        filters={},
    )

    with pytest.raises(RuntimeError, match="server unavailable"):
        run_benchmark(
            spec,
            config,
            health_check=lambda base_url: (_ for _ in ()).throw(RuntimeError("server unavailable")),
        )

    assert not config.log_dir.exists()
