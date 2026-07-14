from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import random
import re
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from rlm import RLM
from rlm.clients import get_client
from rlm.core.types import UsageSummary
from rlm.logger import RLMLogger

MANIFEST_SCHEMA_VERSION = 1
FULL_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
CSV_COLUMNS = [
    "run_id",
    "benchmark",
    "example_id",
    "dataset_name",
    "mode",
    "model",
    "question",
    "gold_answer",
    "prediction",
    "score",
    "score_precision",
    "score_recall",
    "parse_status",
    "latency_sec",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "total_calls",
    "trajectory_log_path",
    "error",
]


@dataclass(frozen=True)
class ValidatedSnapshot:
    benchmark: str
    directory: Path
    manifest: dict[str, Any]
    examples: list[dict[str, Any]]
    contexts: list[dict[str, Any]]
    manifest_sha256: str
    examples_path: Path | None = None

    @property
    def example_count(self) -> int:
        if self.examples_path is not None:
            return int(self.manifest["example_count"])
        return len(self.examples)

    @property
    def context_count(self) -> int:
        return len(self.contexts)

    @property
    def source_revisions(self) -> dict[str, str]:
        return {
            str(source["repository"]): str(source["revision"])
            for source in self.manifest["sources"]
        }

    def iter_examples(self) -> Iterator[dict[str, Any]]:
        if self.examples_path is None:
            yield from self.examples
            return
        yield from iter_jsonl_gz(self.examples_path)


@dataclass(frozen=True)
class ArtifactPaths:
    run_id: str
    csv: Path
    summary: Path


@dataclass(frozen=True)
class ScoreResult:
    score: float
    precision: float | None = None
    recall: float | None = None
    parse_status: str = ""


@dataclass(frozen=True)
class CallResult:
    prediction: str
    usage_summary: UsageSummary
    trajectory_log_path: str | None = None


@dataclass(frozen=True)
class RunnerConfig:
    model: str
    data_dir: Path
    log_dir: Path
    num_examples: int | None
    seed: int
    max_depth: int
    max_iterations: int
    max_tokens: int | None
    filters: Mapping[str, Any]
    base_url: str = DEFAULT_VLLM_BASE_URL

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "data_dir": str(self.data_dir),
            "log_dir": str(self.log_dir),
            "num_examples": self.num_examples,
            "seed": self.seed,
            "max_depth": self.max_depth,
            "max_iterations": self.max_iterations,
            "max_tokens": self.max_tokens,
            "filters": dict(self.filters),
        }


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark: str
    dataset_name: str
    scoring_method: str
    load_examples: Callable[[ValidatedSnapshot, Mapping[str, Any]], list[dict[str, Any]]]
    build_plain_prompt: Callable[[dict[str, Any]], str]
    build_rlm_inputs: Callable[[dict[str, Any]], tuple[str, str]]
    score: Callable[[dict[str, Any], str], ScoreResult]
    gold_answer: Callable[[dict[str, Any]], str]
    is_canonical: Callable[[Mapping[str, Any]], bool]
    canonical_example_count: int | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl_gz(path))


def iter_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record in {path} at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record in {path} at line {line_number} is not an object")
            yield value


def count_jsonl_records(path: Path) -> int:
    record_count = 0
    last_byte = b""
    with gzip.open(path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            record_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if last_byte and last_byte != b"\n":
        raise ValueError(f"JSONL file does not end with a newline: {path}")
    return record_count


def require_manifest_fields(manifest: dict[str, Any], benchmark: str) -> None:
    required = {
        "schema_version",
        "benchmark",
        "created_at",
        "sources",
        "transformations",
        "canonical_filter",
        "files",
        "example_count",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"Manifest for {benchmark} is missing fields: {', '.join(missing)}")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema version for {benchmark}: "
            f"{manifest['schema_version']} (expected {MANIFEST_SCHEMA_VERSION})"
        )
    if manifest["benchmark"] != benchmark:
        raise ValueError(
            f"Manifest benchmark mismatch: expected {benchmark}, got {manifest['benchmark']}"
        )
    if not isinstance(manifest["sources"], list) or not manifest["sources"]:
        raise ValueError(f"Manifest for {benchmark} must contain at least one source")
    for source in manifest["sources"]:
        if not isinstance(source, dict):
            raise ValueError(f"Manifest source for {benchmark} must be an object")
        for field in ("repository", "revision", "split"):
            if field not in source:
                raise ValueError(f"Manifest source for {benchmark} is missing {field}")
        if FULL_REVISION_PATTERN.fullmatch(str(source["revision"])) is None:
            raise ValueError(
                f"Manifest source revision for {source['repository']} is not a full commit SHA"
            )


def validate_declared_file(
    snapshot_dir: Path,
    logical_name: str,
    metadata: dict[str, Any],
) -> Path:
    required = {"name", "byte_size", "record_count", "sha256"}
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"Manifest file {logical_name} is missing fields: {', '.join(missing)}")
    filename = str(metadata["name"])
    if Path(filename).name != filename:
        raise ValueError(f"Manifest file name must not contain a path: {filename}")
    path = snapshot_dir / filename
    if not path.is_file():
        raise ValueError(f"Snapshot file is missing: {path}")
    actual_checksum = sha256_file(path)
    if actual_checksum != metadata["sha256"]:
        raise ValueError(f"Snapshot checksum mismatch for {path}")
    actual_size = path.stat().st_size
    if actual_size != metadata["byte_size"]:
        raise ValueError(
            f"Snapshot byte size mismatch for {path}: "
            f"expected {metadata['byte_size']}, got {actual_size}"
        )
    record_count = count_jsonl_records(path)
    if record_count != metadata["record_count"]:
        raise ValueError(
            f"Snapshot record count mismatch for {path}: "
            f"expected {metadata['record_count']}, got {record_count}"
        )
    return path


def validate_snapshot(data_dir: str | Path, benchmark: str) -> ValidatedSnapshot:
    snapshot_dir = Path(data_dir) / benchmark
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(
            f"Missing local {benchmark} benchmark snapshot at {snapshot_dir}. "
            "Run `make fetch-benchmarks` first."
        )
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Snapshot manifest is missing: {manifest_path}")
    manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_value, dict):
        raise ValueError(f"Snapshot manifest is not a JSON object: {manifest_path}")
    manifest: dict[str, Any] = manifest_value
    require_manifest_fields(manifest, benchmark)

    files = manifest["files"]
    if not isinstance(files, dict) or "examples" not in files:
        raise ValueError(f"Manifest for {benchmark} must declare an examples file")
    validated_paths = {
        logical_name: validate_declared_file(snapshot_dir, logical_name, metadata)
        for logical_name, metadata in files.items()
    }
    examples_path = validated_paths["examples"]
    contexts = read_jsonl_gz(validated_paths["contexts"]) if "contexts" in validated_paths else []
    if files["examples"]["record_count"] != manifest["example_count"]:
        raise ValueError(
            f"Snapshot example count mismatch for {benchmark}: "
            f"expected {manifest['example_count']}, "
            f"got {files['examples']['record_count']}"
        )
    if "context_count" in manifest and len(contexts) != manifest["context_count"]:
        raise ValueError(
            f"Snapshot context count mismatch for {benchmark}: "
            f"expected {manifest['context_count']}, got {len(contexts)}"
        )
    return ValidatedSnapshot(
        benchmark=benchmark,
        directory=snapshot_dir,
        manifest=manifest,
        examples=[],
        contexts=contexts,
        manifest_sha256=sha256_file(manifest_path),
        examples_path=examples_path,
    )


def select_examples(
    examples: Sequence[dict[str, Any]],
    *,
    num_examples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    selected = list(examples)
    if num_examples is None:
        return selected
    if (
        isinstance(num_examples, bool)
        or not isinstance(num_examples, int)
        or num_examples <= 0
        or num_examples > len(selected)
    ):
        raise ValueError(
            "--num-examples must be a positive integer no larger than the "
            f"eligible example count ({len(selected)}), got {num_examples!r}"
        )
    random.Random(seed).shuffle(selected)
    return selected[:num_examples]


def model_slug(model: str) -> str:
    slug = re.sub(r"[^a-z0-9.-]+", "-", model.lower()).strip("-.")
    if not slug:
        raise ValueError(f"Model name cannot produce an artifact slug: {model!r}")
    return slug


def create_artifact_paths(
    log_dir: str | Path,
    benchmark: str,
    model: str,
    started_at: datetime,
) -> ArtifactPaths:
    if started_at.tzinfo is None:
        raise ValueError("Artifact timestamp must be timezone-aware")
    timestamp = started_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{benchmark}_{model_slug(model)}_{timestamp}"
    directory = Path(log_dir)
    csv_path = directory / f"{run_id}.csv"
    summary_path = directory / f"{run_id}.json"
    collisions = [path for path in (csv_path, summary_path) if path.exists()]
    if collisions:
        raise FileExistsError(
            "Benchmark artifacts already exist: " + ", ".join(str(path) for path in collisions)
        )
    return ArtifactPaths(run_id=run_id, csv=csv_path, summary=summary_path)


def append_csv_row(path: Path, row: Mapping[str, Any]) -> None:
    unexpected = sorted(set(row) - set(CSV_COLUMNS))
    missing = sorted(set(CSV_COLUMNS) - set(row))
    if unexpected or missing:
        raise ValueError(f"CSV row schema mismatch; missing={missing}, unexpected={unexpected}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        file.flush()


def write_summary_atomic(path: Path, summary: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Benchmark summary already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(summary, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def usage_totals(usage_summary: UsageSummary | None) -> dict[str, int]:
    if usage_summary is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_calls": 0,
        }
    summaries = usage_summary.model_usage_summaries.values()
    prompt_tokens = sum(summary.total_input_tokens for summary in summaries)
    summaries = usage_summary.model_usage_summaries.values()
    completion_tokens = sum(summary.total_output_tokens for summary in summaries)
    summaries = usage_summary.model_usage_summaries.values()
    total_calls = sum(summary.total_calls for summary in summaries)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "total_calls": total_calls,
    }


def require_vllm_server(base_url: str) -> None:
    response = requests.get(f"{base_url}/models", timeout=5)
    response.raise_for_status()


def run_plain(prompt: str, config: RunnerConfig) -> CallResult:
    sampling_args = {"max_tokens": config.max_tokens} if config.max_tokens is not None else None
    client = get_client(
        "vllm",
        {
            "model_name": config.model,
            "base_url": config.base_url,
            "api_key": "dummy",
            "sampling_args": sampling_args,
        },
    )
    prediction = client.completion(prompt)
    return CallResult(prediction=prediction, usage_summary=client.get_usage_summary())


def run_rlm(prompt: str, root_prompt: str, config: RunnerConfig) -> CallResult:
    config.log_dir.mkdir(parents=True, exist_ok=True)
    logger = RLMLogger(log_dir=str(config.log_dir))
    sampling_args = {"max_tokens": config.max_tokens} if config.max_tokens is not None else None
    rlm = RLM(
        backend="vllm",
        backend_kwargs={
            "model_name": config.model,
            "base_url": config.base_url,
            "api_key": "dummy",
        },
        max_depth=config.max_depth,
        max_iterations=config.max_iterations,
        sampling_args=sampling_args,
        sub_sampling_args=sampling_args,
        logger=logger,
    )
    result = rlm.completion(prompt, root_prompt=root_prompt)
    return CallResult(
        prediction=result.response,
        usage_summary=result.usage_summary,
        trajectory_log_path=logger.log_file_path,
    )


def zero_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "total_calls": 0,
    }


def add_usage(total: dict[str, int], addition: Mapping[str, int]) -> None:
    for key in total:
        total[key] += addition[key]


def utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Run timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def run_benchmark(
    spec: BenchmarkSpec,
    config: RunnerConfig,
    *,
    health_check: Callable[[str], None] = require_vllm_server,
    plain_call: Callable[[str, RunnerConfig], CallResult] = run_plain,
    rlm_call: Callable[[str, str, RunnerConfig], CallResult] = run_rlm,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, Any]:
    snapshot = validate_snapshot(config.data_dir, spec.benchmark)
    eligible = spec.load_examples(snapshot, config.filters)
    if not eligible:
        raise ValueError(f"No {spec.benchmark} examples match filters {dict(config.filters)!r}")
    selected = select_examples(
        eligible,
        num_examples=config.num_examples,
        seed=config.seed,
    )
    health_check(config.base_url)
    started_at = now()
    artifacts = create_artifact_paths(
        config.log_dir,
        spec.benchmark,
        config.model,
        started_at,
    )

    scores: dict[str, list[float]] = {"plain": [], "rlm": []}
    latency: dict[str, float] = {"plain": 0.0, "rlm": 0.0}
    usage: dict[str, dict[str, int]] = {
        "plain": zero_usage(),
        "rlm": zero_usage(),
    }
    succeeded = 0
    failed = 0
    for example in selected:
        plain_prompt = spec.build_plain_prompt(example)
        rlm_prompt, root_prompt = spec.build_rlm_inputs(example)
        for mode in ("plain", "rlm"):
            call_started = time.perf_counter()
            error = ""
            call_result: CallResult | None = None
            try:
                if mode == "plain":
                    call_result = plain_call(plain_prompt, config)
                else:
                    call_result = rlm_call(rlm_prompt, root_prompt, config)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - call_started
            latency[mode] += elapsed

            if call_result is None:
                score_result = ScoreResult(score=0.0, parse_status="inference_error")
                prediction = ""
                row_usage = zero_usage()
                trajectory_log_path = ""
                failed += 1
            else:
                score_result = spec.score(example, call_result.prediction)
                if not 0.0 <= score_result.score <= 1.0:
                    raise ValueError(
                        f"{spec.benchmark} scorer returned an out-of-range score: "
                        f"{score_result.score}"
                    )
                prediction = call_result.prediction
                row_usage = usage_totals(call_result.usage_summary)
                trajectory_log_path = call_result.trajectory_log_path or ""
                succeeded += 1
            scores[mode].append(score_result.score)
            add_usage(usage[mode], row_usage)
            append_csv_row(
                artifacts.csv,
                {
                    "run_id": artifacts.run_id,
                    "benchmark": spec.benchmark,
                    "example_id": str(example["id"]),
                    "dataset_name": spec.dataset_name,
                    "mode": mode,
                    "model": config.model,
                    "question": str(example["question"]),
                    "gold_answer": spec.gold_answer(example),
                    "prediction": prediction,
                    "score": score_result.score,
                    "score_precision": (
                        score_result.precision if score_result.precision is not None else ""
                    ),
                    "score_recall": (
                        score_result.recall if score_result.recall is not None else ""
                    ),
                    "parse_status": score_result.parse_status,
                    "latency_sec": round(elapsed, 6),
                    "prompt_tokens": row_usage["prompt_tokens"],
                    "completion_tokens": row_usage["completion_tokens"],
                    "total_tokens": row_usage["total_tokens"],
                    "total_calls": row_usage["total_calls"],
                    "trajectory_log_path": trajectory_log_path,
                    "error": error,
                },
            )

    denominator = len(selected)
    mode_scores = {mode: sum(mode_values) / denominator for mode, mode_values in scores.items()}
    is_full_evaluation = (
        config.num_examples is None
        and spec.is_canonical(config.filters)
        and (spec.canonical_example_count is None or len(eligible) == spec.canonical_example_count)
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": artifacts.run_id,
        "benchmark": spec.benchmark,
        "model": config.model,
        "started_at": utc_iso(started_at),
        "finished_at": utc_iso(now()),
        "configuration": config.to_dict(),
        "snapshot": {
            "manifest_sha256": snapshot.manifest_sha256,
            "created_at": snapshot.manifest["created_at"],
            "sources": snapshot.manifest["sources"],
            "canonical_filter": snapshot.manifest["canonical_filter"],
        },
        "csv_path": str(artifacts.csv),
        "selection": {
            "is_full_evaluation": is_full_evaluation,
            "eligible_count": len(eligible),
            "selected_count": len(selected),
            "selected_example_ids": [str(example["id"]) for example in selected],
            "num_examples": config.num_examples,
            "seed": config.seed,
            "filters": dict(config.filters),
        },
        "counts": {
            "eligible": len(eligible),
            "selected": len(selected),
            "succeeded": succeeded,
            "failed": failed,
        },
        "scores": {
            "plain": mode_scores["plain"],
            "rlm": mode_scores["rlm"],
            "rlm_minus_plain": mode_scores["rlm"] - mode_scores["plain"],
        },
        "latency_sec": latency,
        "usage": usage,
        "scoring_method": spec.scoring_method,
    }
    write_summary_atomic(artifacts.summary, summary)
    return summary
