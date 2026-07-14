from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from examples.benchmark_common import sha256_file, validate_snapshot

OOLONG_REPOSITORY = "oolongbench/oolong-synth"
OOLONG_REVISION = "f0d59eaf0febf130664cfceb710436c8e3216b2b"
OOLONG_PAIRS_REPOSITORY = "mit-oasys/oolong-pairs"
OOLONG_PAIRS_REVISION = "d1e1522b86ac0c169bbc890b0471408aaa29e8fa"
CODEQA_REPOSITORY = "zai-org/LongBench-v2"
CODEQA_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"

PAIR_CONTEXT_LENGTHS = (
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
)
PAIR_PATTERN = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
OFFICIAL_PAIR_TASK_IDS = {str(task_id) for task_id in range(1, 21)}


def require_fields(row: Mapping[str, Any], fields: set[str], source: str) -> None:
    missing = sorted(fields - row.keys())
    if missing:
        raise ValueError(f"{source} row is missing fields: {', '.join(missing)}")


def normalize_oolong(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contexts_by_id: dict[str, dict[str, Any]] = {}
    examples: list[dict[str, Any]] = []
    seen_example_ids: set[str] = set()
    required_fields = {
        "id",
        "context_len",
        "dataset",
        "context_window_text",
        "question",
        "task_group",
        "task",
        "answer",
        "answer_type",
        "input_subset",
        "num_labels",
        "context_window_id",
    }
    for row in rows:
        require_fields(row, {"dataset"}, OOLONG_REPOSITORY)
        if row["dataset"] != "trec_coarse":
            continue
        require_fields(row, required_fields, OOLONG_REPOSITORY)
        context_len = int(row["context_len"])
        context_window_id = str(row["context_window_id"])
        context_id = f"trec_coarse-{context_len}-{context_window_id}"
        context = {
            "id": context_id,
            "context_len": context_len,
            "context": str(row["context_window_text"]),
        }
        existing = contexts_by_id.get(context_id)
        if existing is not None and existing != context:
            raise ValueError(f"Conflicting context text for {context_id}")
        contexts_by_id.setdefault(context_id, context)

        example_id = str(row["id"])
        if example_id in seen_example_ids:
            raise ValueError(f"Duplicate OOLONG example ID: {example_id}")
        seen_example_ids.add(example_id)
        examples.append(
            {
                "id": example_id,
                "upstream_id": row["id"],
                "dataset": "trec_coarse",
                "context_len": context_len,
                "context_id": context_id,
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "answer_type": str(row["answer_type"]),
                "task_group": str(row["task_group"]),
                "task": str(row["task"]),
                "input_subset": str(row["input_subset"]),
                "num_labels": int(row["num_labels"]),
            }
        )
    if not examples:
        raise ValueError("OOLONG source contains no trec_coarse examples")
    return list(contexts_by_id.values()), examples


def parse_gold_pair(value: Any, *, context_len: int, task_id: str) -> tuple[int, int]:
    match = PAIR_PATTERN.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(
            f"Malformed gold pair for context {context_len}, task {task_id}: {value!r}"
        )
    first, second = (int(match.group(1)), int(match.group(2)))
    if first == second:
        raise ValueError(
            f"Gold pair must contain two different IDs for context {context_len}, "
            f"task {task_id}: {value!r}"
        )
    return min(first, second), max(first, second)


def normalize_oolong_pairs(
    tasks_by_length: Mapping[int, Sequence[Mapping[str, Any]]],
    contexts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contexts_by_length: dict[int, list[Mapping[str, Any]]] = {}
    for context in contexts:
        require_fields(context, {"id", "context_len", "context"}, "OOLONG context")
        contexts_by_length.setdefault(int(context["context_len"]), []).append(context)

    selected_contexts: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for context_len, tasks in tasks_by_length.items():
        matching_contexts = contexts_by_length.get(context_len, [])
        if len(matching_contexts) != 1:
            raise ValueError(
                f"Expected exactly one OOLONG context for length {context_len}, "
                f"found {len(matching_contexts)}"
            )
        context = dict(matching_contexts[0])
        selected_contexts.append(context)

        task_ids = {str(task.get("id")) for task in tasks}
        if task_ids != OFFICIAL_PAIR_TASK_IDS or len(tasks) != len(OFFICIAL_PAIR_TASK_IDS):
            raise ValueError(
                f"OOLONG-Pairs context {context_len} has incorrect task IDs: "
                f"expected {sorted(OFFICIAL_PAIR_TASK_IDS, key=int)}, got {sorted(task_ids)}"
            )
        for task in sorted(tasks, key=lambda item: int(str(item["id"]))):
            require_fields(task, {"id", "question", "answer", "type"}, OOLONG_PAIRS_REPOSITORY)
            task_id = str(task["id"])
            if not isinstance(task["answer"], list):
                raise ValueError(
                    f"OOLONG-Pairs answer for context {context_len}, task {task_id} is not a list"
                )
            gold_pairs = sorted(
                {
                    parse_gold_pair(value, context_len=context_len, task_id=task_id)
                    for value in task["answer"]
                }
            )
            examples.append(
                {
                    "id": f"{context_len}:{task_id}",
                    "task_id": task_id,
                    "dataset": "oolong_pairs",
                    "context_len": context_len,
                    "context_id": str(context["id"]),
                    "question": str(task["question"]),
                    "gold_pairs": [list(pair) for pair in gold_pairs],
                    "answer_type": str(task["type"]),
                }
            )
    return selected_contexts, examples


def stream_pair_examples(
    raw_path: Path,
    output: TextIO,
    *,
    context_len: int,
    context_id: str,
) -> set[str]:
    import ijson

    task_ids: set[str] = set()
    metadata: dict[str, str] | None = None
    pair_spool: TextIO | None = None
    first_pair = True
    previous_pair: tuple[int, int] | None = None
    saw_answer = False
    try:
        with raw_path.open("rb") as raw_file:
            for prefix, event, value in ijson.parse(raw_file):
                if prefix == "item" and event == "start_map":
                    metadata = {}
                    pair_spool = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
                    first_pair = True
                    previous_pair = None
                    saw_answer = False
                    continue
                if metadata is None or pair_spool is None:
                    continue
                if prefix in {"item.id", "item.question", "item.type"} and event in {
                    "string",
                    "number",
                }:
                    metadata[prefix.removeprefix("item.")] = str(value)
                    continue
                if prefix == "item.answer" and event == "start_array":
                    saw_answer = True
                    continue
                if prefix == "item.answer.item" and event == "string":
                    task_id = metadata.get("id", "<unknown>")
                    pair = parse_gold_pair(
                        value,
                        context_len=context_len,
                        task_id=task_id,
                    )
                    if previous_pair is not None and pair < previous_pair:
                        raise ValueError(
                            f"Gold pairs are not sorted for context {context_len}, task {task_id}"
                        )
                    if pair == previous_pair:
                        continue
                    if not first_pair:
                        pair_spool.write(",")
                    json.dump(pair, pair_spool, separators=(",", ":"))
                    first_pair = False
                    previous_pair = pair
                    continue
                if prefix == "item" and event == "end_map":
                    missing = sorted({"id", "question", "type"} - metadata.keys())
                    if missing or not saw_answer:
                        details = missing + ([] if saw_answer else ["answer"])
                        raise ValueError(
                            f"OOLONG-Pairs task at context {context_len} is missing "
                            f"fields: {', '.join(details)}"
                        )
                    task_id = metadata["id"]
                    if task_id in task_ids:
                        raise ValueError(
                            f"Duplicate OOLONG-Pairs task ID {task_id} at context {context_len}"
                        )
                    task_ids.add(task_id)
                    prefix_value = {
                        "answer_type": metadata["type"],
                        "context_id": context_id,
                        "context_len": context_len,
                        "dataset": "oolong_pairs",
                    }
                    prefix_json = json.dumps(
                        prefix_value,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    output.write(prefix_json[:-1])
                    output.write(',"gold_pairs":[')
                    pair_spool.seek(0)
                    while chunk := pair_spool.read(1024 * 1024):
                        output.write(chunk)
                    output.write('],"id":')
                    json.dump(f"{context_len}:{task_id}", output, ensure_ascii=False)
                    output.write(',"question":')
                    json.dump(metadata["question"], output, ensure_ascii=False)
                    output.write(',"task_id":')
                    json.dump(task_id, output, ensure_ascii=False)
                    output.write("}\n")
                    pair_spool.close()
                    pair_spool = None
                    metadata = None
    finally:
        if pair_spool is not None:
            pair_spool.close()
    return task_ids


def normalize_codeqa(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    required_fields = {
        "_id",
        "domain",
        "sub_domain",
        "difficulty",
        "length",
        "question",
        "choice_A",
        "choice_B",
        "choice_C",
        "choice_D",
        "answer",
        "context",
    }
    for row in rows:
        require_fields(row, {"domain", "sub_domain"}, CODEQA_REPOSITORY)
        if row["domain"] != "Code Repository Understanding" or row["sub_domain"] != "Code Repo QA":
            continue
        require_fields(row, required_fields, CODEQA_REPOSITORY)
        answer = str(row["answer"]).strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"Invalid CodeQA gold answer for {row['_id']}: {row['answer']!r}")
        example_id = str(row["_id"])
        if example_id in seen_ids:
            raise ValueError(f"Duplicate CodeQA example ID: {example_id}")
        seen_ids.add(example_id)
        examples.append(
            {
                "id": example_id,
                "question": str(row["question"]),
                "choices": {
                    choice: str(row[f"choice_{choice}"]) for choice in ("A", "B", "C", "D")
                },
                "gold_choice": answer,
                "context": str(row["context"]),
                "difficulty": str(row["difficulty"]),
                "length": str(row["length"]),
                "domain": str(row["domain"]),
                "sub_domain": str(row["sub_domain"]),
            }
        )
    if not examples:
        raise ValueError("LongBench-v2 source contains no Code Repo QA examples")
    return examples


def write_jsonl_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("wb") as raw_file:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_file, mtime=0) as gzip_file:
            with io.TextIOWrapper(gzip_file, encoding="utf-8", newline="\n") as text_file:
                for row in rows:
                    json.dump(
                        row,
                        text_file,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    text_file.write("\n")


def write_snapshot(
    snapshot_dir: Path,
    *,
    benchmark: str,
    sources: list[dict[str, str]],
    transformations: list[str],
    canonical_filter: dict[str, Any],
    rows_by_name: Mapping[str, Sequence[Mapping[str, Any]]],
    created_at: str | None = None,
) -> None:
    if snapshot_dir.exists():
        raise FileExistsError(f"Snapshot staging directory already exists: {snapshot_dir}")
    if "examples" not in rows_by_name:
        raise ValueError("Snapshot must contain examples")
    snapshot_dir.mkdir(parents=True)
    files: dict[str, dict[str, Any]] = {}
    for logical_name, rows in rows_by_name.items():
        path = snapshot_dir / f"{logical_name}.jsonl.gz"
        write_jsonl_gz(path, rows)
        files[logical_name] = {
            "name": path.name,
            "byte_size": path.stat().st_size,
            "record_count": len(rows),
            "sha256": sha256_file(path),
        }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": benchmark,
        "created_at": created_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "sources": sources,
        "transformations": transformations,
        "canonical_filter": canonical_filter,
        "files": files,
        "example_count": len(rows_by_name["examples"]),
    }
    if "contexts" in rows_by_name:
        manifest["context_count"] = len(rows_by_name["contexts"])
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_pairs_snapshot(
    snapshot_dir: Path,
    *,
    raw_paths: Mapping[int, Path],
    contexts: Sequence[Mapping[str, Any]],
    created_at: str | None = None,
) -> None:
    if snapshot_dir.exists():
        raise FileExistsError(f"Snapshot staging directory already exists: {snapshot_dir}")
    snapshot_dir.mkdir(parents=True)
    contexts_path = snapshot_dir / "contexts.jsonl.gz"
    write_jsonl_gz(contexts_path, contexts)

    contexts_by_length = {int(context["context_len"]): context for context in contexts}
    examples_path = snapshot_dir / "examples.jsonl.gz"
    example_count = 0
    with examples_path.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_output,
            mtime=0,
        ) as gzip_output:
            with io.TextIOWrapper(
                gzip_output,
                encoding="utf-8",
                newline="\n",
            ) as output:
                for context_len in PAIR_CONTEXT_LENGTHS:
                    context = contexts_by_length.get(context_len)
                    if context is None:
                        raise ValueError(f"Missing OOLONG context for pair length {context_len}")
                    task_ids = stream_pair_examples(
                        raw_paths[context_len],
                        output,
                        context_len=context_len,
                        context_id=str(context["id"]),
                    )
                    if task_ids != OFFICIAL_PAIR_TASK_IDS:
                        raise ValueError(
                            f"OOLONG-Pairs context {context_len} has incorrect task IDs: "
                            f"expected {sorted(OFFICIAL_PAIR_TASK_IDS, key=int)}, "
                            f"got {sorted(task_ids)}"
                        )
                    example_count += len(task_ids)

    files = {
        "contexts": {
            "name": contexts_path.name,
            "byte_size": contexts_path.stat().st_size,
            "record_count": len(contexts),
            "sha256": sha256_file(contexts_path),
        },
        "examples": {
            "name": examples_path.name,
            "byte_size": examples_path.stat().st_size,
            "record_count": example_count,
            "sha256": sha256_file(examples_path),
        },
    }
    manifest = {
        "schema_version": 1,
        "benchmark": "oolong_pairs",
        "created_at": created_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "sources": [
            {
                "repository": OOLONG_PAIRS_REPOSITORY,
                "revision": OOLONG_PAIRS_REVISION,
                "split": "raw data/oolong-pairs-{context_len}.json",
            },
            {
                "repository": OOLONG_REPOSITORY,
                "revision": OOLONG_REVISION,
                "split": "validation",
            },
        ],
        "transformations": [
            "join every official task to its matching trec_coarse no-label context",
            "stream, canonicalize, validate, and deduplicate gold integer pairs",
        ],
        "canonical_filter": {
            "context_len": 32768,
            "task_ids": list(range(1, 21)),
        },
        "files": files,
        "example_count": example_count,
        "context_count": len(contexts),
    }
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def publish_snapshot(
    data_dir: Path,
    benchmark: str,
    builder: Callable[[Path], None],
    *,
    force: bool,
) -> bool:
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / benchmark
    if target.exists():
        validate_snapshot(data_dir, benchmark)
        if not force:
            return False

    staging_parent = Path(tempfile.mkdtemp(prefix=f".{benchmark}-", dir=data_dir))
    staging = staging_parent / benchmark
    backup: Path | None = None
    try:
        builder(staging)
        validate_snapshot(staging_parent, benchmark)
        if target.exists():
            backup = data_dir / f".{benchmark}.backup-{uuid.uuid4().hex}"
            os.replace(target, backup)
        try:
            os.replace(staging, target)
        except Exception:
            if backup is not None:
                os.replace(backup, target)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return True
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def fetch_oolong(data_dir: Path, cache_dir: Path, *, force: bool) -> bool:
    from datasets import load_dataset

    def builder(staging: Path) -> None:
        rows = load_dataset(
            OOLONG_REPOSITORY,
            split="validation",
            revision=OOLONG_REVISION,
            cache_dir=str(cache_dir),
        )
        contexts, examples = normalize_oolong(rows)
        canonical_examples = [example for example in examples if example["context_len"] == 131072]
        if len(canonical_examples) != 50:
            raise ValueError(
                "Expected 50 canonical OOLONG trec_coarse examples at context length "
                f"131072, found {len(canonical_examples)}"
            )
        write_snapshot(
            staging,
            benchmark="oolong",
            sources=[
                {
                    "repository": OOLONG_REPOSITORY,
                    "revision": OOLONG_REVISION,
                    "split": "validation",
                }
            ],
            transformations=[
                "filter dataset == trec_coarse",
                "deduplicate context_window_text by stable context reference",
                "drop context_window_text_with_labels",
            ],
            canonical_filter={"dataset": "trec_coarse", "context_len": 131072},
            rows_by_name={"contexts": contexts, "examples": examples},
        )

    return publish_snapshot(data_dir, "oolong", builder, force=force)


def fetch_oolong_pairs(data_dir: Path, cache_dir: Path, *, force: bool) -> bool:
    from huggingface_hub import hf_hub_download

    oolong_snapshot = validate_snapshot(data_dir, "oolong")
    contexts_by_length: dict[int, list[dict[str, Any]]] = {}
    for context in oolong_snapshot.contexts:
        contexts_by_length.setdefault(int(context["context_len"]), []).append(context)
    contexts: list[dict[str, Any]] = []
    for context_len in PAIR_CONTEXT_LENGTHS:
        matching = contexts_by_length.get(context_len, [])
        if not matching:
            raise ValueError(f"Missing OOLONG context for pair length {context_len}")
        # oolong-synth samples multiple independent trec_coarse context windows per
        # length; oolong-pairs' own reference loader takes the first trec_coarse
        # example it sees for a given length (`examples[0]`), which is the row order
        # preserved here via contexts_by_id insertion order in normalize_oolong().
        # Verified against upstream: for context_len 1024 and 32768, every gold user
        # ID referenced in oolong-pairs' answers appears in the first context window
        # and none appear in the later ones, so `matching[0]` is the one the gold
        # pairs were computed against.
        contexts.append(matching[0])

    def builder(staging: Path) -> None:
        raw_paths: dict[int, Path] = {}
        for context_len in PAIR_CONTEXT_LENGTHS:
            path = hf_hub_download(
                repo_id=OOLONG_PAIRS_REPOSITORY,
                filename=f"data/oolong-pairs-{context_len}.json",
                repo_type="dataset",
                revision=OOLONG_PAIRS_REVISION,
                cache_dir=str(cache_dir),
            )
            raw_paths[context_len] = Path(path)
        write_pairs_snapshot(
            staging,
            raw_paths=raw_paths,
            contexts=contexts,
        )

    return publish_snapshot(data_dir, "oolong_pairs", builder, force=force)


def fetch_codeqa(data_dir: Path, cache_dir: Path, *, force: bool) -> bool:
    from datasets import load_dataset

    def builder(staging: Path) -> None:
        rows = load_dataset(
            CODEQA_REPOSITORY,
            split="train",
            revision=CODEQA_REVISION,
            cache_dir=str(cache_dir),
        )
        examples = normalize_codeqa(rows)
        if len(examples) != 50:
            raise ValueError(f"Expected 50 CodeQA examples, found {len(examples)}")
        write_snapshot(
            staging,
            benchmark="codeqa",
            sources=[
                {
                    "repository": CODEQA_REPOSITORY,
                    "revision": CODEQA_REVISION,
                    "split": "train",
                }
            ],
            transformations=[
                "filter domain == Code Repository Understanding",
                "filter sub_domain == Code Repo QA",
                "normalize choices A-D and gold choice",
            ],
            canonical_filter={
                "domain": "Code Repository Understanding",
                "sub_domain": "Code Repo QA",
            },
            rows_by_name={"examples": examples},
        )

    return publish_snapshot(data_dir, "codeqa", builder, force=force)


def fetch_all(data_dir: Path, cache_dir: Path, *, force: bool) -> None:
    operations = (
        ("oolong", fetch_oolong),
        ("oolong_pairs", fetch_oolong_pairs),
        ("codeqa", fetch_codeqa),
    )
    for benchmark, fetch in operations:
        changed = fetch(data_dir, cache_dir, force=force)
        print(f"{benchmark}: {'updated' if changed else 'valid snapshot already present'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and validate local paper benchmark snapshots."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data/benchmarks"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        fetch_all(args.data_dir, args.cache_dir, force=args.force)
        return
    with tempfile.TemporaryDirectory(prefix="rlm-benchmark-cache-") as cache_dir:
        fetch_all(args.data_dir, Path(cache_dir), force=args.force)


if __name__ == "__main__":
    main()
