# Local Paper Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build revision-pinned local snapshots and deterministic paired plain/RLM runners for OOLONG, OOLONG-Pairs, and LongBench-v2 CodeQA.

**Architecture:** `scripts/fetch_benchmarks.py` is the only networked dataset workflow and publishes validated gzip-JSONL snapshots. `examples/benchmark_common.py` validates those snapshots, selects examples, executes paired inference, and writes matching incremental CSV and atomic JSON artifacts; three benchmark modules own only their schemas, prompts, filters, parsers, and scorers.

**Tech Stack:** Python 3.11+, standard-library gzip/JSON/CSV/hashlib/tempfile/pathlib, `requests`, existing `RLM` and vLLM client APIs, transient Hugging Face `datasets` and `huggingface_hub`, pytest, ruff, GNU Make.

## Global Constraints

- Pin every upstream dataset to a full 40-character immutable revision.
- Add no Hugging Face or benchmark-only dependency to `pyproject.toml`.
- Benchmark commands never download datasets and never depend on `fetch-benchmarks`.
- Canonical defaults are OOLONG `trec_coarse` at 131072, OOLONG-Pairs at 32768, and all 50 CodeQA rows.
- Every selected example runs once in `plain` and once in `rlm`, in the same order.
- Inference failures score zero and remain in the selected-example denominator.
- Automated tests use no network, live server, Docker, GPU, or live dataset.
- Preserve the existing user-owned `uv.lock` modification.

---

### Task 1: Local snapshot contract

**Files:**
- Create: `examples/benchmark_common.py`
- Create: `tests/test_benchmark_common.py`

**Interfaces:**
- Produces: `ValidatedSnapshot`, `read_jsonl_gz(path)`, `sha256_file(path)`, and `validate_snapshot(root, benchmark)`.
- `validate_snapshot` accepts schema version 1 only and verifies required manifest fields, filenames, byte sizes, record counts, and SHA-256 values before returning any examples.

- [ ] **Step 1: Write failing contract tests**

```python
def test_validate_snapshot_rejects_checksum_mismatch(snapshot_dir):
    snapshot = snapshot_dir("oolong", [{"id": "1"}])
    snapshot.joinpath("examples.jsonl.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        validate_snapshot(snapshot.parent, "oolong")


def test_validate_snapshot_rejects_unsupported_schema(snapshot_dir):
    snapshot = snapshot_dir("codeqa", [{"id": "1"}], schema_version=2)
    with pytest.raises(ValueError, match="schema version"):
        validate_snapshot(snapshot.parent, "codeqa")
```

- [ ] **Step 2: Verify red**

Run: `rtk uv run pytest tests/test_benchmark_common.py -q`

Expected: import failure because `examples.benchmark_common` does not exist.

- [ ] **Step 3: Implement the manifest reader and validator**

Use explicit dataclasses and fail-loud `ValueError` messages. Verify every manifest-declared file before loading examples, and tell users to run `make fetch-benchmarks` when the benchmark directory is absent.

- [ ] **Step 4: Verify green**

Run: `rtk uv run pytest tests/test_benchmark_common.py -q`

Expected: snapshot contract tests pass.

### Task 2: Revision-pinned acquisition and normalization

**Files:**
- Create: `scripts/fetch_benchmarks.py`
- Create: `tests/test_fetch_benchmarks.py`

**Interfaces:**
- Consumes: `validate_snapshot`, `sha256_file` from Task 1.
- Produces: `normalize_oolong(rows)`, `normalize_oolong_pairs(tasks_by_length, contexts)`, `normalize_codeqa(rows)`, `write_snapshot(...)`, `replace_validated_snapshot(...)`, and CLI `main()`.

- [ ] **Step 1: Write failing transformation tests**

```python
def test_normalize_oolong_deduplicates_no_label_contexts():
    contexts, examples = normalize_oolong([
        oolong_row(id=1, context_len=131072, context="ctx", labelled="leak"),
        oolong_row(id=2, context_len=131072, context="ctx", labelled="leak"),
        oolong_row(id=3, dataset="other"),
    ])
    assert contexts == [{"id": "trec_coarse-131072-7", "context_len": 131072, "context": "ctx"}]
    assert [row["id"] for row in examples] == ["1", "2"]
    assert all("context_window_text_with_labels" not in row for row in examples)


def test_normalize_pairs_canonicalizes_gold_pairs():
    contexts = {32768: {"id": "trec_coarse-32768-7", "context": "ctx"}}
    examples = normalize_oolong_pairs(
        {32768: [{"id": "1", "question": "q", "answer": ["(9, 2)", "(2, 9)"]}]},
        contexts,
    )
    assert examples[0]["gold_pairs"] == [[2, 9]]


def test_normalize_codeqa_keeps_only_code_repo_qa():
    examples = normalize_codeqa([codeqa_row(), codeqa_row(sub_domain="Other")])
    assert len(examples) == 1
    assert examples[0]["gold_choice"] == "A"
```

- [ ] **Step 2: Verify red**

Run: `rtk uv run pytest tests/test_fetch_benchmarks.py -q`

Expected: import failure because `scripts.fetch_benchmarks` does not exist.

- [ ] **Step 3: Implement deterministic normalization and snapshot publication**

Pin:

```python
OOLONG_REVISION = "f0d59eaf0febf130664cfceb710436c8e3216b2b"
OOLONG_PAIRS_REVISION = "d1e1522b86ac0c169bbc890b0471408aaa29e8fa"
CODEQA_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
PAIR_CONTEXT_LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)
```

Use `load_dataset(..., split=..., revision=..., cache_dir=...)` for OOLONG and LongBench-v2. Use `hf_hub_download(..., repo_type="dataset", revision=...)` for each full OOLONG-Pairs answer JSON. Write deterministic gzip streams (`mtime=0`), construct manifest metadata after files close, revalidate staging, then rename it into place. Skip a valid destination unless `--force` is present; transformation or validation failures must leave an existing snapshot unchanged.

- [ ] **Step 4: Add invariant and preservation tests**

Cover missing columns, non-unique OOLONG context per length, absent pair IDs 1–20, self-pairs, missing pair contexts, non-A–D CodeQA answers, record counts, checksums, and preservation of an existing destination when staging validation fails.

- [ ] **Step 5: Verify green**

Run: `rtk uv run pytest tests/test_fetch_benchmarks.py tests/test_benchmark_common.py -q`

Expected: all acquisition and snapshot tests pass offline.

### Task 3: Deterministic selection and artifact lifecycle

**Files:**
- Modify: `examples/benchmark_common.py`
- Modify: `tests/test_benchmark_common.py`

**Interfaces:**
- Produces: `select_examples(examples, num_examples, seed)`, `model_slug(model)`, `create_artifact_paths(...)`, `append_csv_row(...)`, `write_summary_atomic(...)`, and usage aggregation helpers.

- [ ] **Step 1: Write failing selection and artifact tests**

```python
def test_cap_selects_exact_stable_ids():
    examples = [{"id": str(i)} for i in range(10)]
    first = select_examples(examples, num_examples=3, seed=42)
    second = select_examples(examples, num_examples=3, seed=42)
    assert [x["id"] for x in first] == [x["id"] for x in second]
    assert len(first) == 3


def test_artifacts_share_stem_and_refuse_overwrite(tmp_path):
    paths = create_artifact_paths(tmp_path, "codeqa", "Qwen/Qwen3-1.7B", fixed_utc())
    assert paths.csv.stem == paths.summary.stem
    paths.csv.touch()
    with pytest.raises(FileExistsError):
        create_artifact_paths(tmp_path, "codeqa", "Qwen/Qwen3-1.7B", fixed_utc())
```

- [ ] **Step 2: Verify red**

Run: `rtk uv run pytest tests/test_benchmark_common.py -q`

Expected: failures for missing selection/artifact APIs.

- [ ] **Step 3: Implement selection and durable artifacts**

With no cap, preserve source order and ignore the seed. With a cap, reject booleans, non-positive values, and values above the eligible count, then shuffle a copy with `random.Random(seed)`. Slug models to lowercase ASCII `[a-z0-9.-]` groups joined by `-`. Append and flush one CSV row after each inference. Write JSON through a same-directory temporary file followed by `os.replace`.

- [ ] **Step 4: Verify green**

Run: `rtk uv run pytest tests/test_benchmark_common.py -q`

Expected: selection, naming, incremental CSV, failure-as-zero, usage, and atomic-summary tests pass.

### Task 4: Shared paired runner

**Files:**
- Modify: `examples/benchmark_common.py`
- Modify: `tests/test_benchmark_common.py`

**Interfaces:**
- Produces: `ScoreResult`, `BenchmarkSpec`, `RunnerConfig`, `run_plain(...)`, `run_rlm(...)`, `require_vllm_server(...)`, and `run_benchmark(spec, config)`.
- Benchmark callbacks consume normalized example dictionaries and return prompts or `ScoreResult` without performing I/O.

- [ ] **Step 1: Write a failing paired-runner test**

```python
def test_runner_uses_same_examples_for_plain_and_rlm_and_counts_failures(tmp_path, snapshot_dir):
    calls = []
    spec = fake_spec(score=lambda prediction: ScoreResult(float(prediction)))
    summary = run_benchmark(
        spec,
        fake_config(tmp_path, num_examples=2),
        health_check=lambda: None,
        plain_call=lambda example, _: calls.append(("plain", example["id"])) or fake_call("1"),
        rlm_call=lambda example, _: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert calls == [("plain", summary["selected_example_ids"][0]), ("plain", summary["selected_example_ids"][1])]
    assert summary["scores"] == {"plain": 1.0, "rlm": 0.0, "rlm_minus_plain": -1.0}
```

- [ ] **Step 2: Verify red**

Run: `rtk uv run pytest tests/test_benchmark_common.py -q`

Expected: failure for missing runner APIs.

- [ ] **Step 3: Implement startup ordering and paired evaluation**

Validate data, benchmark filters, and selection; perform the vLLM health check; only then choose artifact paths. Iterate selected examples outermost and `("plain", "rlm")` innermost. Catch inference exceptions only, emit a score-zero row, and continue. Treat loader, scorer, writer, and configuration errors as fatal. Build a schema-version-1 summary with effective config, manifest identity and revisions, selected IDs, full/partial status, counts, per-mode score/latency/usage, scoring method, CSV path, and `rlm_minus_plain`.

- [ ] **Step 4: Verify green**

Run: `rtk uv run pytest tests/test_benchmark_common.py -q`

Expected: mocked runner tests pass with no server or GPU.

### Task 5: Refactor OOLONG to local data

**Files:**
- Modify: `examples/benchmark_oolong.py`
- Modify: `tests/test_vllm_oolong_examples.py`

**Interfaces:**
- Consumes: common snapshot, selection, and runner APIs.
- Produces: local `load_examples`, context join, canonical filter, prompt builders, retained `synth_score`, CLI, and `main()`.

- [ ] **Step 1: Replace runtime-download tests with local-loader/scorer tests**

Add date, comparison, unparseable, missing-context, default-131072, numeric-exclusion partial-status, and no-Hugging-Face-import assertions. Keep current literal and numeric-distance expectations.

- [ ] **Step 2: Verify red**

Run: `rtk uv run pytest tests/test_vllm_oolong_examples.py -q`

Expected: failures because the current runner downloads `datasets` and defaults to a capped 1K–4K slice.

- [ ] **Step 3: Refactor to common runner**

Remove `load_dataset`, `--output`, min/max context ranges, duplicated inference/artifact code, and compatibility paths. Default `--context-length=131072`, `--num-examples=None`, and `--seed=42`; retain `--exclude-numeric`. Join examples to contexts by `context_id` only after snapshot validation.

- [ ] **Step 4: Verify green**

Run: `rtk uv run pytest tests/test_vllm_oolong_examples.py tests/test_benchmark_common.py -q`

Expected: OOLONG and common tests pass.

### Task 6: Add OOLONG-Pairs

**Files:**
- Create: `examples/benchmark_oolong_pairs.py`
- Create: `tests/test_benchmark_oolong_pairs.py`

**Interfaces:**
- Produces: `parse_pairs(prediction) -> set[tuple[int, int]]`, `score_pairs(gold, prediction) -> ScoreResult`, local loader/context join, prompts, canonical filter, CLI, and `main()`.

- [ ] **Step 1: Write failing parser and F1 tests**

```python
@pytest.mark.parametrize(
    ("prediction", "expected"),
    [("(9, 2), (2, 9), junk", {(2, 9)}), ("(4, 4)", set()), ("none", set())],
)
def test_parse_pairs(prediction, expected):
    assert parse_pairs(prediction) == expected


def test_pair_f1_tracks_precision_and_recall():
    result = score_pairs({(1, 2), (3, 4)}, "(2,1), (8,9)")
    assert result.score == pytest.approx(0.5)
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
```

- [ ] **Step 2: Verify red**

Run: `rtk uv run pytest tests/test_benchmark_oolong_pairs.py -q`

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement runner**

Canonicalize pair order, discard self-pairs and malformed fragments, deduplicate, and implement the specified empty-set rules. Default context length 32768 and use all 20 tasks.

- [ ] **Step 4: Verify green**

Run: `rtk uv run pytest tests/test_benchmark_oolong_pairs.py -q`

Expected: parser, F1, loader, prompt, and default-filter tests pass.

### Task 7: Add LongBench-v2 CodeQA

**Files:**
- Create: `examples/benchmark_codeqa.py`
- Create: `tests/test_benchmark_codeqa.py`

**Interfaces:**
- Produces: `parse_choice(prediction) -> tuple[str | None, str]`, `score_choice(gold, prediction) -> ScoreResult`, local loader, prompts, CLI, and `main()`.

- [ ] **Step 1: Write failing strict-parser tests**

```python
@pytest.mark.parametrize(
    ("prediction", "choice", "status"),
    [("Final answer: C", "C", "explicit"), ("C", "C", "standalone"),
     ("A or B", None, "ambiguous"), ("unknown", None, "unparseable")],
)
def test_parse_choice(prediction, choice, status):
    assert parse_choice(prediction) == (choice, status)
```

- [ ] **Step 2: Verify red**

Run: `rtk uv run pytest tests/test_benchmark_codeqa.py -q`

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement CodeQA runner**

Format choices A–D in both prompts. Prefer one unambiguous case-insensitive `final answer`/`answer` match; otherwise accept only an output consisting of one standalone A–D choice with surrounding whitespace or punctuation. Store parse status and exact-match score. Full evaluation is all 50 stored rows with no cap.

- [ ] **Step 4: Verify green**

Run: `rtk uv run pytest tests/test_benchmark_codeqa.py -q`

Expected: parser, scoring, loader, prompt, and full-slice tests pass.

### Task 8: Makefile and ignore integration

**Files:**
- Modify: `Makefile`
- Modify: `.gitignore`
- Modify: `tests/test_vllm_oolong_examples.py`

**Interfaces:**
- Produces: `fetch-benchmarks`, `benchmark-oolong`, `benchmark-oolong-pairs`, and `benchmark-codeqa` targets with shared variables.

- [ ] **Step 1: Write failing Makefile assertions**

Assert the four targets, `BENCHMARK_DATA_DIR ?= ./data/benchmarks`, `LOG_DIR ?= ./logs`, `FETCH_ARGS ?=`, and `BENCHMARK_ARGS ?=` exist; parse target dependency lines and assert no benchmark target lists `fetch-benchmarks` as a prerequisite.

- [ ] **Step 2: Verify red**

Run: `rtk uv run pytest tests/test_vllm_oolong_examples.py -q`

Expected: missing target/default assertions fail.

- [ ] **Step 3: Implement Make targets and ignore rule**

Use transient fetch dependencies:

```make
fetch-benchmarks:
	uv run --with datasets --with huggingface_hub python scripts/fetch_benchmarks.py --data-dir $(BENCHMARK_DATA_DIR) $(FETCH_ARGS)
```

Each runner calls its module with `--model`, `--data-dir`, `--log-dir`, and `$(BENCHMARK_ARGS)`. Add `data/benchmarks/` to `.gitignore` and no `benchmark-all` target.

- [ ] **Step 4: Verify green**

Run: `rtk uv run pytest tests/test_vllm_oolong_examples.py -q`

Expected: Makefile integration tests pass.

### Task 9: Completion verification and requirement audit

**Files:**
- Modify only files required by failures found below.

- [ ] **Step 1: Run focused offline benchmark tests**

Run: `rtk uv run pytest tests/test_benchmark_common.py tests/test_fetch_benchmarks.py tests/test_vllm_oolong_examples.py tests/test_benchmark_oolong_pairs.py tests/test_benchmark_codeqa.py -q`

Expected: all pass.

- [ ] **Step 2: Run formatting and lint**

Run: `rtk uv run ruff format examples/benchmark_common.py examples/benchmark_oolong.py examples/benchmark_oolong_pairs.py examples/benchmark_codeqa.py scripts/fetch_benchmarks.py tests/test_benchmark_common.py tests/test_fetch_benchmarks.py tests/test_vllm_oolong_examples.py tests/test_benchmark_oolong_pairs.py tests/test_benchmark_codeqa.py`

Run: `rtk uv run ruff check .`

Expected: clean output.

- [ ] **Step 3: Run complete offline test suite**

Run: `rtk uv run pytest -q`

Expected: all project tests pass without network, Docker, GPU, or a live dataset/server (environment-dependent integration tests may be skipped by their existing markers).

- [ ] **Step 4: Audit all twelve acceptance criteria**

Inspect current source and tests for every criterion in `docs/plans/2026-07-14-local-paper-benchmarks-design.md`. Confirm no imports or calls to Hugging Face exist in benchmark modules, no benchmark Make target depends on fetch, both artifacts use the same stem, failure rows score zero, source revisions appear in summaries, and BrowseComp-Plus/S-NIAH were not added.
