# GPU Benchmark Progress Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add readable per-example progress events to parallel GPU benchmark runs without changing benchmark results or trajectory logging.

**Architecture:** A small `BenchmarkProgressLogger` in `examples/benchmark_common.py` formats and flushes complete inference-event lines. `run_benchmark()` supplies orchestration context and emits events around its existing inference/CSV flow, while `scripts/run_gpu_benchmarks.sh` prefixes every subprocess line with the GPU and suite iteration it already owns.

**Tech Stack:** Python 3.12, pytest, Bash, Ruff; no new dependencies.

## Global Constraints

- Do not add progress bars, terminal redraws, new dependencies, prompt or prediction output, persistent progress state, or RLM trajectory changes.
- Only caught inference exceptions become `error` events; configuration, validation, scoring, and artifact failures remain fail-fast.
- Emit `done` or `error` only after the matching CSV row is appended successfully.
- Preserve all existing result counts, scores, usage totals, and artifact schemas.

---

### Task 1: Python benchmark progress events

**Files:**
- Modify: `tests/test_benchmark_common.py`
- Modify: `examples/benchmark_common.py`

**Interfaces:**
- Produces: `BenchmarkProgressLogger(model: str, benchmark: str, total_examples: int, *, now: Callable[[], datetime] = ...)`
- Produces: `BenchmarkProgressLogger.log(example_index: int, example_id: str, mode: str, status: str, *, elapsed: float | None = None, score: float | None = None, tokens: int | None = None, error: str = "") -> None`
- Consumes: the existing `run_benchmark()` example/mode loop, inference exception handling, score result, usage totals, and incremental CSV append.

- [ ] **Step 1: Extend the mocked runner test with failing progress assertions**

  Capture stdout with `capsys`, make the inference exception contain a newline, and assert the ordered eight events for two selected examples: `running` then `done` for plain, and `running` then one-line `error` for RLM. Assert model, benchmark, `example I/N`, dataset ID, mode, elapsed suffixes, score, and tokens while retaining all current artifact/result assertions.

- [ ] **Step 2: Run the focused test and verify RED**

  Run: `rtk uv run pytest tests/test_benchmark_common.py::test_runner_pairs_same_examples_and_counts_inference_failures_as_zero -q`

  Expected: FAIL because `run_benchmark()` emits no progress events.

- [ ] **Step 3: Implement the minimal logger and event calls**

  Add the logger beside the benchmark runner types. Format the model with its final `/`-separated component, format timestamps as local `HH:MM:SS`, collapse error whitespace with `" ".join(error.split())`, and use `print(..., flush=True)` for one complete line. Enumerate selected examples from one, emit `running` immediately before inference, and emit the terminal event immediately after `append_csv_row()`.

- [ ] **Step 4: Run the focused test and verify GREEN**

  Run: `rtk uv run pytest tests/test_benchmark_common.py::test_runner_pairs_same_examples_and_counts_inference_failures_as_zero -q`

  Expected: PASS.

### Task 2: Parallel shell stream prefixes and dynamic totals

**Files:**
- Modify: `scripts/run_gpu_benchmarks.sh`

**Interfaces:**
- Consumes: `run_suite(model, base_url, tag)` and its current per-GPU process substitution to `tee`.
- Produces: every per-suite output line prefixed as `[gpuN run I/N]`; computed total and per-GPU run counts.

- [ ] **Step 1: Add shell prefixing at the benchmark subprocess boundary**

  In each suite iteration, form `prefix="[$tag run $i/$ITERATIONS]"`, pipe `run_benchmark` combined stdout/stderr through `sed -u "s/^/$prefix /"`, and rely on `set -o pipefail` to preserve benchmark failures. Keep the outer per-GPU `tee` streams unchanged.

- [ ] **Step 2: Replace hard-coded success counts**

  Compute `runs_per_gpu=$((${#BENCHMARKS[@]} * ITERATIONS))` and `total_runs=$((runs_per_gpu * 2))`, then interpolate both values in the final success line.

- [ ] **Step 3: Verify shell syntax**

  Run: `rtk bash -n scripts/run_gpu_benchmarks.sh`

  Expected: exit status 0 with no output.

### Task 3: Focused regression verification

**Files:**
- Verify: `examples/benchmark_common.py`
- Verify: `tests/test_benchmark_common.py`
- Verify: `scripts/run_gpu_benchmarks.sh`

- [ ] **Step 1: Run focused tests**

  Run: `rtk uv run pytest tests/test_benchmark_common.py -q`

  Expected: all tests pass.

- [ ] **Step 2: Run Ruff without mutating unrelated files**

  Run: `rtk uv run ruff check examples/benchmark_common.py tests/test_benchmark_common.py`

  Run: `rtk uv run ruff format --check examples/benchmark_common.py tests/test_benchmark_common.py`

  Expected: both commands pass.

- [ ] **Step 3: Review the diff for scope and accidental user-file changes**

  Run: `rtk git diff --check`

  Run: `rtk git status --short`

  Expected: no whitespace errors; only the plan and three intended implementation files are changed, alongside the user's pre-existing untracked files.
