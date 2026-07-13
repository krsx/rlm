# vLLM Inference and OOLONG Benchmark Design

Date: 2026-07-13
Status: Draft approved for planning

## Goal

Add a lean, server-oriented workflow for:

1. Running a simple RLM inference call against a local vLLM OpenAI-compatible server.
2. Running an OOLONG benchmark that compares plain model calls against recursive RLM calls.
3. Starting vLLM with official pre-built Docker images through simple `make` commands.

The implementation should stay minimal:

- no new README content
- script header comments should show the exact `uv` commands to run
- strict local vLLM OpenAI-compatible server usage for simplicity
- benchmark results stored in CSV
- RLM inference path should use `verbose=True` and `RLMLogger(...)`

## Constraints

- Reuse the existing `rlm` inference surface rather than adding a new client layer.
- Keep the diff small and focused.
- Do not add heavy OOLONG dependencies to the root package install path.
- Use the local vLLM OpenAI-compatible server at `http://localhost:8000/v1`.
- Follow the official vLLM Docker run pattern from the current docs:
  `vllm/vllm-openai`, `--gpus all`, `-p 8000:8000`, Hugging Face cache mount, optional `HF_TOKEN`, and `--ipc=host`.

## Files To Add Or Change

Add:

- `examples/simple_inference.py`
- `examples/benchmark_oolong.py`

Change:

- `Makefile`

Tests may be added only for deterministic helper logic if the implementation naturally factors that way, but no live-server or live-dataset integration tests are required.

## Architecture

### 1. Simple Inference Script

`examples/simple_inference.py` will be a minimal entry point for one RLM completion against local vLLM.

Header comments will show the intended commands, for example:

```bash
make vllm-up MODEL=Qwen/Qwen3-0.6B
uv run python -m examples.simple_inference --model Qwen/Qwen3-0.6B --prompt "..."
```

Behavior:

- construct `RLM` with:
  - `backend="vllm"`
  - `backend_kwargs={"model_name": ..., "base_url": "http://localhost:8000/v1", "api_key": "dummy"}`
  - `verbose=True`
  - `logger=RLMLogger(log_dir=...)`
- run one completion
- print:
  - final response
  - usage summary
  - logger file path

Expected arguments:

- required:
  - `--model`
  - `--prompt`
- optional:
  - `--max-depth`
  - `--max-iterations`
  - `--max-tokens`
  - `--log-dir`

### 2. OOLONG Benchmark Script

`examples/benchmark_oolong.py` will be a minimal benchmark runner for comparing plain inference against RLM inference on OOLONG.

Header comments will show the intended command, for example:

```bash
uv run --with datasets --with python-dateutil python -m examples.benchmark_oolong --model Qwen/Qwen3-0.6B
```

This script will load OOLONG directly from `oolongbench/oolong-synth` and reuse the same high-level dataset filtering shape already present in the training environment:

- `dataset_name`
- `min_ctx`
- `max_ctx`
- `num_examples`
- `seed`
- `exclude_numeric`

The benchmark will run two modes on the same sampled examples:

- `plain`
- `rlm`

#### Plain mode

Plain mode will build a single long prompt containing:

- the OOLONG task instruction
- the question
- the full context

It will send that prompt directly to the existing OpenAI-compatible client path against local vLLM.

#### RLM mode

RLM mode will use the library in its intended shape:

- pass the OOLONG long context as the `prompt` argument to `RLM.completion(...)`
- pass the task instruction plus question as `root_prompt`

This makes the long OOLONG context available inside the REPL as `context`, instead of flattening RLM into another plain long-prompt call.

### 3. vLLM Docker Commands

The `Makefile` will add a small set of vLLM-focused commands built around the official pre-built image:

- `vllm-pull`
- `vllm-up`
- `vllm-health`
- `simple-infer`
- `benchmark-oolong`

The main server-start path will take the model from `MODEL=...`.

Expected Docker behavior:

- use `vllm/vllm-openai:latest`
- expose `8000:8000`
- mount `~/.cache/huggingface:/root/.cache/huggingface`
- pass through `HF_TOKEN` when set
- use `--runtime nvidia --gpus all`
- use `--ipc=host`
- append `--model $(MODEL)` after the image name

## Data Flow

### Simple Inference

1. User starts vLLM with `make vllm-up MODEL=...`.
2. Script constructs `RLM` targeting `http://localhost:8000/v1`.
3. Script runs one recursive completion.
4. Console shows the verbose trajectory and final answer.
5. JSONL trajectory logs are written to the chosen log directory.

### OOLONG Benchmark

1. Script samples OOLONG examples with deterministic filtering and optional shuffling.
2. For each example:
   - build plain prompt
   - run plain mode
   - run RLM mode
   - score both outputs using OOLONG scoring logic adapted from the training environment
   - append two CSV rows, one per mode
3. Script writes a single CSV file for the full run.
4. Script prints aggregate metrics at the end.

## CSV Output

The benchmark will write one row per `(example, mode)`.

Planned columns:

- `example_id`
- `dataset_name`
- `mode`
- `model`
- `question`
- `gold_answer`
- `prediction`
- `score`
- `latency_sec`
- `prompt_tokens`
- `completion_tokens`
- `total_calls`
- `log_file`
- `error`

Notes:

- `log_file` is mainly relevant for `rlm` rows.
- token and call counts should come from the existing client / usage summary surfaces when available.
- if a field is unavailable for a row, write an empty value rather than inventing one.

## Error Handling

Keep the implementation lean and fail-fast, but do not lose an entire benchmark run over a single bad example.

Benchmark behavior:

- per-example errors should be captured into the CSV `error` column and the run should continue
- startup failures should abort immediately:
  - missing `--model`
  - local vLLM server unreachable
  - missing temporary benchmark dependencies
  - dataset load failure before iteration starts

Simple inference behavior:

- fail immediately if the server is unreachable or required arguments are missing

## Validation

Keep verification small and deterministic.

Good candidates if helper functions are introduced:

- OOLONG scoring helper behavior
- prompt construction for plain vs RLM modes
- CSV row schema generation

Do not add tests that require:

- a live vLLM server
- Docker
- downloading the OOLONG dataset during test execution

## Non-Goals

- no new generalized CLI framework
- no support for arbitrary OpenAI-compatible base URLs
- no README expansion
- no training-harness integration
- no new root dependency on `datasets`
- no production-grade orchestration around container lifecycle

## Implementation Notes

- The benchmark should stay self-contained even if that means copying a small amount of OOLONG-specific scoring and filtering logic from the training environment.
- The script comments are the primary usage documentation.
- Keep naming and output conventions simple and obvious.

## Open Decisions Resolved

- Use minimal scripts, not reusable CLIs.
- Benchmark both plain model calls and RLM calls.
- Target only the local vLLM OpenAI-compatible server for simplicity.
- Store benchmark results in CSV.
- Use `verbose=True` and `RLMLogger(...)` for the RLM inference path.
- Write the design doc under `docs/plans/`.
- Do not add README documentation.
- Do not commit the design doc as part of this step.
