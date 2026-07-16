# GPU Benchmark Progress Logging Design

## Goal

Make `scripts/run_gpu_benchmarks.sh` report which dataset example and
inference mode each GPU-backed vLLM worker is processing. Output must remain
readable when both workers interleave in the terminal and when it is captured
by `tee`.

## Scope

Add a lean, line-oriented progress layer. Do not add progress bars, terminal
redraws, new dependencies, prompt or prediction output, persistent progress
state, or changes to RLM trajectory logging.

The existing loggers have separate responsibilities:

- `RLMLogger` records detailed RLM trajectories to JSONL and stays silent.
- `VerbosePrinter` remains opt-in, detailed debugging for one RLM call.
- A small `BenchmarkProgressLogger` in `examples/benchmark_common.py` reports
  benchmark orchestration progress to the terminal.

## Terminal Layout

Each plain or RLM inference emits a `running` event before the request and a
terminal `done` or `error` event after its CSV row is successfully appended.

```text
[gpu0 run 1/3] 14:32:08 | Qwen3-0.6B | oolong | example 2/20 | id=oolong-123 | plain | running
[gpu0 run 1/3] 14:32:15 | Qwen3-0.6B | oolong | example 2/20 | id=oolong-123 | plain | done | 7.1s | score=0.750 | tokens=923
[gpu1 run 1/3] 14:32:27 | Qwen3-1.7B | codeqa | example 8/50 | id=codeqa-456 | rlm | error | 12.3s | TimeoutError: request timed out
```

The Python logger owns the timestamp, model, benchmark, example position,
dataset ID, inference mode, status, and result metrics. It writes and flushes
one complete line per event. Model display may omit a repository prefix, such
as displaying `Qwen3-0.6B` for `Qwen/Qwen3-0.6B`, while the result artifacts
continue storing the exact configured model name.

The shell script owns GPU and suite-iteration identity because it already knows
those values. It prefixes every benchmark subprocess line with
`[gpuN run I/N]` before forwarding the stream to the existing per-GPU `tee`
file. This avoids adding GPU-specific arguments to the three generic benchmark
CLIs.

## Benchmark Flow

`run_benchmark()` enumerates the selected examples so each event can include
`example I/N`. For each example it retains the existing plain-then-RLM order:

1. Emit `running` for the mode.
2. Perform inference and measure elapsed time.
3. Score successful inference or construct the existing zero-score inference
   error result.
4. Append the incremental CSV row.
5. Emit `done` with elapsed time, score, and total tokens, or `error` with
   elapsed time and the caught inference exception.

No question, context, prediction, gold answer, or model reasoning is printed.
Detailed RLM trajectories continue to use `RLMLogger` without modification.

## Failure Behavior

Caught inference failures retain current semantics: record score zero, append
the CSV row, emit `error`, and continue. Error text is collapsed to one line so
parallel output remains readable.

Configuration, snapshot validation, scoring, and artifact-writing errors remain
fail-fast. They are not converted into benchmark results. The shell continues
to return a nonzero status if either GPU suite fails and points to both console
logs.

The final success message calculates its run totals from the number of
benchmarks and `ITERATIONS`; it does not hard-code 18 runs.

## Testing

Extend the existing mocked `run_benchmark()` test and capture terminal output.
Assert that:

- successful and failed calls emit `running` followed by `done` or `error`;
- events include model, benchmark, example position, dataset ID, and mode;
- completion events include elapsed time plus score and tokens when successful;
- error text occupies one line; and
- logging does not change result counts, scores, usage, or CSV artifacts.

Verification also runs `bash -n scripts/run_gpu_benchmarks.sh`, the focused
benchmark-common tests, and Ruff on the touched files. Tests require no vLLM
server, GPU, Docker container, timing sleeps, Rich snapshots, or full benchmark
execution.
