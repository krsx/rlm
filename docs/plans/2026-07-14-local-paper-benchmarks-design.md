# Local Paper Benchmark Data and Result Artifacts Design

Date: 2026-07-14
Status: Approved for specification review

## Context

The current vLLM benchmark workflow is centered on
`examples/benchmark_oolong.py`. It loads `oolongbench/oolong-synth` from the
Hugging Face Hub at benchmark runtime, compares plain inference with RLM
inference, appends per-example results to a CSV file, and only prints aggregate
scores to the console.

This design separates dataset acquisition from evaluation, expands deterministic
coverage to the main paper benchmarks that do not require an LLM judge, and makes
the aggregate score a durable artifact. It supersedes the runtime dataset-loading
and result-output sections of
`docs/plans/2026-07-13-vllm-oolong-inference-design.md`.

The benchmark selection follows the main evaluation table in the
[Recursive Language Models paper](https://arxiv.org/html/2512.24601):

- OOLONG `trec_coarse`
- OOLONG-Pairs
- LongBench-v2 CodeQA

BrowseComp-Plus is deferred. Its authoritative evaluation uses an additional LLM
judge, which conflicts with the current goal of lean, deterministic, locally
scored benchmarks. S-NIAH is also outside this design because it is not one of the
four main table benchmarks selected for this refactor.

## Goals

1. Download the complete paper-specific subsets once and store them under
   `data/benchmarks/`.
2. Make benchmark execution fully independent of dataset downloads and external
   dataset services.
3. Provide one runner for each in-scope benchmark while sharing inference and
   artifact code.
4. Evaluate identical examples in paired `plain` and `rlm` modes.
5. Make the full paper slice the default. Example caps exist only for deliberate
   selective checks.
6. Store timestamped per-example CSV results and aggregate JSON summaries in
   `logs/`, using the same filename stem.
7. Preserve authoritative source provenance, immutable revisions, transformations,
   and checksums in local manifests.

## Non-Goals

- BrowseComp-Plus inference, retrieval, or LLM-as-judge evaluation
- S-NIAH generation or evaluation
- A generalized benchmark plugin framework
- Automatic dataset downloads from benchmark commands
- Live-server, live-dataset, or GPU-dependent automated tests
- Adding benchmark dependencies to the core package dependency list
- Reproducing every baseline or agent scaffold from the RLM paper
- A command that automatically runs every expensive benchmark

## High-Level Architecture

Dataset acquisition and benchmark execution are separate workflows with no
implicit dependency between them:

```text
make fetch-benchmarks
        |
        v
data/benchmarks/
        |
        +--> make benchmark-oolong MODEL=...
        +--> make benchmark-oolong-pairs MODEL=...
        +--> make benchmark-codeqa MODEL=...
                         |
                         v
             logs/<run-stem>.csv
             logs/<run-stem>.json
```

`make fetch-benchmarks` is the only workflow that contacts upstream dataset
services. Benchmark commands read the normalized local snapshot, contact only the
configured local vLLM server for inference, and never attempt an online dataset
fallback.

The implementation has five focused components:

1. `scripts/fetch_benchmarks.py` downloads, transforms, validates, and stores all
   in-scope subsets.
2. `examples/benchmark_common.py` owns shared inference, selection, usage,
   artifact, naming, and aggregation behavior.
3. `examples/benchmark_oolong.py` owns OOLONG loading, prompts, filters, and
   scoring.
4. `examples/benchmark_oolong_pairs.py` owns OOLONG-Pairs loading, prompts, tuple
   parsing, and F1 scoring.
5. `examples/benchmark_codeqa.py` owns CodeQA loading, prompts, answer parsing,
   and accuracy scoring.

The existing OOLONG runner is refactored rather than preserved alongside a second
implementation.

## Local Dataset Layout

The normalized snapshot uses compressed JSON Lines so runners can read it with
the Python standard library and do not require Hugging Face `datasets` at
evaluation time.

```text
data/benchmarks/
├── oolong/
│   ├── contexts.jsonl.gz
│   ├── examples.jsonl.gz
│   └── manifest.json
├── oolong_pairs/
│   ├── contexts.jsonl.gz
│   ├── examples.jsonl.gz
│   └── manifest.json
└── codeqa/
    ├── examples.jsonl.gz
    └── manifest.json
```

`data/benchmarks/` is gitignored. The manifest remains beside its downloaded
files so copying or deleting a snapshot cannot accidentally separate its data
from its provenance.

### Why Normalized JSONL

Hugging Face `Dataset.save_to_disk()` and `load_from_disk()` provide a supported
local Arrow workflow. Arrow is fast to reload, but its on-disk representation is
uncompressed and it requires `datasets` during benchmark execution. Raw upstream
snapshots avoid transformation but force each runner to understand different
source layouts and repeatedly filter large releases.

Compressed normalized JSONL gives this project a small, explicit schema, portable
standard-library readers, streaming iteration, and reduced storage. Contexts that
are repeated across tasks are stored once and joined locally by ID.

## Authoritative Sources and Transformations

Every source is pinned to a full immutable upstream revision committed in the
fetch script. Branch names such as `main` are not valid source revisions. The
fetcher passes the revision to the relevant download API and records it in the
manifest.

### OOLONG

Source: [`oolongbench/oolong-synth`](https://huggingface.co/datasets/oolongbench/oolong-synth)

- Read the `validation` split.
- Retain every row whose `dataset` is `trec_coarse`.
- Store unique no-label contexts in `contexts.jsonl.gz`.
- Store questions, answers, answer types, context lengths, upstream IDs, and
  context references in `examples.jsonl.gz`.
- Do not persist `context_window_text_with_labels`, because it leaks ground-truth
  labels into evaluation context.
- Preserve all available context lengths in the local paper-specific subset.

The canonical full evaluation slice is all 50 `trec_coarse` tasks at context
length 131072. Other stored lengths support intentional scaling checks without a
new download.

### OOLONG-Pairs

Sources:

- [`mit-oasys/oolong-pairs`](https://huggingface.co/datasets/mit-oasys/oolong-pairs)
- the pinned OOLONG source above for matching no-label contexts

The fetcher stores all 20 official tasks at every available paper context length.
Each task record references a context by context length and stable context ID.
Gold answers are stored as canonical sets of integer pairs, not as presentation
strings.

The fetcher validates that every task length has exactly one matching context,
that all expected task IDs are present, and that every gold pair contains two
different IDs in ascending order.

The canonical full evaluation slice is all 20 tasks at context length 32768.

### LongBench-v2 CodeQA

Source: [`zai-org/LongBench-v2`](https://huggingface.co/datasets/zai-org/LongBench-v2)

- Read the published Hugging Face `train` split, which contains the evaluation
  records.
- Retain the 50 rows in Code Repository Understanding / Code Repo QA.
- Store the upstream ID, question, choices A-D, gold choice, context, difficulty,
  length bucket, domain, and sub-domain.
- Validate that every gold answer is one of `A`, `B`, `C`, or `D`.

The canonical full evaluation slice is all 50 stored rows.

## Manifest Contract

Each `manifest.json` contains:

- `schema_version`
- canonical benchmark name
- creation timestamp in UTC
- source repository names and full immutable revisions
- source split names
- transformation and filter descriptions
- canonical full-evaluation filter
- file names, byte sizes, record counts, and SHA-256 checksums
- context and example counts where both are present

The manifest schema starts at version 1. Benchmark runners reject unsupported
versions, missing files, count mismatches, and checksum mismatches before
contacting vLLM.

## Fetch Workflow

The Makefile exposes:

```bash
make fetch-benchmarks
```

The target invokes the fetcher with transient Hugging Face dependencies through
`uv --with`; these dependencies are not added to the core project install.

The fetcher performs these steps for each benchmark:

1. If a complete local snapshot exists, validate its manifest and checksums.
2. Skip a valid snapshot unless `--force` is supplied.
3. Download the pinned source revision into a temporary staging area.
4. Filter and normalize the paper-specific subset.
5. Validate source schema, joins, expected counts, and benchmark invariants.
6. Write gzip JSONL files and the manifest.
7. Re-read the staged files and verify counts and checksums.
8. Atomically replace the benchmark directory only after validation succeeds.

A failure leaves the previous valid snapshot untouched. Authentication, network,
source-schema, count, join, serialization, and checksum failures abort loudly.
There are no alternate sources, silent revision upgrades, or partial-success
snapshots.

## Benchmark Command Interface

The Makefile exposes three independent evaluation targets:

```bash
make benchmark-oolong MODEL=Qwen/Qwen3-1.7B
make benchmark-oolong-pairs MODEL=Qwen/Qwen3-1.7B
make benchmark-codeqa MODEL=Qwen/Qwen3-1.7B
```

Shared Makefile defaults are:

```make
BENCHMARK_DATA_DIR ?= ./data/benchmarks
LOG_DIR ?= ./logs
FETCH_ARGS ?=
BENCHMARK_ARGS ?=
```

`fetch-benchmarks` is not a prerequisite of any benchmark target. Missing data
causes an actionable startup error that names `make fetch-benchmarks`.

Common runner arguments include:

- required `--model`
- `--data-dir`, defaulting to `./data/benchmarks`
- `--log-dir`, defaulting to `./logs`
- `--num-examples`, omitted by default
- `--seed`, defaulting to `42` in Python
- `--max-depth`
- `--max-iterations`
- `--max-tokens`

OOLONG and OOLONG-Pairs also accept a benchmark context length. Their defaults are
131072 and 32768 respectively. OOLONG retains the optional `--exclude-numeric`
filter for diagnostics. Noncanonical context filters, numeric exclusion, or
example caps make the run a selective check rather than a full evaluation.

`FETCH_ARGS` exposes intentional fetch controls such as a validated refresh:

```bash
make fetch-benchmarks FETCH_ARGS="--force"
```

`BENCHMARK_ARGS` passes intentional overrides without adding a Make variable for
every runner option:

```bash
make benchmark-codeqa \
  MODEL=Qwen/Qwen3-1.7B \
  BENCHMARK_ARGS="--num-examples 5 --seed 7"
```

No `benchmark-all` target is added. Expensive evaluations must be launched
intentionally.

## Full Evaluation and Selective Checks

Full paper-slice evaluation is the default:

| Benchmark | Default eligible examples | Default context |
|---|---:|---:|
| OOLONG | 50 | 131072 |
| OOLONG-Pairs | 20 | 32768 |
| CodeQA | 50 | dataset-defined |

`--num-examples N` is a run-time cap for selective checks. It never changes the
local snapshot. When a cap is present, the runner deterministically shuffles the
eligible IDs with `random.Random(seed)` and selects exactly `N`. The same selected
IDs and order are used for both modes. Invalid caps and caps larger than the
eligible set fail instead of silently changing the requested sample size.

Without a cap, the runner evaluates the complete eligible slice in stable source
order. The seed does not affect membership in a full run.

The JSON summary records:

- `is_full_evaluation`
- the eligible and selected counts
- selected example IDs
- the cap and seed
- all benchmark-specific filters

Only a run matching the canonical benchmark slice with no cap or exclusion is
labeled as a full evaluation.

## Paired Plain and RLM Evaluation

Each selected example produces one `plain` row and one `rlm` row.

### Plain Mode

Plain mode sends one prompt containing the benchmark instruction, question, and
full context through the existing local vLLM-compatible client.

### RLM Mode

RLM mode passes the long context as the `prompt` to `RLM.completion(...)` and the
benchmark instruction plus question as `root_prompt`. The context is therefore
available inside the REPL as `context` instead of being flattened into the root
model request.

Both modes use the same model, selected examples, and configured output limit.
RLM uses the configured recursion and iteration limits and continues to create
trajectory JSONL files with `RLMLogger` in `logs/`.

The shared runner module owns:

- local manifest validation
- vLLM health checks and client construction
- plain and RLM invocation
- deterministic selection
- latency and usage accounting
- incremental CSV writing
- final aggregate calculation
- timestamped artifact naming
- atomic summary writing

Benchmark modules own only local schema loading, prompt construction, selection
eligibility, answer parsing, and scoring.

## Deterministic Scoring

### OOLONG

Retain the current OOLONG scorer behavior for literal, comparison, numeric, and
date answers. Numeric distance uses the existing `0.75 ** absolute_error`
function. Unparseable answers score zero.

### OOLONG-Pairs

Parse pairs from the final prediction, canonicalize each pair into ascending
integer order, and deduplicate the result. Compare the predicted and gold sets:

```text
precision = true_positive_pairs / predicted_pairs
recall = true_positive_pairs / gold_pairs
F1 = 2 * precision * recall / (precision + recall)
```

The primary row score is F1. Precision and recall are also stored. Empty-set edge
cases are defined explicitly: two empty sets score 1; only one empty set scores 0.
Malformed fragments are ignored, and a prediction with no valid pairs receives
the corresponding empty-set score.

### CodeQA

Extract one final answer from `A`, `B`, `C`, or `D` using a strict deterministic
parser that prioritizes an explicit final-answer pattern and otherwise accepts a
standalone final choice. Exact agreement with the gold choice scores 1; a wrong or
unparseable choice scores 0. Parse status is retained in the CSV.

## Result Artifacts

At startup, the runner creates one UTC run timestamp and one sanitized stem:

```text
<benchmark>_<model-slug>_<YYYYMMDDTHHMMSSZ>
```

For example:

```text
logs/oolong_qwen-qwen3-1.7b_20260714T143012Z.csv
logs/oolong_qwen-qwen3-1.7b_20260714T143012Z.json
```

The CSV and JSON always have exactly the same stem. Existing files are never
overwritten. The previous free-form OOLONG `--output` argument is removed so the
runner owns the paired naming contract.

### CSV

The CSV contains one row per `(example, mode)`. Shared columns include:

- run ID and benchmark
- example ID and dataset name
- mode and model
- question and gold answer
- prediction and primary score
- score precision and recall where applicable
- parse status where applicable
- latency
- prompt, completion, and total token counts
- total model-call count
- RLM trajectory log path
- error text

Rows are appended and flushed after each mode finishes so completed work survives
a later per-example failure or interruption.

### JSON Summary

The matching JSON is an aggregate run summary, not a duplicate of the CSV. It
contains:

- schema version and run ID
- benchmark and model
- UTC start and finish timestamps
- complete effective CLI configuration
- local manifest identity and source revisions
- CSV path
- full-evaluation or selective-check status
- eligible, selected, succeeded, and failed counts
- selected example IDs, cap, and seed
- separate `plain` and `rlm` aggregate scores
- `rlm_minus_plain`
- aggregate latency and usage totals by mode
- scoring method name

Per-mode aggregate scores use the full selected-example denominator. An inference
failure is recorded in the CSV and contributes zero, preventing failed calls from
artificially improving the final score.

The summary is written atomically only after the run completes. An interrupted
run retains its incremental CSV but does not create a misleading final JSON.

## Error Handling

Startup failures abort before result files are created:

- missing or invalid local snapshot
- unsupported manifest schema
- checksum or count mismatch
- invalid selection cap or filter
- unavailable local vLLM server
- missing required transient runner dependency

Once evaluation begins, an inference failure is isolated to its `(example, mode)`
row. The runner records the error, assigns score zero, and continues. Data,
manifest, scorer-configuration, and result-writer errors remain fatal because
continuing would make the aggregate unreliable.

No benchmark runner downloads data, upgrades revisions, swaps sources, or silently
changes filters.

## Testing Strategy

All automated tests are deterministic and offline.

### Fetch Tests

Use compact local fixtures to verify:

- authoritative subset filtering
- context deduplication and example references
- OOLONG-Pairs context joins
- CodeQA category filtering
- gzip JSONL serialization
- manifest fields, counts, byte sizes, and checksums
- staging validation and preservation of an existing valid snapshot after failure

### Scoring Tests

Cover:

- OOLONG literal, comparison, numeric-distance, and date scores
- OOLONG-Pairs parsing, canonicalization, duplicates, malformed fragments, empty
  sets, precision, recall, and F1
- CodeQA explicit, standalone, wrong, ambiguous, and unparseable choices

### Selection Tests

Verify:

- the default selects the complete canonical slice
- seed 42 is defined in Python
- a cap selects exactly `N` stable IDs
- equal seeds select equal IDs
- plain and RLM receive the same selected IDs and order
- invalid and excessive caps fail
- caps and noncanonical filters set `is_full_evaluation` to false

### Artifact Tests

Verify:

- model slug sanitization
- one timestamp shared by the CSV and JSON
- matching stems and no overwrite
- incremental CSV rows
- failure-as-zero aggregation
- correct full-evaluation metadata
- atomic JSON summary creation

### Runner and Makefile Tests

Mock vLLM client and RLM calls. No automated test requires network access, Docker,
a live vLLM server, a GPU, or a live dataset. Makefile tests confirm that fetch
and benchmark targets exist and that benchmark targets do not depend on the fetch
target.

Existing OOLONG tests are adapted to the shared runner and local loader rather
than duplicated.

## Expected Implementation Scope

Add:

- `scripts/fetch_benchmarks.py`
- `examples/benchmark_common.py`
- `examples/benchmark_oolong_pairs.py`
- `examples/benchmark_codeqa.py`
- deterministic fetch, scorer, selection, artifact, and runner tests

Change:

- `examples/benchmark_oolong.py`
- `Makefile`
- `.gitignore`
- `tests/test_vllm_oolong_examples.py` as needed

Delete dead runtime dataset-loading code from the OOLONG runner instead of
keeping a compatibility path.

## Acceptance Criteria

1. `make fetch-benchmarks` creates valid, revision-pinned local snapshots for
   OOLONG, OOLONG-Pairs, and CodeQA under `data/benchmarks/`.
2. Re-running fetch skips valid snapshots; `--force` refreshes through validated
   staging.
3. All benchmark runners work without external dataset access after fetching.
4. No benchmark command invokes or depends on the fetch command.
5. Full canonical benchmark slices are the default; selective caps are explicit,
   deterministic, and labeled partial.
6. Every selected example is evaluated in both plain and RLM modes.
7. OOLONG, OOLONG-Pairs, and CodeQA use deterministic benchmark-appropriate
   scorers.
8. Each completed run creates matching timestamped CSV and JSON files in `logs/`.
9. The JSON records separate final plain and RLM scores, the difference, complete
   selection metadata, and source provenance.
10. Per-example inference failures remain in the aggregate denominator as zero.
11. Tests pass without network, Docker, GPUs, live vLLM, or live datasets.
12. BrowseComp-Plus and S-NIAH remain outside the implementation.
