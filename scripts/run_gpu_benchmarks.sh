#!/usr/bin/env bash
# Run the full local-paper benchmark suite (oolong, oolong-pairs, codeqa) on
# two GPUs in parallel, against vLLM servers that are already running (see
# scripts/vllm_up_gpus.sh):
#   GPU0 -> Qwen3-0.6B on port 8000
#   GPU1 -> Qwen3-1.7B on port 8001
# Each GPU runs all 3 benchmarks for ITERATIONS passes (default 3): 9 runs
# per GPU, 18 total. Results land in LOG_DIR with unique, timestamped
# filenames (see examples/benchmark_common.py:artifact_paths), so nothing is
# overwritten across iterations. This script does not start or stop any
# containers, so a benchmark failure never takes down the vLLM servers.
#
# Usage:
#   scripts/vllm_up_gpus.sh      # once, leaves servers running
#   scripts/run_gpu_benchmarks.sh
#   scripts/vllm_down_gpus.sh    # when you're done
#
# Override any default by exporting the variable, e.g.:
#   ITERATIONS=5 scripts/run_gpu_benchmarks.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU0_MODEL="${GPU0_MODEL:-Qwen/Qwen3-0.6B}"
GPU1_MODEL="${GPU1_MODEL:-Qwen/Qwen3-1.7B}"
VLLM_PORT_0="${VLLM_PORT_0:-8000}"
VLLM_PORT_1="${VLLM_PORT_1:-8001}"
ITERATIONS="${ITERATIONS:-3}"
BENCHMARKS=(oolong oolong-pairs codeqa)
DATA_DIR="${BENCHMARK_DATA_DIR:-./data/benchmarks}"
LOG_DIR="${LOG_DIR:-./logs}"

mkdir -p "$LOG_DIR"

check_healthy() {
  local base_url="$1" label="$2"
  if ! curl -fsS "${base_url}/models" >/dev/null 2>&1; then
    echo "vLLM server for $label is not reachable at $base_url" >&2
    echo "Start it first with scripts/vllm_up_gpus.sh" >&2
    exit 1
  fi
}

run_benchmark() {
  local benchmark="$1" model="$2" base_url="$3"
  local target
  case "$benchmark" in
    oolong) target=benchmark-oolong ;;
    oolong-pairs) target=benchmark-oolong-pairs ;;
    codeqa) target=benchmark-codeqa ;;
    *) echo "Unknown benchmark: $benchmark" >&2; exit 1 ;;
  esac
  make "$target" MODEL="$model" VLLM_BASE_URL="$base_url" \
    BENCHMARK_DATA_DIR="$DATA_DIR" LOG_DIR="$LOG_DIR"
}

run_suite() {
  local model="$1" base_url="$2" tag="$3" prefix
  for i in $(seq 1 "$ITERATIONS"); do
    prefix="[$tag run $i/$ITERATIONS]"
    for benchmark in "${BENCHMARKS[@]}"; do
      echo "$prefix $benchmark ($model)"
      run_benchmark "$benchmark" "$model" "$base_url" 2>&1 | sed -u "s/^/$prefix /"
    done
  done
}

check_healthy "http://localhost:${VLLM_PORT_0}/v1" "GPU0 ($GPU0_MODEL)"
check_healthy "http://localhost:${VLLM_PORT_1}/v1" "GPU1 ($GPU1_MODEL)"

run_suite "$GPU0_MODEL" "http://localhost:${VLLM_PORT_0}/v1" "gpu0" \
  > >(tee "${LOG_DIR}/gpu0_console.log") 2>&1 &
PID0=$!

run_suite "$GPU1_MODEL" "http://localhost:${VLLM_PORT_1}/v1" "gpu1" \
  > >(tee "${LOG_DIR}/gpu1_console.log") 2>&1 &
PID1=$!

status=0
wait "$PID0" || status=1
wait "$PID1" || status=1

if [ "$status" -ne 0 ]; then
  echo "One or more benchmark suites failed. Check ${LOG_DIR}/gpu{0,1}_console.log" >&2
  exit 1
fi

runs_per_gpu=$((${#BENCHMARKS[@]} * ITERATIONS))
total_runs=$((runs_per_gpu * 2))
echo "All $total_runs benchmark runs complete ($runs_per_gpu per GPU). Results in $LOG_DIR"
