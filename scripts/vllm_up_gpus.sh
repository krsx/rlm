#!/usr/bin/env bash
# Start two long-running vLLM servers, one per GPU:
#   GPU0 -> Qwen3-0.6B (port 8000)
#   GPU1 -> Qwen3-1.7B (port 8001)
# The containers keep running after this script exits (--rm, but not
# stopped on exit) so benchmarks can be run against them repeatedly without
# paying the model-load cost each time. Use scripts/vllm_down_gpus.sh to
# stop them.
#
# Usage:
#   scripts/vllm_up_gpus.sh
#
# Override defaults by exporting: GPU0_MODEL, GPU1_MODEL, VLLM_PORT_0,
# VLLM_PORT_1, VLLM_DTYPE_0, VLLM_DTYPE_1, VLLM_IMAGE, HEALTH_TIMEOUT.
#
# GPU0 defaults to float16: Turing cards (e.g. RTX 2080 Ti) have no native
# bf16 support, so vLLM's dtype=auto (bf16 on modern GPUs) fails there.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU0_MODEL="${GPU0_MODEL:-Qwen/Qwen3-0.6B}"
GPU1_MODEL="${GPU1_MODEL:-Qwen/Qwen3-1.7B}"
VLLM_PORT_0="${VLLM_PORT_0:-8000}"
VLLM_PORT_1="${VLLM_PORT_1:-8001}"
VLLM_DTYPE_0="${VLLM_DTYPE_0:-float16}"
VLLM_DTYPE_1="${VLLM_DTYPE_1:-auto}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.8.5}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"

CONTAINER_0="rlm-vllm-gpu0"
CONTAINER_1="rlm-vllm-gpu1"

start_vllm() {
  local gpu="$1" model="$2" port="$3" name="$4" dtype="$5"
  docker rm -f "$name" >/dev/null 2>&1 || true
  # No --rm here: if the container crashes on startup we need `docker logs`
  # to still work afterward. vllm_down_gpus.sh removes it on teardown.
  docker run -d --name "$name" \
    --runtime nvidia --gpus "\"device=${gpu}\"" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -p "${port}:8000" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN \
    --ipc=host \
    "$VLLM_IMAGE" \
    --model "$model" \
    --dtype "$dtype" >/dev/null
}

wait_healthy() {
  local base_url="$1" name="$2"
  echo "Waiting for $name at $base_url ..."
  local waited=0
  until curl -fsS "${base_url}/models" >/dev/null 2>&1; do
    local state
    state="$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null || echo missing)"
    if [ "$state" != "running" ]; then
      echo "Container $name is not running (state: $state). Logs:" >&2
      docker logs "$name" --tail 200 || true
      exit 1
    fi
    sleep 5
    waited=$((waited + 5))
    if [ "$waited" -ge "$HEALTH_TIMEOUT" ]; then
      echo "Timed out waiting for $name after ${HEALTH_TIMEOUT}s" >&2
      docker logs "$name" --tail 200 || true
      exit 1
    fi
  done
  echo "$name is healthy."
}

echo "Starting vLLM server on GPU0 ($GPU0_MODEL, port $VLLM_PORT_0, dtype $VLLM_DTYPE_0)..."
start_vllm 0 "$GPU0_MODEL" "$VLLM_PORT_0" "$CONTAINER_0" "$VLLM_DTYPE_0"
echo "Starting vLLM server on GPU1 ($GPU1_MODEL, port $VLLM_PORT_1, dtype $VLLM_DTYPE_1)..."
start_vllm 1 "$GPU1_MODEL" "$VLLM_PORT_1" "$CONTAINER_1" "$VLLM_DTYPE_1"

wait_healthy "http://localhost:${VLLM_PORT_0}/v1" "$CONTAINER_0"
wait_healthy "http://localhost:${VLLM_PORT_1}/v1" "$CONTAINER_1"

echo "Both servers are up:"
echo "  GPU0: $GPU0_MODEL -> http://localhost:${VLLM_PORT_0}/v1"
echo "  GPU1: $GPU1_MODEL -> http://localhost:${VLLM_PORT_1}/v1"
echo "Run scripts/run_gpu_benchmarks.sh next. Stop with scripts/vllm_down_gpus.sh."
