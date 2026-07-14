#!/usr/bin/env bash
# Stop the vLLM servers started by scripts/vllm_up_gpus.sh.
#
# Usage:
#   scripts/vllm_down_gpus.sh

set -euo pipefail

CONTAINER_0="rlm-vllm-gpu0"
CONTAINER_1="rlm-vllm-gpu1"

echo "Stopping vLLM containers..."
docker stop "$CONTAINER_0" "$CONTAINER_1" >/dev/null 2>&1 || true
echo "Done."
