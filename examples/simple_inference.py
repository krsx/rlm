# Run a local vLLM server first:
#   make vllm-up MODEL=Qwen/Qwen3-0.6B
#
# Run one recursive inference call:
#   uv run python -m examples.simple_inference --model Qwen/Qwen3-0.6B --prompt "Find the answer."

from __future__ import annotations

import argparse
import json

import requests

from rlm import RLM
from rlm.logger import RLMLogger

LOCAL_VLLM_BASE_URL = "http://localhost:8000/v1"


def require_vllm_server() -> None:
    response = requests.get(f"{LOCAL_VLLM_BASE_URL}/models", timeout=5)
    response.raise_for_status()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one RLM completion against local vLLM."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--log-dir", default="./logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_vllm_server()

    logger = RLMLogger(log_dir=args.log_dir)
    sampling_args = (
        {"max_tokens": args.max_tokens} if args.max_tokens is not None else None
    )
    rlm = RLM(
        backend="vllm",
        backend_kwargs={
            "model_name": args.model,
            "base_url": LOCAL_VLLM_BASE_URL,
            "api_key": "dummy",
        },
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        sampling_args=sampling_args,
        sub_sampling_args=sampling_args,
        verbose=True,
        logger=logger,
    )
    result = rlm.completion(args.prompt)

    print("\nFinal response:")
    print(result.response)
    print("\nUsage summary:")
    print(json.dumps(result.usage_summary.to_dict(), indent=2))
    print("\nLogger file path:")
    print(logger.log_file_path or "")


if __name__ == "__main__":
    main()
