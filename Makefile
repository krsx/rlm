.PHONY: help install install-dev install-modal run-all \
        quickstart docker-repl lm-repl modal-repl \
        vllm-pull vllm-up vllm-health simple-infer \
        fetch-benchmarks benchmark-oolong benchmark-oolong-pairs benchmark-codeqa \
        lint format test check

MODEL ?= Qwen/Qwen3-1.7B
GPU ?= 0
VLLM_PORT_0 ?= 8000
VLLM_PORT_1 ?= 8001
ifneq ($(GPU),0)
ifneq ($(GPU),1)
$(error GPU must be 0 or 1)
endif
endif
VLLM_PORT ?= $(VLLM_PORT_$(GPU))
VLLM_BASE_URL ?= http://localhost:$(VLLM_PORT)/v1
# Default is vLLM's own "auto" (bf16 on modern GPUs). Override for GPUs
# without native bf16 support (e.g. Turing/2080Ti, compute capability < 8.0):
#   make vllm-up VLLM_DTYPE=half
VLLM_DTYPE ?= auto
BENCHMARK_DATA_DIR ?= ./data/benchmarks
LOG_DIR ?= ./logs
FETCH_ARGS ?=
BENCHMARK_ARGS ?=

define DEFAULT_PROMPT
You are given a collection of customer-support records.

For every record:
1. Classify the issue into exactly one category: BILLING, TECHNICAL, ACCOUNT, or DELIVERY.
2. Count how many records belong to each category.
3. Identify the category with the highest average resolution time.
4. Return the result as valid JSON only.

Records:
1. Customer was charged twice for the same subscription. Resolution time: 18 minutes.
2. The mobile application crashes immediately after login. Resolution time: 42 minutes.
3. Customer forgot their password and cannot access the account. Resolution time: 12 minutes.
4. Package arrived three days later than promised. Resolution time: 35 minutes.
5. Refund has not appeared on the customers credit card. Resolution time: 28 minutes.
6. Website displays a blank screen during checkout. Resolution time: 50 minutes.
7. Customer wants to change the email linked to the account. Resolution time: 16 minutes.
8. Tracking states delivered, but the package was not received. Resolution time: 55 minutes.
9. Subscription price increased unexpectedly. Resolution time: 24 minutes.
10. Desktop software fails to install after an update. Resolution time: 38 minutes.

Required output format:
{"category_counts":{"BILLING":0,"TECHNICAL":0,"ACCOUNT":0,"DELIVERY":0},"highest_average_resolution_time":{"category":"","average_minutes":0}}
endef

PROMPT ?= $(DEFAULT_PROMPT)
export PROMPT

help:
	@echo "RLM Examples Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make install        - Install base dependencies with uv"
	@echo "  make install-dev    - Install dev dependencies with uv"
	@echo "  make install-modal  - Install modal dependencies with uv"
	@echo "  make run-all        - Run all examples (requires all deps and API keys)"
	@echo ""
	@echo "Examples:"
	@echo "  GPU=0 or GPU=1 selects the API for all local vLLM commands"
	@echo "  make quickstart     - Run quickstart.py (needs OPENAI_API_KEY)"
	@echo "  make docker-repl    - Run docker_repl_example.py (needs Docker)"
	@echo "  make lm-repl        - Run lm_in_repl.py (needs PORTKEY_API_KEY)"
	@echo "  make modal-repl     - Run modal_repl_example.py (needs Modal)"
	@echo "  make vllm-up GPU=0 MODEL=Qwen/Qwen3-0.6B - Start vLLM on GPU 0 (port 8000)"
	@echo "  make vllm-up GPU=1 MODEL=Qwen/Qwen3-0.6B - Start vLLM on GPU 1 (port 8001)"
	@echo "  make vllm-up VLLM_DTYPE=half - Force fp16 (needed on Turing GPUs like RTX 2080 Ti, no native bf16)"
	@echo "  make simple-infer GPU=0 MODEL=Qwen/Qwen3-0.6B PROMPT='...' - Run local vLLM RLM inference"
	@echo "  make fetch-benchmarks - Download and validate local paper benchmark snapshots"
	@echo "  make benchmark-oolong MODEL=Qwen/Qwen3-0.6B - Run local vLLM OOLONG benchmark"
	@echo "  make benchmark-oolong-pairs MODEL=Qwen/Qwen3-0.6B - Run local vLLM OOLONG-Pairs benchmark"
	@echo "  make benchmark-codeqa MODEL=Qwen/Qwen3-0.6B - Run local vLLM CodeQA benchmark"
	@echo ""
	@echo "Development:"
	@echo "  make lint           - Run ruff linter"
	@echo "  make format         - Run ruff formatter"
	@echo "  make test           - Run tests"
	@echo "  make check          - Run lint + format + tests"

install:
	uv sync

install-dev:
	uv sync --group dev --group test

install-modal:
	uv pip install -e ".[modal]"

run-all: quickstart docker-repl lm-repl modal-repl

quickstart: install
	uv run python -m examples.quickstart

docker-repl: install
	uv run python -m examples.docker_repl_example

lm-repl: install
	uv run python -m examples.lm_in_repl

modal-repl: install-modal
	uv run python -m examples.modal_repl_example

vllm-pull:
	docker pull vllm/vllm-openai:latest

# GPU 0 uses host port 8000; GPU 1 uses host port 8001. Docker exposes the
# selected physical GPU as device 0 inside its container.
vllm-up:
	docker run --runtime nvidia --gpus '"device=$(GPU)"' \
		-e CUDA_VISIBLE_DEVICES=0 \
		-e CUDA_DEVICE_ORDER=PCI_BUS_ID \
		-p $(VLLM_PORT):8000 \
		-v ~/.cache/huggingface:/root/.cache/huggingface \
		-e HF_TOKEN \
		--ipc=host \
		vllm/vllm-openai:v0.8.5 \
		--model $(MODEL) \
		--dtype $(VLLM_DTYPE)

vllm-health:
	curl -fsS $(VLLM_BASE_URL)/models

simple-infer:
	uv run python -m examples.simple_inference --model $(MODEL) \
		--base-url $(VLLM_BASE_URL) --prompt "$$PROMPT"

fetch-benchmarks:
	uv run --with datasets --with huggingface_hub --with ijson \
		python -m scripts.fetch_benchmarks \
		--data-dir $(BENCHMARK_DATA_DIR) $(FETCH_ARGS)

benchmark-oolong:
	uv run --with python-dateutil python -m examples.benchmark_oolong \
		--model $(MODEL) --base-url $(VLLM_BASE_URL) \
		--data-dir $(BENCHMARK_DATA_DIR) --log-dir $(LOG_DIR) \
		$(BENCHMARK_ARGS)

benchmark-oolong-pairs:
	uv run python -m examples.benchmark_oolong_pairs \
		--model $(MODEL) --base-url $(VLLM_BASE_URL) \
		--data-dir $(BENCHMARK_DATA_DIR) --log-dir $(LOG_DIR) \
		$(BENCHMARK_ARGS)

benchmark-codeqa:
	uv run python -m examples.benchmark_codeqa \
		--model $(MODEL) --base-url $(VLLM_BASE_URL) \
		--data-dir $(BENCHMARK_DATA_DIR) --log-dir $(LOG_DIR) \
		$(BENCHMARK_ARGS)

lint: install-dev
	uv run ruff check .

format: install-dev
	uv run ruff format .

test: install-dev
	uv run pytest

check: lint format test
