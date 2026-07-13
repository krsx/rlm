.PHONY: help install install-dev install-modal run-all \
        quickstart docker-repl lm-repl modal-repl \
        vllm-pull vllm-up vllm-health simple-infer benchmark-oolong \
        lint format test check

MODEL ?= Qwen/Qwen3-1.7B

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
	@echo "  make quickstart     - Run quickstart.py (needs OPENAI_API_KEY)"
	@echo "  make docker-repl    - Run docker_repl_example.py (needs Docker)"
	@echo "  make lm-repl        - Run lm_in_repl.py (needs PORTKEY_API_KEY)"
	@echo "  make modal-repl     - Run modal_repl_example.py (needs Modal)"
	@echo "  make vllm-up MODEL=Qwen/Qwen3-0.6B - Start local vLLM OpenAI server"
	@echo "  make simple-infer MODEL=Qwen/Qwen3-0.6B PROMPT='...' - Run local vLLM RLM inference"
	@echo "  make benchmark-oolong MODEL=Qwen/Qwen3-0.6B - Run local vLLM OOLONG benchmark"
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

# Specify the GPU used by changing the device number in --gpus '"device=1"' and CUDA_VISIBLE_DEVICES=0.
# For example, to use GPU 0, change --gpus '"device=0"' and CUDA_VISIBLE_DEVICES=0.
vllm-up:
	docker run --runtime nvidia --gpus '"device=1"' \
		-e CUDA_VISIBLE_DEVICES=0 \
		-e CUDA_DEVICE_ORDER=PCI_BUS_ID \
		-p 8000:8000 \
		-v ~/.cache/huggingface:/root/.cache/huggingface \
		-e HF_TOKEN \
		--ipc=host \
		vllm/vllm-openai:v0.8.5 \
		--model $(MODEL)

vllm-health:
	curl -fsS http://localhost:8000/v1/models

simple-infer:
	uv run python -m examples.simple_inference --model $(MODEL) --prompt "$$PROMPT"

benchmark-oolong:
	uv run --with datasets --with python-dateutil python -m examples.benchmark_oolong --model $(MODEL)

lint: install-dev
	uv run ruff check .

format: install-dev
	uv run ruff format .

test: install-dev
	uv run pytest

check: lint format test
