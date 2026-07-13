.PHONY: help install install-dev install-modal run-all \
        quickstart docker-repl lm-repl modal-repl \
        vllm-pull vllm-up vllm-health simple-infer benchmark-oolong \
        lint format test check

MODEL ?= Qwen/Qwen3-0.6B

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

vllm-up:
	docker run --runtime nvidia --gpus all \
		-p 8000:8000 \
		-v ~/.cache/huggingface:/root/.cache/huggingface \
		-e HF_TOKEN \
		--ipc=host \
		vllm/vllm-openai:latest \
		--model $(MODEL)

vllm-health:
	curl -fsS http://localhost:8000/v1/models

simple-infer:
	uv run python -m examples.simple_inference --model $(MODEL) --prompt "$(PROMPT)"

benchmark-oolong:
	uv run --with datasets --with python-dateutil python -m examples.benchmark_oolong --model $(MODEL)

lint: install-dev
	uv run ruff check .

format: install-dev
	uv run ruff format .

test: install-dev
	uv run pytest

check: lint format test
