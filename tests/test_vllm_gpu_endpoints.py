from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from examples import (
    benchmark_codeqa,
    benchmark_common,
    benchmark_oolong,
    benchmark_oolong_pairs,
    simple_inference,
)
from examples.benchmark_common import RunnerConfig
from rlm.core.types import UsageSummary


def make_dry_run(target: str, gpu: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "--no-print-directory", "-n", target, f"GPU={gpu}"],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("gpu", "port"),
    [(0, 8000), (1, 8001)],
)
def test_vllm_up_maps_each_gpu_to_its_own_api_port(gpu: int, port: int) -> None:
    result = make_dry_run("vllm-up", gpu)

    assert result.returncode == 0, result.stderr
    assert f"--gpus '\"device={gpu}\"'" in result.stdout
    assert f"-p {port}:8000" in result.stdout
    assert "-e CUDA_VISIBLE_DEVICES=0" in result.stdout


@pytest.mark.parametrize(
    "target",
    [
        "vllm-health",
        "simple-infer",
        "benchmark-oolong",
        "benchmark-oolong-pairs",
        "benchmark-codeqa",
    ],
)
def test_gpu_one_routes_every_inference_target_to_second_api(target: str) -> None:
    result = make_dry_run(target, 1)

    assert result.returncode == 0, result.stderr
    assert "http://localhost:8001/v1" in result.stdout


def test_makefile_rejects_gpu_other_than_zero_or_one() -> None:
    result = make_dry_run("simple-infer", 2)

    assert result.returncode != 0
    assert "GPU must be 0 or 1" in result.stderr


def test_make_help_explains_gpu_selection_for_all_local_vllm_commands() -> None:
    result = subprocess.run(
        ["make", "--no-print-directory", "help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "GPU=0 or GPU=1 selects the API for all local vLLM commands" in result.stdout


def test_simple_inference_accepts_explicit_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simple_inference.py",
            "--model",
            "model",
            "--prompt",
            "prompt",
            "--base-url",
            "http://localhost:8001/v1",
        ],
    )

    assert simple_inference.parse_args().base_url == "http://localhost:8001/v1"


def test_simple_inference_defaults_to_first_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["simple_inference.py", "--model", "model", "--prompt", "prompt"],
    )

    assert simple_inference.parse_args().base_url == "http://localhost:8000/v1"


@pytest.mark.parametrize(
    "build_parser",
    [
        benchmark_oolong.build_parser,
        benchmark_oolong_pairs.build_parser,
        benchmark_codeqa.build_parser,
    ],
)
def test_benchmark_clis_accept_explicit_base_url(build_parser: Any) -> None:
    args = build_parser().parse_args(["--model", "model", "--base-url", "http://localhost:8001/v1"])

    assert args.base_url == "http://localhost:8001/v1"
    assert build_parser().parse_args(["--model", "model"]).base_url == "http://localhost:8000/v1"


def test_benchmark_calls_use_configured_base_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configured_url = "http://localhost:8001/v1"
    client_kwargs: dict[str, Any] = {}
    rlm_kwargs: dict[str, Any] = {}

    class FakeClient:
        def completion(self, prompt: str) -> str:
            return prompt

        def get_usage_summary(self) -> UsageSummary:
            return UsageSummary({})

    def fake_get_client(backend: str, kwargs: dict[str, Any]) -> FakeClient:
        assert backend == "vllm"
        client_kwargs.update(kwargs)
        return FakeClient()

    class FakeRLM:
        def __init__(self, **kwargs: Any) -> None:
            rlm_kwargs.update(kwargs)

        def completion(self, prompt: str, *, root_prompt: str) -> SimpleNamespace:
            return SimpleNamespace(response=prompt, usage_summary=UsageSummary({}))

    monkeypatch.setattr(benchmark_common, "get_client", fake_get_client)
    monkeypatch.setattr(benchmark_common, "RLM", FakeRLM)
    config = RunnerConfig(
        model="model",
        base_url=configured_url,
        data_dir=tmp_path / "data",
        log_dir=tmp_path / "logs",
        num_examples=None,
        seed=42,
        max_depth=1,
        max_iterations=2,
        max_tokens=None,
        filters={},
    )

    benchmark_common.run_plain("plain", config)
    benchmark_common.run_rlm("context", "root", config)

    assert client_kwargs["base_url"] == configured_url
    assert rlm_kwargs["backend_kwargs"]["base_url"] == configured_url
