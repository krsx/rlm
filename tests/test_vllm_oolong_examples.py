from __future__ import annotations

from pathlib import Path

from rlm.core.types import ModelUsageSummary, UsageSummary


def test_plain_prompt_contains_instruction_question_and_context():
    from examples.benchmark_oolong import build_plain_prompt

    prompt = build_plain_prompt(
        question="Which category wins?", context="User 1: sample"
    )

    assert "Answer the following aggregate question." in prompt
    assert "Question: Which category wins?" in prompt
    assert "Context:\nUser 1: sample" in prompt


def test_rlm_inputs_keep_context_separate_from_root_prompt():
    from examples.benchmark_oolong import build_rlm_inputs

    prompt, root_prompt = build_rlm_inputs(
        question="Which category wins?", context="long context"
    )

    assert prompt == "long context"
    assert "Question: Which category wins?" in root_prompt
    assert "long context" not in root_prompt


def test_synth_score_matches_exact_literal_answer():
    from examples.benchmark_oolong import synth_score

    datapoint = {"answer": "['Paris']", "answer_type": "ANSWER_TYPE.ENTITY"}

    assert synth_score(datapoint, "Answer: Paris") == 1.0


def test_synth_score_decays_numeric_distance():
    from examples.benchmark_oolong import synth_score

    datapoint = {"answer": "[10]", "answer_type": "ANSWER_TYPE.NUMERIC"}

    assert synth_score(datapoint, "Answer: 12") == 0.75**2


def test_csv_row_uses_declared_schema_and_usage_totals():
    from examples.benchmark_oolong import CSV_COLUMNS, make_csv_row

    row = make_csv_row(
        example_id=7,
        dataset_name="trec_coarse",
        mode="plain",
        model="Qwen/Qwen3-0.6B",
        question="Question?",
        gold_answer="['A']",
        prediction="A",
        score=1.0,
        latency_sec=0.25,
        usage_summary=UsageSummary(
            {
                "Qwen/Qwen3-0.6B": ModelUsageSummary(
                    total_calls=2,
                    total_input_tokens=11,
                    total_output_tokens=5,
                )
            }
        ),
        log_file=None,
        error="",
    )

    assert list(row) == CSV_COLUMNS
    assert row["prompt_tokens"] == 11
    assert row["completion_tokens"] == 5
    assert row["total_calls"] == 2
    assert row["log_file"] == ""


def test_filter_examples_respects_context_dataset_numeric_limit_and_seed():
    from examples.benchmark_oolong import filter_examples

    examples = [
        {
            "id": "a",
            "dataset": "trec_coarse",
            "context_len": 100,
            "answer_type": "ANSWER_TYPE.ENTITY",
        },
        {
            "id": "b",
            "dataset": "trec_coarse",
            "context_len": 100,
            "answer_type": "ANSWER_TYPE.NUMERIC",
        },
        {
            "id": "c",
            "dataset": "other",
            "context_len": 100,
            "answer_type": "ANSWER_TYPE.ENTITY",
        },
        {
            "id": "d",
            "dataset": "trec_coarse",
            "context_len": 500,
            "answer_type": "ANSWER_TYPE.ENTITY",
        },
    ]

    filtered = filter_examples(
        examples,
        dataset_name="trec_coarse",
        min_ctx=50,
        max_ctx=200,
        num_examples=1,
        seed=123,
        exclude_numeric=True,
    )

    assert [example["id"] for example in filtered] == ["a"]


def test_makefile_exposes_vllm_targets():
    makefile = Path("Makefile").read_text()

    assert "vllm-pull:" in makefile
    assert "vllm-up:" in makefile
    assert "vllm-health:" in makefile
    assert "simple-infer:" in makefile
    assert "benchmark-oolong:" in makefile
    assert "vllm/vllm-openai:latest" in makefile
    assert "--gpus all" in makefile
    assert "--ipc=host" in makefile
