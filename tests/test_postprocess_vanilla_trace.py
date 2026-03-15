from __future__ import annotations

import json
from pathlib import Path

from src.training.postprocess_vanilla_trace import (
    extract_first_prolog_block,
    process_trace,
)


def test_extract_first_prolog_block_prefers_first_fenced_prolog_block() -> None:
    model_output = (
        "prefix\n"
        "```prolog\n"
        "solve(Result) :- Result = 5.\n"
        "```\n"
        "middle\n"
        "```prolog\n"
        "solve(Result) :- Result = 99.\n"
        "```\n"
    )

    assert extract_first_prolog_block(model_output) == "solve(Result) :- Result = 5."


def test_process_trace_rewrites_execution_fields_from_extracted_block(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "gsm8k_prolog"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "ground_truth_by_prompt.json").write_text(
        json.dumps(
            {
                "all": {
                    "q1": "5",
                    "q2": "8",
                    "q3": "1",
                }
            }
        ),
        encoding="utf-8",
    )

    trace_path = tmp_path / "trace.jsonl"
    rows = [
        {
            "index": 0,
            "suite": "gsm8k_prolog_test",
            "dataset_dir": str(dataset_dir),
            "split": "test",
            "prompt_key": "q1",
            "model_input": "prompt1",
            "model_output": (
                "junk\n"
                "```prolog\n"
                "solve(Result) :- Result = 5.\n"
                "```\n"
                "```prolog\n"
                "solve(Result) :- Result = 99.\n"
                "```"
            ),
            "expected_answer": "5.0",
            "exec_ok": False,
            "exec_error_type": "syntax_error",
            "exec_error": "old",
            "exec_normalized_answer": None,
            "is_correct": False,
        },
        {
            "index": 1,
            "suite": "gsm8k_prolog_test",
            "dataset_dir": str(dataset_dir),
            "split": "test",
            "prompt_key": "q2",
            "model_input": "prompt2",
            "model_output": "```prolog\nsolve(Result) :- Result = 7.\n```",
            "expected_answer": "8.0",
            "exec_ok": False,
            "exec_error_type": "syntax_error",
            "exec_error": "old",
            "exec_normalized_answer": None,
            "is_correct": False,
        },
        {
            "index": 2,
            "suite": "gsm8k_prolog_test",
            "dataset_dir": str(dataset_dir),
            "split": "test",
            "prompt_key": "q3",
            "model_input": "prompt3",
            "model_output": "no fenced code here",
            "expected_answer": "1.0",
            "exec_ok": False,
            "exec_error_type": "syntax_error",
            "exec_error": "old",
            "exec_normalized_answer": None,
            "is_correct": False,
        },
    ]
    trace_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    output_trace_path = tmp_path / "trace_adjusted_first_prolog_block.jsonl"
    output_summary_path = tmp_path / "vanilla_model_adjusted_summary.json"

    summary = process_trace(
        trace_path=trace_path,
        output_trace_path=output_trace_path,
        output_summary_path=output_summary_path,
    )

    adjusted_rows = [
        json.loads(line)
        for line in output_trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(adjusted_rows) == 3
    assert adjusted_rows[0]["exec_ok"] is True
    assert adjusted_rows[0]["exec_normalized_answer"] == "5.0"
    assert adjusted_rows[0]["is_correct"] is True
    assert adjusted_rows[0]["extracted_prolog_block"] == "solve(Result) :- Result = 5."

    assert adjusted_rows[1]["exec_ok"] is True
    assert adjusted_rows[1]["exec_normalized_answer"] == "7.0"
    assert adjusted_rows[1]["is_correct"] is False

    assert adjusted_rows[2]["exec_ok"] is False
    assert adjusted_rows[2]["exec_error_type"] == "no_prolog_block"
    assert adjusted_rows[2]["is_correct"] is False
    assert adjusted_rows[2]["extracted_prolog_block"] is None

    assert summary["counts"]["checked_rows"] == 3
    assert summary["counts"]["code_block_found"] == 2
    assert summary["counts"]["exec_ok"] == 2
    assert summary["counts"]["answer_correct"] == 1
    assert summary["rates"]["code_block_found_rate"] == 2 / 3
    assert summary["rates"]["exec_ok_rate"] == 2 / 3
    assert summary["rates"]["answer_accuracy"] == 1 / 3
    assert summary["rates"]["answer_accuracy_on_exec_ok"] == 0.5
    assert summary["execution_outcomes"]["counts"]["ok"] == 2
    assert summary["execution_outcomes"]["counts"]["no_prolog_block"] == 1
