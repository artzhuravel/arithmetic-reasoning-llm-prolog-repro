from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from src.prolog.execute import execute_solve, normalize_prolog_answer_for_eval

_PROLOG_FENCE_RE = re.compile(
    r"```[ \t]*prolog[ \t]*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_GENERIC_FENCE_RE = re.compile(
    r"```[ \t]*[a-zA-Z0-9_-]*[ \t]*\n(.*?)```",
    re.DOTALL,
)
_KNOWN_EXECUTION_OUTCOMES: tuple[str, ...] = (
    "ok",
    "no_prolog_block",
    "no_solution",
    "syntax_error",
    "execution_error",
    "timeout",
    "dependency_error",
    "unknown_error",
)


def extract_first_prolog_block(model_output: str) -> str | None:
    """
    Extract the first fenced Prolog block from model output.

    Preference order:
    1) ```prolog ... ```
    2) first generic fenced block
    """
    text = str(model_output)
    match = _PROLOG_FENCE_RE.search(text)
    if match is None:
        match = _GENERIC_FENCE_RE.search(text)
    if match is None:
        return None
    extracted = match.group(1).strip()
    return extracted or None


def _load_ground_truth_map(dataset_dir: Path) -> dict[str, str]:
    cur = dataset_dir.resolve()
    gt_path: Path | None = None
    for candidate_dir in (cur, *cur.parents):
        candidate = candidate_dir / "ground_truth_by_prompt.json"
        if candidate.exists():
            gt_path = candidate
            break
    if gt_path is None:
        return {}

    payload = json.loads(gt_path.read_text(encoding="utf-8"))
    all_map = payload.get("all")
    if not isinstance(all_map, dict):
        return {}
    return {str(k).strip(): normalize_prolog_answer_for_eval(v) for k, v in all_map.items()}


def _resolve_expected_answer(
    row: dict[str, Any],
    *,
    gt_cache: dict[str, dict[str, str]],
) -> str:
    expected = normalize_prolog_answer_for_eval(row.get("expected_answer"))
    if expected:
        return expected

    dataset_dir_value = row.get("dataset_dir")
    prompt_key = str(row.get("prompt_key", "")).strip()
    if not dataset_dir_value or not prompt_key:
        return ""

    dataset_dir = str(dataset_dir_value)
    if dataset_dir not in gt_cache:
        gt_cache[dataset_dir] = _load_ground_truth_map(Path(dataset_dir))
    return gt_cache[dataset_dir].get(prompt_key, "")


def _resolve_execution_outcome(exec_ok: bool, error_type: str | None) -> str:
    if exec_ok:
        return "ok"
    if error_type is None or not str(error_type).strip():
        return "unknown_error"
    return str(error_type).strip()


def _build_execution_outcome_summary(
    outcome_counts: Counter[str], *, total_rows: int
) -> tuple[dict[str, int], dict[str, float]]:
    ordered_counts: dict[str, int] = {
        key: int(outcome_counts.get(key, 0)) for key in _KNOWN_EXECUTION_OUTCOMES
    }
    for key in sorted(outcome_counts):
        if key not in ordered_counts:
            ordered_counts[key] = int(outcome_counts[key])

    ordered_rates = {
        key: (value / total_rows if total_rows else 0.0)
        for key, value in ordered_counts.items()
    }
    return ordered_counts, ordered_rates


def process_trace(
    *,
    trace_path: Path,
    output_trace_path: Path,
    output_summary_path: Path,
) -> dict[str, Any]:
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    output_trace_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)

    gt_cache: dict[str, dict[str, str]] = {}
    checked_rows = 0
    code_block_found = 0
    exec_ok_count = 0
    answer_correct_count = 0
    outcome_counts: Counter[str] = Counter()

    with trace_path.open("r", encoding="utf-8") as in_f, output_trace_path.open(
        "w", encoding="utf-8"
    ) as out_f:
        for line in in_f:
            line_s = line.strip()
            if not line_s:
                continue

            row = json.loads(line_s)
            expected = _resolve_expected_answer(row, gt_cache=gt_cache)
            extracted = extract_first_prolog_block(str(row.get("model_output", "")))

            checked_rows += 1
            adjusted_row = dict(row)
            adjusted_row["expected_answer"] = expected
            adjusted_row["extracted_prolog_block"] = extracted

            if extracted is None:
                adjusted_row["exec_ok"] = False
                adjusted_row["exec_error_type"] = "no_prolog_block"
                adjusted_row["exec_error"] = (
                    "No fenced Prolog block found in model_output."
                )
                adjusted_row["exec_normalized_answer"] = None
                adjusted_row["is_correct"] = False
                outcome_counts["no_prolog_block"] += 1
            else:
                code_block_found += 1
                result = execute_solve(extracted)
                adjusted_row["exec_ok"] = result.ok
                adjusted_row["exec_error_type"] = result.error_type
                adjusted_row["exec_error"] = result.error
                adjusted_row["exec_normalized_answer"] = result.normalized_answer

                is_correct = bool(
                    result.ok and expected and result.normalized_answer == expected
                )
                adjusted_row["is_correct"] = is_correct

                if result.ok:
                    exec_ok_count += 1
                if is_correct:
                    answer_correct_count += 1

                outcome_counts[
                    _resolve_execution_outcome(result.ok, result.error_type)
                ] += 1

            out_f.write(json.dumps(adjusted_row) + "\n")

    code_block_found_rate = (
        code_block_found / checked_rows if checked_rows else 0.0
    )
    exec_ok_rate = exec_ok_count / checked_rows if checked_rows else 0.0
    answer_accuracy = answer_correct_count / checked_rows if checked_rows else 0.0
    answer_accuracy_on_exec_ok = (
        answer_correct_count / exec_ok_count if exec_ok_count else 0.0
    )
    outcome_count_summary, outcome_rate_summary = _build_execution_outcome_summary(
        outcome_counts, total_rows=checked_rows
    )

    summary: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "checked_rows": checked_rows,
            "code_block_found": code_block_found,
            "exec_ok": exec_ok_count,
            "answer_correct": answer_correct_count,
        },
        "rates": {
            "code_block_found_rate": code_block_found_rate,
            "exec_ok_rate": exec_ok_rate,
            "answer_accuracy": answer_accuracy,
            "answer_accuracy_on_exec_ok": answer_accuracy_on_exec_ok,
        },
        "execution_outcomes": {
            "counts": outcome_count_summary,
            "rates": outcome_rate_summary,
        },
        "artifacts": {
            "output_trace_jsonl": str(output_trace_path),
            "summary_json": str(output_summary_path),
        },
    }
    output_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process a vanilla eval trace by extracting first fenced Prolog "
            "blocks and re-running execution scoring."
        )
    )
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--output-trace-path", type=Path, required=True)
    parser.add_argument("--output-summary-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = process_trace(
        trace_path=args.trace_path,
        output_trace_path=args.output_trace_path,
        output_summary_path=args.output_summary_path,
    )
    print(
        (
            "[Postprocess] checked={checked} code_block_found={code_found} exec_ok={exec_ok} "
            "answer_correct={correct} summary={summary_path}"
        ).format(
            checked=summary["counts"]["checked_rows"],
            code_found=summary["counts"]["code_block_found"],
            exec_ok=summary["counts"]["exec_ok"],
            correct=summary["counts"]["answer_correct"],
            summary_path=args.output_summary_path,
        )
    )


if __name__ == "__main__":
    main()
