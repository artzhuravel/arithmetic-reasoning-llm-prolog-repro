from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.callbacks import score_predicted_prolog_batch, tqdm


def _strip_prompt_overlap(model_input: str, model_output: str) -> str:
    if not model_input or not model_output:
        return model_output
    max_k = min(len(model_input), len(model_output))
    for k in range(max_k, 0, -1):
        if model_input[-k:] == model_output[:k]:
            return model_output[k:]
    return model_output


def _after_output_header(text: str) -> str:
    marker = "### Output"
    idx = text.rfind(marker)
    if idx < 0:
        return text
    tail = text[idx + len(marker):]
    if tail.startswith("\r\n"):
        return tail[2:]
    if tail.startswith("\n"):
        return tail[1:]
    return tail


def _prepare_predicted_code(record: dict[str, Any], *, clean_mode: str) -> str:
    model_output = str(record.get("model_output", ""))
    model_input = str(record.get("model_input", ""))

    if clean_mode == "raw":
        return model_output
    if clean_mode == "strip_prompt_overlap":
        return _strip_prompt_overlap(model_input, model_output)
    if clean_mode == "after_output_header":
        return _after_output_header(model_output)
    if clean_mode == "auto":
        cleaned = _strip_prompt_overlap(model_input, model_output)
        cleaned = _after_output_header(cleaned)
        return cleaned.lstrip()
    raise ValueError(f"Unsupported clean_mode: {clean_mode}")


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _iter_trace_rows(path: Path, *, max_samples: int | None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break
            line_s = line.strip()
            if not line_s:
                continue
            yield json.loads(line_s)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay an existing eval trace through the Prolog executor without "
            "running generation again."
        )
    )
    parser.add_argument(
        "--trace-jsonl",
        type=Path,
        required=True,
        help="Path to existing eval trace.jsonl.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for replay summary/artifacts. Default: trace_dir/replay_<timestamp>.",
    )
    parser.add_argument(
        "--clean-mode",
        type=str,
        choices=("raw", "strip_prompt_overlap", "after_output_header", "auto"),
        default="auto",
        help=(
            "How to transform model_output before Prolog execution. "
            "'auto' strips prompt overlap and keeps only text after '### Output'."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of workers for Prolog execution scoring.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for replay scoring.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of trace rows to replay.",
    )
    parser.add_argument(
        "--save-trace",
        action="store_true",
        help="Write rescored trace records as trace_replayed.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    trace_path: Path = args.trace_jsonl

    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else trace_path.parent / f"replay_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    total_rows_raw = _count_lines(trace_path)
    total_rows = min(total_rows_raw, args.max_samples) if args.max_samples is not None else total_rows_raw

    replay_trace_path: Path | None = None
    replay_trace_file = None
    if args.save_trace:
        replay_trace_path = out_dir / "trace_replayed.jsonl"
        replay_trace_file = replay_trace_path.open("w", encoding="utf-8")

    checked_rows = 0
    rows_with_ground_truth = 0
    missing_ground_truth = 0
    exec_ok = 0
    answer_correct = 0
    answer_correct_when_exec_ok = 0
    error_type_counts: Counter[str] = Counter()

    batch_records: list[dict[str, Any]] = []
    batch_items: list[tuple[str, str]] = []

    def _flush_batch() -> None:
        nonlocal checked_rows
        nonlocal rows_with_ground_truth
        nonlocal missing_ground_truth
        nonlocal exec_ok
        nonlocal answer_correct
        nonlocal answer_correct_when_exec_ok
        if not batch_items:
            return

        scored = score_predicted_prolog_batch(batch_items, workers=args.workers)
        for row, item, (exec_ok_inc, raw_correct_inc, exec_result) in zip(
            batch_records, batch_items, scored
        ):
            pred_code, expected = item
            checked_rows += 1
            exec_ok += exec_ok_inc
            has_expected = bool(expected)
            if has_expected:
                rows_with_ground_truth += 1
                correct_inc = raw_correct_inc
            else:
                missing_ground_truth += 1
                correct_inc = 0
            answer_correct += correct_inc
            answer_correct_when_exec_ok += correct_inc
            if not exec_result.ok:
                error_type_counts[exec_result.error_type or "unknown"] += 1

            if replay_trace_file is not None:
                replay_row = dict(row)
                replay_row["replay_clean_mode"] = args.clean_mode
                replay_row["prolog_code_executed"] = pred_code
                replay_row["replay_exec_ok"] = exec_result.ok
                replay_row["replay_exec_error_type"] = exec_result.error_type
                replay_row["replay_exec_error"] = exec_result.error
                replay_row["replay_exec_normalized_answer"] = exec_result.normalized_answer
                replay_row["replay_is_correct"] = bool(correct_inc)
                replay_trace_file.write(json.dumps(replay_row) + "\n")

        batch_records.clear()
        batch_items.clear()

    started = perf_counter()
    print(
        (
            "[Replay] rows={rows} clean_mode={clean_mode} workers={workers} "
            "batch_size={batch_size}"
        ).format(
            rows=total_rows,
            clean_mode=args.clean_mode,
            workers=args.workers,
            batch_size=args.batch_size,
        )
    )
    try:
        with tqdm(total=total_rows, desc="Replay execute", unit="sample") as pbar:
            for row in _iter_trace_rows(trace_path, max_samples=args.max_samples):
                pred_code = _prepare_predicted_code(row, clean_mode=args.clean_mode)
                expected = str(row.get("expected_answer", "")).strip()
                batch_records.append(row)
                batch_items.append((pred_code, expected))
                if len(batch_items) >= args.batch_size:
                    prev_checked = checked_rows
                    _flush_batch()
                    pbar.update(checked_rows - prev_checked)

            prev_checked = checked_rows
            _flush_batch()
            pbar.update(checked_rows - prev_checked)
    finally:
        if replay_trace_file is not None:
            replay_trace_file.close()

    elapsed_s = perf_counter() - started
    exec_ok_rate = exec_ok / checked_rows if checked_rows else 0.0
    answer_accuracy = answer_correct / checked_rows if checked_rows else 0.0
    answer_accuracy_on_exec_ok = (
        answer_correct_when_exec_ok / exec_ok if exec_ok else 0.0
    )
    ground_truth_coverage = rows_with_ground_truth / checked_rows if checked_rows else 0.0

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "trace_jsonl": str(trace_path),
            "clean_mode": args.clean_mode,
            "workers": args.workers,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "save_trace": args.save_trace,
        },
        "counts": {
            "checked_rows": checked_rows,
            "rows_with_ground_truth": rows_with_ground_truth,
            "missing_ground_truth": missing_ground_truth,
            "exec_ok": exec_ok,
            "answer_correct": answer_correct,
            "answer_correct_when_exec_ok": answer_correct_when_exec_ok,
        },
        "rates": {
            "ground_truth_coverage": ground_truth_coverage,
            "exec_ok_rate": exec_ok_rate,
            "answer_accuracy": answer_accuracy,
            "answer_accuracy_on_exec_ok": answer_accuracy_on_exec_ok,
        },
        "error_type_counts": dict(error_type_counts),
        "runtime_seconds": elapsed_s,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "trace_replayed_jsonl": str(replay_trace_path) if replay_trace_path is not None else None,
        },
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        (
            "[Replay] checked={checked} exec_ok={exec_ok} correct={correct} "
            "exec_ok_rate={exec_ok_rate:.4f} answer_acc={answer_acc:.4f} "
            "answer_acc_on_exec_ok={answer_acc_exec:.4f}"
        ).format(
            checked=checked_rows,
            exec_ok=exec_ok,
            correct=answer_correct,
            exec_ok_rate=exec_ok_rate,
            answer_acc=answer_accuracy,
            answer_acc_exec=answer_accuracy_on_exec_ok,
        )
    )
    print(f"[Replay] summary={summary_path}")
    if replay_trace_path is not None:
        print(f"[Replay] trace={replay_trace_path}")


if __name__ == "__main__":
    main()
