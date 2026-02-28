from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, cast

import pytest
from datasets import DatasetDict, load_from_disk

from src.data.prepare_splits import get_default_splits_dir
from src.data.proper_permute import _extract_openai_gsm8k_final_answer
from src.data.hf_loaders import (
    GSM8K_PROLOG_REVISION,
    OPENAI_GSM8K_CONFIG,
    OPENAI_GSM8K_REVISION,
    load_gsm8k_datasets,
)
from src.prolog.execute import execute_solve

TESTS_DIR = Path(__file__).resolve().parent
VALIDATION_RESULTS_DIR = TESTS_DIR / "validation_results"


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"


def _splits_available(splits_dir: Path, split: str) -> bool:
    return (
        (splits_dir / "gsm8k_prolog" / split).exists()
        and (splits_dir / "openai_gsm8k" / split).exists()
    )


def _load_split_pair(
    split: str,
    *,
    source: str,
    splits_dir: Optional[Path] = None,
    raw_pair: Optional[tuple[DatasetDict, DatasetDict]] = None,
):
    if source == "splits":
        if splits_dir is None:
            splits_dir = get_default_splits_dir()
        prolog_path = splits_dir / "gsm8k_prolog" / split
        openai_path = splits_dir / "openai_gsm8k" / split
        if not (prolog_path.exists() and openai_path.exists()):
            raise FileNotFoundError(
                "Split files not found. "
                f"Expected: {prolog_path} and {openai_path}"
            )
        return load_from_disk(str(prolog_path)), load_from_disk(str(openai_path))

    if source == "raw":
        if raw_pair is None:
            raise ValueError("raw_pair required when source='raw'")
        prolog_ds, openai_ds = raw_pair
        prolog_ds = prolog_ds[split]
        openai_ds = openai_ds[split]
        return prolog_ds, openai_ds

    raise ValueError("source must be one of: 'splits', 'raw'")


def _with_optional_tqdm(iterable: Iterable[Any], *, total: int, desc: str) -> Iterable[Any]:
    if os.environ.get("PROLOG_DATASET_CHECK_PROGRESS", "1") != "1":
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc, unit="row")
    except Exception:
        return iterable


def _write_validation_report(report: dict[str, Any]) -> Path:
    VALIDATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    source = str(report.get("source", "unknown"))
    out_path = VALIDATION_RESULTS_DIR / f"dataset_alignment_{source}_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path


def run_dataset_alignment_and_answer_match(
    *,
    source: str,
    workers: int = 10,
    splits_dir: Optional[Path] = None,
    prolog_revision: str = GSM8K_PROLOG_REVISION,
    openai_revision: str = OPENAI_GSM8K_REVISION,
    openai_config: str = OPENAI_GSM8K_CONFIG,
) -> dict[str, Any]:
    """
    Run comprehensive dataset validation and return the JSON report dict.

    Args:
        source: "splits" (prepared) or "raw" (HF cache).
        workers: Number of thread workers for row checks.
        splits_dir: When source="splits", directory containing gsm8k_prolog/ and openai_gsm8k/.
            Falls back to default versioned splits dir if None.
        prolog_revision: When source="raw", HF revision for Thomas-X-Yang/gsm8k-prolog.
        openai_revision: When source="raw", HF revision for openai/gsm8k.
        openai_config: When source="raw", HF config for openai/gsm8k.
    """

    if source not in {"splits", "raw"}:
        raise ValueError(f"Invalid source: {source}")

    raw_pair: Optional[tuple[DatasetDict, DatasetDict]] = None
    if source == "raw":
        raw_pair = load_gsm8k_datasets(
            prolog_revision=prolog_revision,
            openai_revision=openai_revision,
            openai_config=openai_config,
        )
        prolog_keys = set(cast(DatasetDict, raw_pair[0]).keys())
        openai_keys = set(cast(DatasetDict, raw_pair[1]).keys())
        splits = [s for s in ("train", "test", "val") if s in prolog_keys and s in openai_keys]
    else:
        _splits_dir = splits_dir if splits_dir is not None else get_default_splits_dir()
        splits = [s for s in ("train", "test", "val") if _splits_available(_splits_dir, s)]

    if not splits:
        splits_dir_msg = str(splits_dir) if splits_dir is not None else "default"
        raise FileNotFoundError(
            "No common dataset splits found. "
            f"source={source!r} splits_dir={splits_dir_msg} raw_cache={DATA_RAW_DIR}"
        )

    if workers <= 0:
        raise ValueError("workers must be greater than 0")
    
    split_reports: dict[str, Any] = {}

    def _check_row(row: Tuple[int, str, str, str, str]) -> tuple[str, int, Optional[dict[str, Any]]]:
        i, prolog_input, prolog_output, openai_question, openai_answer = row
        if prolog_input.strip() != openai_question.strip():
            return (
                "alignment",
                i,
                {
                    "index": i,
                    "prolog_input": prolog_input,
                    "openai_question": openai_question,
                },
            )
        try:
            expected = _extract_openai_gsm8k_final_answer(openai_answer)
        except Exception as e:
            return (
                "extract",
                i,
                {
                    "index": i,
                    "error": f"{type(e).__name__}: {e}",
                    "openai_answer": openai_answer,
                },
            )
        got = execute_solve(prolog_output)
        if not got.ok:
            return (
                "exec",
                i,
                {
                    "index": i,
                    "error_type": got.error_type,
                    "error": got.error,
                    "stderr": got.stderr,
                },
            )
        if got.normalized_answer != expected:
            return (
                "mismatch",
                i,
                {
                    "index": i,
                    "got": str(got.normalized_answer),
                    "expected": str(expected),
                },
            )
        return ("ok", i, None)
    
    for split in splits:
        prolog_ds, openai_ds = _load_split_pair(
            split,
            source=source,
            splits_dir=splits_dir,
            raw_pair=raw_pair,
        )
        assert len(prolog_ds) == len(openai_ds)

        rows_to_check = len(prolog_ds)
        rows: list[Tuple[int, str, str, str, str]] = []
        for i in range(rows_to_check):
            rows.append(
                (
                    i,
                    str(prolog_ds[i]["input"]),
                    str(prolog_ds[i]["output"]),
                    str(openai_ds[i]["question"]),
                    str(openai_ds[i]["answer"]),
                )
            )

        alignment_mismatches: list[dict[str, Any]] = []
        extraction_failures: list[dict[str, Any]] = []
        execution_failures: list[dict[str, Any]] = []
        answer_mismatches: list[dict[str, Any]] = []
        matched = 0

        result_iter: Iterable[tuple[str, int, Optional[dict[str, Any]]]]
        executor: Optional[ThreadPoolExecutor] = None
        if workers == 1:
            result_iter = (_check_row(row) for row in rows)
        else:
            executor = ThreadPoolExecutor(max_workers=workers)
            result_iter = executor.map(_check_row, rows)

        try:
            for status, _idx, payload in _with_optional_tqdm(
                result_iter,
                total=rows_to_check,
                desc=f"Validate {split} (workers={workers})",
            ):
                if status == "ok":
                    matched += 1
                elif status == "alignment":
                    if payload is not None:
                        alignment_mismatches.append(payload)
                elif status == "extract":
                    if payload is not None:
                        extraction_failures.append(payload)
                elif status == "exec":
                    if payload is not None:
                        execution_failures.append(payload)
                elif status == "mismatch":
                    if payload is not None:
                        answer_mismatches.append(payload)
                else:
                    execution_failures.append(
                        {
                            "index": _idx,
                            "error_type": "unknown_status",
                            "error": f"Unexpected status: {status}",
                            "stderr": None,
                        }
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        total_failures = (
            len(alignment_mismatches)
            + len(extraction_failures)
            + len(execution_failures)
            + len(answer_mismatches)
        )
        split_reports[split] = {
            "counts": {
                "checked_rows": rows_to_check,
                "matched_rows": matched,
                "alignment_mismatches": len(alignment_mismatches),
                "answer_extraction_failures": len(extraction_failures),
                "execution_failures": len(execution_failures),
                "answer_mismatches": len(answer_mismatches),
                "total_failures": total_failures,
            },
            "rates": {
                "match_rate_over_checked": (matched / rows_to_check) if rows_to_check > 0 else 0.0,
                "failure_rate_over_checked": (total_failures / rows_to_check) if rows_to_check > 0 else 0.0,
            },
            "examples": {
                "alignment_mismatches": alignment_mismatches,
                "answer_extraction_failures": extraction_failures,
                "execution_failures": execution_failures,
                "answer_mismatches": answer_mismatches,
            },
        }

    grand_total_failures = sum(s["counts"]["total_failures"] for s in split_reports.values())
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "splits": split_reports,
        "total_failures": grand_total_failures,
    }
    if source == "raw":
        report["prolog_revision"] = prolog_revision
        report["openai_revision"] = openai_revision
        report["openai_config"] = openai_config
    elif source == "splits" and splits_dir is not None:
        report["splits_dir"] = str(splits_dir)

    out_path = _write_validation_report(report)
    for split in splits:
        c = split_reports[split]["counts"]
        print(
            "[Dataset Validation] split={split} checked={checked} matched={matched} "
            "align={align} extract={extract} exec={exec_fail} mismatch={mismatch} total_fail={total_fail}".format(
                split=split,
                checked=c["checked_rows"],
                matched=c["matched_rows"],
                align=c["alignment_mismatches"],
                extract=c["answer_extraction_failures"],
                exec_fail=c["execution_failures"],
                mismatch=c["answer_mismatches"],
                total_fail=c["total_failures"],
            )
        )
    print(f"[Dataset Validation] source={source}")
    print(f"[Dataset Validation] report={out_path}")

    if grand_total_failures:
        pytest.fail(
            "Dataset validation failed. "
            f"total_failures={grand_total_failures}, report={out_path}"
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GSM8K Prolog/OpenAI dataset alignment validation.")
    parser.add_argument(
        "--source",
        choices=("raw", "splits"),
        default="splits",
        help="Dataset source: raw (HF cache) or splits (prepared local).",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        help="When source=splits: directory with gsm8k_prolog/ and openai_gsm8k/. Default: versioned dir.",
    )
    parser.add_argument(
        "--prolog-revision",
        default=GSM8K_PROLOG_REVISION,
        help="When source=raw: HF revision for Thomas-X-Yang/gsm8k-prolog.",
    )
    parser.add_argument(
        "--openai-revision",
        default=OPENAI_GSM8K_REVISION,
        help="When source=raw: HF revision for openai/gsm8k.",
    )
    parser.add_argument(
        "--openai-config",
        default=OPENAI_GSM8K_CONFIG,
        help="When source=raw: HF config for openai/gsm8k.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads.",
    )
    args = parser.parse_args()
    run_dataset_alignment_and_answer_match(
        source=args.source,
        workers=args.workers,
        splits_dir=args.splits_dir,
        prolog_revision=args.prolog_revision,
        openai_revision=args.openai_revision,
        openai_config=args.openai_config,
    )
