from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm

from src.data.prepare_splits import get_default_splits_dir
from src.prolog.execute import execute_solve, normalize_answer_for_eval


GROUND_TRUTH_FILENAME = "ground_truth_by_prompt.json"


def _infer_ground_truth_path(proper_path: Path) -> Path:
    for candidate_dir in [proper_path, *proper_path.parents]:
        p = candidate_dir / GROUND_TRUTH_FILENAME
        if p.exists():
            return p

    default_path = get_default_splits_dir() / GROUND_TRUTH_FILENAME
    if default_path.exists():
        return default_path

    raise FileNotFoundError(
        "Could not find ground truth file. "
        f"Expected {GROUND_TRUTH_FILENAME} near {proper_path} or in {get_default_splits_dir()}."
    )


def _load_ground_truth(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    all_map = payload.get("all")
    if not isinstance(all_map, dict):
        raise ValueError(f"Invalid ground truth format in {path}: missing dict key 'all'.")
    return {str(k).strip(): float(v) for k, v in all_map.items()}


def _iter_rows(proper_ds: Dataset | DatasetDict) -> Iterable[tuple[str, int, str, str]]:
    if isinstance(proper_ds, DatasetDict):
        for split, ds in proper_ds.items():
            for i in range(len(ds)):
                yield str(split), i, str(ds[i]["input"]).strip(), str(ds[i]["output"])
    else:
        for i in range(len(proper_ds)):
            yield "dataset", i, str(proper_ds[i]["input"]).strip(), str(proper_ds[i]["output"])


def validate_proper_dataset(
    *,
    proper_path: Path | None = None,
    workers: int = 10,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be > 0")

    if proper_path is None:
        proper_path = get_default_splits_dir() / "gsm8k_proper" / "ratio_1to10"
    if not proper_path.exists():
        raise FileNotFoundError(f"PROPER dataset path not found: {proper_path}")

    proper_ds = load_from_disk(str(proper_path))
    gt_path = _infer_ground_truth_path(proper_path)
    gt_map = _load_ground_truth(gt_path)

    rows = list(_iter_rows(proper_ds))

    def _check_row(item: tuple[str, int, str, str]) -> tuple[str, dict[str, Any] | None]:
        split, idx, prompt, output = item

        if prompt not in gt_map:
            return (
                "missing_ground_truth",
                {"split": split, "index": idx, "prompt": prompt},
            )

        expected = normalize_answer_for_eval(gt_map[prompt])
        got = execute_solve(output)
        if not got.ok:
            return (
                "execution_failure",
                {
                    "split": split,
                    "index": idx,
                    "error_type": got.error_type,
                    "error": got.error,
                    "stderr": got.stderr,
                },
            )

        if got.normalized_answer != expected:
            return (
                "answer_mismatch",
                {
                    "split": split,
                    "index": idx,
                    "expected": expected,
                    "got": got.normalized_answer,
                },
            )
        return "ok", None

    ok = 0
    missing_gt: list[dict[str, Any]] = []
    exec_fail: list[dict[str, Any]] = []
    mismatch: list[dict[str, Any]] = []

    if workers == 1:
        result_iter = (_check_row(row) for row in rows)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        result_iter = executor.map(_check_row, rows)

    try:
        for status, payload in tqdm(
            result_iter, total=len(rows), desc=f"Validate PROPER (workers={workers})", unit="row"
        ):
            if status == "ok":
                ok += 1
            elif status == "missing_ground_truth" and payload is not None:
                missing_gt.append(payload)
            elif status == "execution_failure" and payload is not None:
                exec_fail.append(payload)
            elif status == "answer_mismatch" and payload is not None:
                mismatch.append(payload)
    finally:
        if workers != 1:
            executor.shutdown(wait=True)  # type: ignore[name-defined]

    total_failures = len(missing_gt) + len(exec_fail) + len(mismatch)
    report: dict[str, Any] = {
        "proper_path": str(proper_path),
        "ground_truth_path": str(gt_path),
        "workers": workers,
        "counts": {
            "checked_rows": len(rows),
            "ok_rows": ok,
            "missing_ground_truth": len(missing_gt),
            "execution_failures": len(exec_fail),
            "answer_mismatches": len(mismatch),
            "total_failures": total_failures,
        },
        "examples": {
            "missing_ground_truth": missing_gt[:20],
            "execution_failures": exec_fail[:20],
            "answer_mismatches": mismatch[:20],
        },
    }

    print(
        "[PROPER Validation] checked={checked} ok={ok} missing_gt={missing_gt} "
        "exec_fail={exec_fail} mismatch={mismatch} total_failures={total}".format(
            checked=report["counts"]["checked_rows"],
            ok=report["counts"]["ok_rows"],
            missing_gt=report["counts"]["missing_ground_truth"],
            exec_fail=report["counts"]["execution_failures"],
            mismatch=report["counts"]["answer_mismatches"],
            total=report["counts"]["total_failures"],
        )
    )

    if total_failures:
        raise AssertionError(
            "PROPER dataset validation failed. "
            f"total_failures={total_failures} (see report dict for examples)."
        )

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate PROPER dataset outputs against ground truth by executing every row."
    )
    parser.add_argument(
        "--proper-path",
        type=Path,
        default=get_default_splits_dir() / "gsm8k_proper" / "ratio_1to10",
        help="Path to PROPER dataset directory (DatasetDict or split dataset).",
    )
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    report = validate_proper_dataset(proper_path=args.proper_path, workers=args.workers)
    print(json.dumps(report, indent=2))
