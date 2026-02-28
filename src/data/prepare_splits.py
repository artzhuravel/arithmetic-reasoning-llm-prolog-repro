from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast, Optional
from datasets import Dataset, DatasetDict, load_from_disk
from pathlib import Path
from src.data.hf_loaders import (
    GSM8K_PROLOG_REVISION,
    OPENAI_GSM8K_CONFIG,
    OPENAI_GSM8K_REVISION,
    load_gsm8k_datasets,
)
from src.prolog.execute import execute_solve, normalize_answer_for_eval

import argparse
import json
import random
import re

def _noop_tqdm(iterable: Any, **_: Any) -> Any:
    return iterable

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = _noop_tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_SPLITS_DIR = REPO_ROOT / "data" / "splits"
_GSM8K_FINAL_ANSWER_RE = re.compile(r"####\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)")


def _normalize(text: str) -> str:
    # Normalization is intentionally minimal. We only ignore leading/trailing whitespace.
    return text.strip()


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())


def _get_splits_base_dir(
    *,
    prolog_revision: str,
    openai_revision: str,
    openai_config: str,
) -> Path:
    """
    Return a versioned directory for splits. Both gsm8k_prolog and openai_gsm8k
    splits are stored under this folder. By default uses GSM8K_PROLOG_REVISION,
    OPENAI_GSM8K_CONFIG, and OPENAI_GSM8K_REVISION.
    """
    version_dir = (
        f"prolog_rev_{_safe_component(prolog_revision)}__"
        f"openai_cfg_{_safe_component(openai_config)}__"
        f"openai_rev_{_safe_component(openai_revision)}"
    )
    return DATA_SPLITS_DIR / version_dir


def get_default_splits_dir() -> Path:
    """Return the splits directory for default dataset versions."""
    return _get_splits_base_dir(
        prolog_revision=GSM8K_PROLOG_REVISION,
        openai_revision=OPENAI_GSM8K_REVISION,
        openai_config=OPENAI_GSM8K_CONFIG,
    )


def _check_split(gsm8k_prolog_split: Dataset, openai_gsm8k_split: Dataset, split_name: str) -> int:

    if len(gsm8k_prolog_split) != len(openai_gsm8k_split):
        print(
            f"[FAIL] Split length mismatch: "
            f"gsm8k-prolog={len(gsm8k_prolog_split)} vs openai/gsm8k={len(openai_gsm8k_split)}"
        )
        return 1

    mismatches = 0
    for i in range(len(gsm8k_prolog_split)):
        prolog_question = _normalize(gsm8k_prolog_split[i]["input"])
        openai_question = _normalize(openai_gsm8k_split[i]["question"])
        if prolog_question != openai_question:
            mismatches += 1
            print(f"[MISMATCH] idx={i}")
            print(f"  prolog : {prolog_question[:200]!r}")
            print(f"  openai : {openai_question[:200]!r}")


    if mismatches == 0:
        print(
            f"[OK] {split_name} split is 100% aligned by index "
            f"({len(gsm8k_prolog_split)} / {len(gsm8k_prolog_split)} questions match)."
        )
        return 0

    print(
        f"[FAIL] {split_name} split has mismatches: "
        f"{mismatches} / {len(gsm8k_prolog_split)} checked rows."
    )
    return 1


def _extract_openai_gsm8k_final_answer(answer_text: str) -> str:
    m = _GSM8K_FINAL_ANSWER_RE.search(answer_text)
    if m is None:
        raise ValueError("Could not find final GSM8K answer marker ('#### <number>').")
    raw = m.group(1).replace(",", "")
    return normalize_answer_for_eval(raw)


def _build_ground_truth_map(
    gsm8k_prolog_split: Dataset,
    openai_gsm8k_split: Dataset,
    *,
    split_name: str,
) -> dict[str, float]:
    if len(gsm8k_prolog_split) != len(openai_gsm8k_split):
        raise ValueError(
            f"Cannot build ground truth for {split_name}: length mismatch "
            f"{len(gsm8k_prolog_split)} vs {len(openai_gsm8k_split)}"
        )

    ground_truth: dict[str, float] = {}
    for i in range(len(gsm8k_prolog_split)):
        prompt = _normalize(str(gsm8k_prolog_split[i]["input"]))
        question = _normalize(str(openai_gsm8k_split[i]["question"]))
        if prompt != question:
            raise ValueError(
                f"Cannot build ground truth for {split_name}: prompt mismatch at index {i}."
            )
        answer_norm = _extract_openai_gsm8k_final_answer(str(openai_gsm8k_split[i]["answer"]))
        try:
            answer_num = float(answer_norm)
        except ValueError as e:
            raise ValueError(
                f"Non-numeric normalized answer in {split_name} at index {i}: {answer_norm!r}"
            ) from e

        if prompt in ground_truth and ground_truth[prompt] != answer_num:
            raise ValueError(
                f"Conflicting ground truth for duplicated prompt in {split_name} at index {i}."
            )
        ground_truth[prompt] = answer_num

    return ground_truth


def _write_ground_truth_file(
    *,
    splits_base_dir: Path,
    gsm8k_prolog_train: Dataset,
    openai_gsm8k_train: Dataset,
    gsm8k_prolog_val: Dataset,
    openai_gsm8k_val: Dataset,
    gsm8k_prolog_test: Dataset,
    openai_gsm8k_test: Dataset,
) -> Path:
    gt_train = _build_ground_truth_map(
        gsm8k_prolog_train, openai_gsm8k_train, split_name="train"
    )
    gt_val = _build_ground_truth_map(
        gsm8k_prolog_val, openai_gsm8k_val, split_name="val"
    )
    gt_test = _build_ground_truth_map(
        gsm8k_prolog_test, openai_gsm8k_test, split_name="test"
    )

    gt_all: dict[str, float] = {}
    for split_map in (gt_train, gt_val, gt_test):
        for prompt, answer in split_map.items():
            if prompt in gt_all and gt_all[prompt] != answer:
                raise ValueError("Conflicting ground truth answers across splits for same prompt.")
            gt_all[prompt] = answer

    payload = {
        "train": gt_train,
        "val": gt_val,
        "test": gt_test,
        "all": gt_all,
    }
    out_path = splits_base_dir / "ground_truth_by_prompt.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Saved ground truth mapping to {out_path} "
        f"(train={len(gt_train)}, val={len(gt_val)}, test={len(gt_test)}, all={len(gt_all)})"
    )
    return out_path


def _filter_split_by_execution_match(
    gsm8k_prolog_split: Dataset,
    openai_gsm8k_split: Dataset,
    *,
    split_name: str,
    workers: int = 1,
) -> tuple[Dataset, Dataset]:
    if len(gsm8k_prolog_split) != len(openai_gsm8k_split):
        raise ValueError(
            f"Split length mismatch for {split_name}: "
            f"gsm8k-prolog={len(gsm8k_prolog_split)} vs openai/gsm8k={len(openai_gsm8k_split)}"
        )

    if workers <= 0:
        raise ValueError("workers must be greater than 0")

    kept_indices: list[int] = []
    eliminated_indices: list[int] = []
    reason_counts: dict[str, int] = {
        "alignment_mismatch": 0,
        "answer_extraction_failure": 0,
        "execution_failure": 0,
        "answer_mismatch": 0,
    }

    def _check_row(i: int) -> tuple[bool, Optional[str]]:
        prolog_question = _normalize(str(gsm8k_prolog_split[i]["input"]))
        openai_question = _normalize(str(openai_gsm8k_split[i]["question"]))
        if prolog_question != openai_question:
            return False, "alignment_mismatch"

        try:
            expected = _extract_openai_gsm8k_final_answer(str(openai_gsm8k_split[i]["answer"]))
        except Exception:
            return False, "answer_extraction_failure"

        exec_result = execute_solve(str(gsm8k_prolog_split[i]["output"]))
        if not exec_result.ok:
            return False, "execution_failure"

        if exec_result.normalized_answer != expected:
            return False, "answer_mismatch"

        return True, None

    indices = list(range(len(gsm8k_prolog_split)))
    if workers == 1:
        for i in tqdm(indices, total=len(indices), desc=f"Filter {split_name}", unit="row"):
            ok, reason = _check_row(i)
            if ok:
                kept_indices.append(i)
            else:
                eliminated_indices.append(i)
                if reason is not None:
                    reason_counts[reason] += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            mapped = executor.map(_check_row, indices)
            for i, (ok, reason) in zip(
                indices,
                tqdm(mapped, total=len(indices), desc=f"Filter {split_name}", unit="row"),
            ):
                if ok:
                    kept_indices.append(i)
                else:
                    eliminated_indices.append(i)
                    if reason is not None:
                        reason_counts[reason] += 1

    print(
        f"[FILTER] {split_name}: kept={len(kept_indices)} / {len(gsm8k_prolog_split)} "
        f"eliminated={len(eliminated_indices)}"
    )
    print(f"[FILTER] {split_name} eliminated indices: {eliminated_indices}")
    print(
        f"[FILTER] {split_name} reasons: "
        f"alignment_mismatch={reason_counts['alignment_mismatch']}, "
        f"answer_extraction_failure={reason_counts['answer_extraction_failure']}, "
        f"execution_failure={reason_counts['execution_failure']}, "
        f"answer_mismatch={reason_counts['answer_mismatch']}"
    )

    return gsm8k_prolog_split.select(kept_indices), openai_gsm8k_split.select(kept_indices)


def prepare_splits(train_size: Optional[int] = None,
                   test_size:  Optional[int] = None,
                   validation_size:  Optional[int] = None,
                   seed: int = 42,
                   workers: int = 10,
                   prolog_revision: str = GSM8K_PROLOG_REVISION,
                   openai_revision: str = OPENAI_GSM8K_REVISION,
                   openai_config: str = OPENAI_GSM8K_CONFIG) -> tuple[DatasetDict, DatasetDict]:
    
    splits_base_dir = _get_splits_base_dir(
        prolog_revision=prolog_revision,
        openai_revision=openai_revision,
        openai_config=openai_config,
    )
    splits_base_dir.mkdir(parents=True, exist_ok=True)
    gsm8k_prolog_train_path = splits_base_dir / "gsm8k_prolog" / "train"
    gsm8k_prolog_test_path = splits_base_dir / "gsm8k_prolog" / "test"
    openai_gsm8k_train_path = splits_base_dir / "openai_gsm8k" / "train"
    openai_gsm8k_test_path = splits_base_dir / "openai_gsm8k" / "test"
    gsm8k_prolog_val_path = splits_base_dir / "gsm8k_prolog" / "val"
    openai_gsm8k_val_path = splits_base_dir / "openai_gsm8k" / "val"

    if gsm8k_prolog_train_path.exists() and openai_gsm8k_train_path.exists() and gsm8k_prolog_test_path.exists() and openai_gsm8k_test_path.exists() and gsm8k_prolog_val_path.exists() and openai_gsm8k_val_path.exists():
        print("Splits already exist, loading from disk")
        
        gsm8k_prolog_train = cast(Dataset, load_from_disk(str(gsm8k_prolog_train_path)))
        openai_gsm8k_train = cast(Dataset, load_from_disk(str(openai_gsm8k_train_path)))
        gsm8k_prolog_test = cast(Dataset, load_from_disk(str(gsm8k_prolog_test_path)))
        openai_gsm8k_test = cast(Dataset, load_from_disk(str(openai_gsm8k_test_path)))
        gsm8k_prolog_val = cast(Dataset, load_from_disk(str(gsm8k_prolog_val_path)))
        openai_gsm8k_val = cast(Dataset, load_from_disk(str(openai_gsm8k_val_path)))

        gsm8k_prolog_splits = DatasetDict(
            {"train": gsm8k_prolog_train, "val": gsm8k_prolog_val, "test": gsm8k_prolog_test}
        )
        openai_gsm8k_splits = DatasetDict(
            {"train": openai_gsm8k_train, "val": openai_gsm8k_val, "test": openai_gsm8k_test}
        )
        _write_ground_truth_file(
            splits_base_dir=splits_base_dir,
            gsm8k_prolog_train=gsm8k_prolog_train,
            openai_gsm8k_train=openai_gsm8k_train,
            gsm8k_prolog_val=gsm8k_prolog_val,
            openai_gsm8k_val=openai_gsm8k_val,
            gsm8k_prolog_test=gsm8k_prolog_test,
            openai_gsm8k_test=openai_gsm8k_test,
        )
        return gsm8k_prolog_splits, openai_gsm8k_splits
    
    gsm8k_prolog, openai_gsm8k = load_gsm8k_datasets(
        prolog_revision=prolog_revision,
        openai_revision=openai_revision,
        openai_config=openai_config,
    )
    
    # Split GSM8K-Prolog into train and test sets
    gsm8k_prolog_train = cast(Dataset, gsm8k_prolog["train"])
    gsm8k_prolog_test = cast(Dataset, gsm8k_prolog["test"])

    # Split OpenAI GSM8K into train and test sets
    openai_gsm8k_train = cast(Dataset, openai_gsm8k["train"])
    openai_gsm8k_test = cast(Dataset, openai_gsm8k["test"])
    
    # Check if splits are aligned
    _check_split(gsm8k_prolog_train, openai_gsm8k_train, "train")
    _check_split(gsm8k_prolog_test, openai_gsm8k_test, "test")

    # Filter out rows that fail alignment/execution/answer match across full train and test.
    gsm8k_prolog_train, openai_gsm8k_train = _filter_split_by_execution_match(
        gsm8k_prolog_train, openai_gsm8k_train, split_name="train", workers=workers
    )
    gsm8k_prolog_test, openai_gsm8k_test = _filter_split_by_execution_match(
        gsm8k_prolog_test, openai_gsm8k_test, split_name="test", workers=workers
    )
    
    # Validate requested subset sizes against actual split sizes
    if train_size is not None and train_size > len(gsm8k_prolog_train):
        raise ValueError(
            f"train_size={train_size} exceeds train split size={len(gsm8k_prolog_train)}"
        )
    if test_size is not None and test_size > len(gsm8k_prolog_test):
        raise ValueError(
            f"test_size={test_size} exceeds test split size={len(gsm8k_prolog_test)}"
        )
    
    if train_size is None:
        train_size = len(gsm8k_prolog_train)
    if test_size is None:
        test_size = len(gsm8k_prolog_test)
    if validation_size is None:
        validation_size = min(100, train_size // 10)
    
    if validation_size > train_size:
        print(f"Validation size {validation_size} is greater than train size {train_size}")
        raise ValueError("Validation size cannot be greater than train size")

    # Shuffle train and test indices
    train_indices = list(range(len(gsm8k_prolog_train)))
    test_indices = list(range(len(gsm8k_prolog_test)))
    random.Random(seed).shuffle(train_indices)
    random.Random(seed).shuffle(test_indices)

    # Subset train and test indices
    train_indices = train_indices[:train_size]
    test_indices = test_indices[:test_size]

    # Validation split from shuffled train
    val_indices = train_indices[:validation_size]
    train_indices_no_val = train_indices[validation_size:]

    # Apply same indices to BOTH datasets to preserve alignment
    # Extract validation from original train BEFORE reassigning train (val_indices refer to original)
    gsm8k_prolog_val = gsm8k_prolog_train.select(val_indices)
    openai_gsm8k_val = openai_gsm8k_train.select(val_indices)

    gsm8k_prolog_train = gsm8k_prolog_train.select(train_indices_no_val)
    openai_gsm8k_train = openai_gsm8k_train.select(train_indices_no_val)

    gsm8k_prolog_test = gsm8k_prolog_test.select(test_indices)
    openai_gsm8k_test = openai_gsm8k_test.select(test_indices)
    
    # Save splits
    gsm8k_prolog_train.save_to_disk(gsm8k_prolog_train_path)
    openai_gsm8k_train.save_to_disk(openai_gsm8k_train_path)
    gsm8k_prolog_test.save_to_disk(gsm8k_prolog_test_path)
    openai_gsm8k_test.save_to_disk(openai_gsm8k_test_path)
    gsm8k_prolog_val.save_to_disk(gsm8k_prolog_val_path)
    openai_gsm8k_val.save_to_disk(openai_gsm8k_val_path)

    # Write manifest with dataset versions used
    manifest_path = splits_base_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "prolog_revision": prolog_revision,
                "openai_config": openai_config,
                "openai_revision": openai_revision,
                "counts": {
                    "train": len(gsm8k_prolog_train),
                    "val": len(gsm8k_prolog_val),
                    "test": len(gsm8k_prolog_test),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_ground_truth_file(
        splits_base_dir=splits_base_dir,
        gsm8k_prolog_train=gsm8k_prolog_train,
        openai_gsm8k_train=openai_gsm8k_train,
        gsm8k_prolog_val=gsm8k_prolog_val,
        openai_gsm8k_val=openai_gsm8k_val,
        gsm8k_prolog_test=gsm8k_prolog_test,
        openai_gsm8k_test=openai_gsm8k_test,
    )
    print(f"Saved splits to {splits_base_dir}")
    
    gsm8k_prolog_splits = DatasetDict(
        {"train": gsm8k_prolog_train, "val": gsm8k_prolog_val, "test": gsm8k_prolog_test}
    )
    openai_gsm8k_splits = DatasetDict(
        {"train": openai_gsm8k_train, "val": openai_gsm8k_val, "test": openai_gsm8k_test}
    )

    return gsm8k_prolog_splits, openai_gsm8k_splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare aligned/validated GSM8K-Prolog and OpenAI GSM8K splits."
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Optional train subset size after filtering.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="Optional test subset size after filtering.",
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=None,
        help="Optional validation size sampled from filtered train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling/sampling.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker threads for split filtering.",
    )
    parser.add_argument(
        "--prolog-revision",
        default=GSM8K_PROLOG_REVISION,
        help="HF revision for Thomas-X-Yang/gsm8k-prolog.",
    )
    parser.add_argument(
        "--openai-revision",
        default=OPENAI_GSM8K_REVISION,
        help="HF revision for openai/gsm8k.",
    )
    parser.add_argument(
        "--openai-config",
        default=OPENAI_GSM8K_CONFIG,
        help="HF config for openai/gsm8k.",
    )
    args = parser.parse_args()

    prepare_splits(
        train_size=args.train_size,
        test_size=args.test_size,
        validation_size=args.validation_size,
        seed=args.seed,
        workers=args.workers,
        prolog_revision=args.prolog_revision,
        openai_revision=args.openai_revision,
        openai_config=args.openai_config,
    )
