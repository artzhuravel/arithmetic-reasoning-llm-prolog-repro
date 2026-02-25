from typing import cast, Optional
from datasets import Dataset, DatasetDict, load_from_disk
from pathlib import Path
from src.data.hf_loaders import load_gsm8k_prolog, load_openai_gsm8k

import random

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_SPLITS_DIR = REPO_ROOT / "data" / "splits"


def _normalize(text: str) -> str:
    # Normalization is intentionally minimal. We only ignore leading/trailing whitespace.
    return text.strip()

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

def prepare_splits(train_size: Optional[int] = None,
                   test_size:  Optional[int] = None,
                   validation_size:  Optional[int] = None,
                   seed: int = 42) -> tuple[DatasetDict, DatasetDict]:
    
    DATA_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    gsm8k_prolog_train_path = DATA_SPLITS_DIR / "gsm8k_prolog" / "train"
    gsm8k_prolog_test_path = DATA_SPLITS_DIR / "gsm8k_prolog" / "test"
    openai_gsm8k_train_path = DATA_SPLITS_DIR / "openai_gsm8k" / "train"
    openai_gsm8k_test_path = DATA_SPLITS_DIR / "openai_gsm8k" / "test"
    gsm8k_prolog_val_path = DATA_SPLITS_DIR / "gsm8k_prolog" / "val"
    openai_gsm8k_val_path = DATA_SPLITS_DIR / "openai_gsm8k" / "val"

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
        return gsm8k_prolog_splits, openai_gsm8k_splits
    
    gsm8k_prolog = load_gsm8k_prolog()
    openai_gsm8k = load_openai_gsm8k()
    
    # Split GSM8K-Prolog into train and test sets
    gsm8k_prolog_train = cast(Dataset, gsm8k_prolog["train"])
    gsm8k_prolog_test = cast(Dataset, gsm8k_prolog["test"])

    # Split OpenAI GSM8K into train and test sets
    openai_gsm8k_train = cast(Dataset, openai_gsm8k["train"])
    openai_gsm8k_test = cast(Dataset, openai_gsm8k["test"])
    
    # Check if splits are aligned
    if _check_split(gsm8k_prolog_train, openai_gsm8k_train, "train") != 0:
        raise ValueError("Train splits are not aligned")
    if _check_split(gsm8k_prolog_test, openai_gsm8k_test, "test") != 0:
        raise ValueError("Test splits are not aligned")
    
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
        train_size = len(gsm8k_prolog["train"])
    if test_size is None:
        test_size = len(gsm8k_prolog["test"])
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
    print(f"Saved splits to {DATA_SPLITS_DIR}")
    
    gsm8k_prolog_splits = DatasetDict(
        {"train": gsm8k_prolog_train, "val": gsm8k_prolog_val, "test": gsm8k_prolog_test}
    )
    openai_gsm8k_splits = DatasetDict(
        {"train": openai_gsm8k_train, "val": openai_gsm8k_val, "test": openai_gsm8k_test}
    )

    return gsm8k_prolog_splits, openai_gsm8k_splits
