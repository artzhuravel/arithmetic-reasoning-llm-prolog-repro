from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.hf_loaders import load_gsm8k_prolog, load_openai_gsm8k


def _normalize(text: str) -> str:
    # Normalization is intentionally minimal. We only ignore leading/trailing whitespace.
    return text.strip()


def _check_split(split_name: str) -> int:
    gsm8k_prolog = load_gsm8k_prolog()
    openai_gsm8k = load_openai_gsm8k()

    prolog_split = gsm8k_prolog[split_name]
    openai_split = openai_gsm8k[split_name]

    if len(prolog_split) != len(openai_split):
        print(
            f"[FAIL] Split '{split_name}' length mismatch: "
            f"gsm8k-prolog={len(prolog_split)} vs openai/gsm8k={len(openai_split)}"
        )
        return 1

    mismatches = 0
    for i in range(len(prolog_split)):
        prolog_question = _normalize(prolog_split[i]["input"])
        openai_question = _normalize(openai_split[i]["question"])
        if prolog_question != openai_question:
            mismatches += 1
            print(f"[MISMATCH] split={split_name} idx={i}")
            print(f"  prolog : {prolog_question[:200]!r}")
            print(f"  openai : {openai_question[:200]!r}")
            if mismatches >= 10:
                print("[FAIL] Stopping after 10 mismatches.")
                break

    if mismatches == 0:
        print(
            f"[OK] Split '{split_name}' is 100% aligned by index "
            f"({len(prolog_split)} / {len(prolog_split)} questions match)."
        )
        return 0

    print(
        f"[FAIL] Split '{split_name}' has mismatches: "
        f"{mismatches} / {len(prolog_split)} checked rows."
    )
    return 1


def main(splits: Iterable[str] = ("train", "test")) -> int:
    exit_code = 0
    for split_name in splits:
        exit_code |= _check_split(split_name)
    if exit_code == 0:
        print("[OK] All checked splits are aligned.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
