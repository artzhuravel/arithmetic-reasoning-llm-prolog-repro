from pathlib import Path
from typing import cast

import argparse
import re
from datasets import DatasetDict, load_dataset, load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"

# The paper (NAACL 2024, June 16-21, 2024) does not report exact Hugging Face commit SHAs.
# These defaults pin reproducible, pre-publication revisions:
# - gsm8k-prolog: last data-changing commit before later README-only updates
# - openai/gsm8k: latest repo revision available before the paper publication window
GSM8K_PROLOG_REVISION = "49fe7b81d6fb3d6e96f39dce4f9b3afc3462e309"
OPENAI_GSM8K_REVISION = "cc7b047b6e5bb11b4f1af84efc572db110a51b3c"
OPENAI_GSM8K_CONFIG = "main"


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())


def load_gsm8k_prolog(*, prolog_revision: str) -> DatasetDict:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    prolog_rev_dir = f"revision_{_safe_component(prolog_revision)}"
    gsm8k_prolog_path = DATA_RAW_DIR / "gsm8k-prolog" / prolog_rev_dir
    if gsm8k_prolog_path.exists():
        return cast(DatasetDict, load_from_disk(str(gsm8k_prolog_path)))

    gsm8k_prolog = cast(
        DatasetDict,
        load_dataset("Thomas-X-Yang/gsm8k-prolog", revision=prolog_revision),
    )
    gsm8k_prolog.save_to_disk(str(gsm8k_prolog_path))
    print(f"Saved GSM8K-Prolog dataset to {gsm8k_prolog_path} (revision={prolog_revision})")
    return gsm8k_prolog


def load_openai_gsm8k(*, openai_revision: str, openai_config: str) -> DatasetDict:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    openai_cfg_dir = f"config_{_safe_component(openai_config)}"
    openai_rev_dir = f"revision_{_safe_component(openai_revision)}"
    openai_gsm8k_path = DATA_RAW_DIR / "openai-gsm8k" / openai_cfg_dir / openai_rev_dir
    if openai_gsm8k_path.exists():
        return cast(DatasetDict, load_from_disk(str(openai_gsm8k_path)))

    openai_gsm8k = cast(
        DatasetDict,
        load_dataset("openai/gsm8k", openai_config, revision=openai_revision),
    )
    openai_gsm8k.save_to_disk(str(openai_gsm8k_path))
    print(
        f"Saved OpenAI GSM8K dataset to {openai_gsm8k_path} "
        f"(config={openai_config}, revision={openai_revision})"
    )
    return openai_gsm8k


def load_gsm8k_datasets(
    *,
    prolog_revision: str = GSM8K_PROLOG_REVISION,
    openai_revision: str = OPENAI_GSM8K_REVISION,
    openai_config: str = OPENAI_GSM8K_CONFIG,
) -> tuple[DatasetDict, DatasetDict]:
    """
    Load both GSM8K-Prolog and OpenAI GSM8K datasets (from cache or HF), together.
    """
    gsm8k_prolog = load_gsm8k_prolog(prolog_revision=prolog_revision)
    openai_gsm8k = load_openai_gsm8k(
        openai_revision=openai_revision, openai_config=openai_config
    )
    return gsm8k_prolog, openai_gsm8k


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load/cache GSM8K-Prolog and OpenAI GSM8K datasets.")
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
    parser.add_argument(
        "--dataset",
        choices=("both", "prolog", "openai"),
        default="both",
        help="Choose which dataset(s) to download/load.",
    )
    args = parser.parse_args()

    if args.dataset == "prolog":
        gsm8k_prolog = load_gsm8k_prolog(prolog_revision=args.prolog_revision)
        print("[hf_loaders] Loaded dataset: gsm8k-prolog")
        print(f"[hf_loaders] gsm8k-prolog splits: {list(gsm8k_prolog.keys())}")
        for split in sorted(gsm8k_prolog.keys()):
            print(f"[hf_loaders] split={split} prolog_rows={len(gsm8k_prolog[split])}")
    elif args.dataset == "openai":
        openai_gsm8k = load_openai_gsm8k(
            openai_revision=args.openai_revision, openai_config=args.openai_config
        )
        print("[hf_loaders] Loaded dataset: openai/gsm8k")
        print(f"[hf_loaders] openai/gsm8k splits: {list(openai_gsm8k.keys())}")
        for split in sorted(openai_gsm8k.keys()):
            print(f"[hf_loaders] split={split} openai_rows={len(openai_gsm8k[split])}")
    else:
        gsm8k_prolog, openai_gsm8k = load_gsm8k_datasets(
            prolog_revision=args.prolog_revision,
            openai_revision=args.openai_revision,
            openai_config=args.openai_config,
        )
        print("[hf_loaders] Loaded datasets")
        print(f"[hf_loaders] gsm8k-prolog splits: {list(gsm8k_prolog.keys())}")
        print(f"[hf_loaders] openai/gsm8k splits: {list(openai_gsm8k.keys())}")
        for split in sorted(set(gsm8k_prolog.keys()) & set(openai_gsm8k.keys())):
            print(
                f"[hf_loaders] split={split} prolog_rows={len(gsm8k_prolog[split])} "
                f"openai_rows={len(openai_gsm8k[split])}"
            )
