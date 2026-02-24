from pathlib import Path
from typing import cast

from datasets import DatasetDict, load_dataset, load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"

# The paper (NAACL 2024, June 16-21, 2024) does not report exact Hugging Face commit SHAs.
# These defaults pin reproducible, pre-publication revisions:
# - gsm8k-prolog: last data-changing commit before later README-only updates
# - openai/gsm8k: latest repo revision available before the paper publication window
GSM8K_PROLOG_REVISION = "450cda77c563dd45687e9fc4c4fff2835cab7fde"
OPENAI_GSM8K_REVISION = "e53f048856ff4f594e959d75785d2c2d37b678ee"
OPENAI_GSM8K_CONFIG = "main"


def load_gsm8k_prolog(revision: str = GSM8K_PROLOG_REVISION) -> DatasetDict:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    gsm8k_prolog_path = DATA_RAW_DIR / "gsm8k-prolog"
    if gsm8k_prolog_path.exists():
        return cast(DatasetDict, load_from_disk(str(gsm8k_prolog_path)))
    else:
        gsm8k_prolog = cast(
            DatasetDict,
            load_dataset("Thomas-X-Yang/gsm8k-prolog", revision=revision),
        )
        gsm8k_prolog_path.parent.mkdir(parents=True, exist_ok=True)
        gsm8k_prolog.save_to_disk(str(gsm8k_prolog_path))
        print(f"Saved GSM8K-Prolog dataset to {gsm8k_prolog_path} (revision={revision})")
        return gsm8k_prolog
    

def load_openai_gsm8k(revision: str = OPENAI_GSM8K_REVISION):
    openai_gsm8k_path = DATA_RAW_DIR / "openai-gsm8k"
    if openai_gsm8k_path.exists():
        return load_from_disk(str(openai_gsm8k_path))
    else:
        openai_gsm8k = cast(
            DatasetDict,
            load_dataset("openai/gsm8k", OPENAI_GSM8K_CONFIG, revision=revision),
        )
        openai_gsm8k_path.parent.mkdir(parents=True, exist_ok=True)
        openai_gsm8k.save_to_disk(str(openai_gsm8k_path))
        print(
            f"Saved OpenAI GSM8K dataset to {openai_gsm8k_path} "
            f"(config={OPENAI_GSM8K_CONFIG}, revision={revision})"
        )
        return openai_gsm8k
    
