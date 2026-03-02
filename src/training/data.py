from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

from datasets import Dataset, DatasetDict, load_from_disk

_OPENAI_GSM8K_DEFAULT_INSTRUCTION = "Please solve the given math problem."


@dataclass(frozen=True)
class PromptTemplate:
    """
    Minimal text template for SFT records.

    Keep this simple while learning, then iterate.
    """

    instruction_header: str = "### Instruction"
    input_header: str = "### Input"
    output_header: str = "### Output"


PROLOG_PROMPT_TEMPLATE = PromptTemplate(
    instruction_header="### Instruction",
    input_header="### Input",
    output_header="### Output",
)

OPENAI_GSM8K_PROMPT_TEMPLATE = PromptTemplate(
    instruction_header="### Instruction",
    input_header="### Question",
    output_header="### Answer",
)


def load_prepared_dataset(dataset_dir: Path) -> DatasetDict:
    """
    Load a previously prepared dataset from disk.

    Supports both layouts:
    - A DatasetDict saved with `save_to_disk()`
    - A directory containing split subdirectories (e.g., train/val/test),
      each saved independently with `save_to_disk()`
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    try:
        loaded = load_from_disk(str(dataset_dir))
    except FileNotFoundError:
        loaded_splits: dict[str, Dataset] = {}
        for split_name in ("train", "val", "validation", "dev", "test"):
            split_dir = dataset_dir / split_name
            if not split_dir.exists():
                continue
            split_loaded = load_from_disk(str(split_dir))
            if not isinstance(split_loaded, Dataset):
                continue
            loaded_splits[split_name] = cast(Dataset, split_loaded)

        if not loaded_splits:
            raise FileNotFoundError(
                f"Directory {dataset_dir} is neither a Dataset/DatasetDict "
                "directory nor a split-parent directory with saved split datasets."
            )
        return DatasetDict(loaded_splits.items())

    if isinstance(loaded, DatasetDict):
        return cast(DatasetDict, loaded)
    if isinstance(loaded, Dataset):
        # Support a single-split dataset directory.
        return DatasetDict([("train", cast(Dataset, loaded))])

    raise TypeError(
        f"Expected Dataset or DatasetDict at {dataset_dir}, got {type(loaded).__name__}."
    )


def _infer_template_from_dataset_dir(dataset_dir: Path) -> Optional[PromptTemplate]:
    """
    Infer template from dataset path components (starting from the last directory).

    Heuristics:
    - OpenAI template: component contains "openai_gsm8k" or "openai" (without mixed prolog marker)
    - Prolog template: component contains "gsm8k_prolog", "gsm8k_proper", "prolog", "proper", or starts with "ratio_"

    Returns None when uncertain; caller can then fall back to schema-based inference.
    """
    for component in reversed(dataset_dir.parts):
        name = component.lower()
        if not name:
            continue

        if "openai_gsm8k" in name:
            return OPENAI_GSM8K_PROMPT_TEMPLATE
        if (
            "gsm8k_prolog" in name
            or "gsm8k_proper" in name
            or "proper" in name
            or name.startswith("ratio_")
        ):
            return PROLOG_PROMPT_TEMPLATE

        # Generic fallback markers (skip mixed components like prolog_rev__openai_cfg__...)
        if "openai" in name and "prolog" not in name:
            return OPENAI_GSM8K_PROMPT_TEMPLATE
        if "prolog" in name and "openai" not in name:
            return PROLOG_PROMPT_TEMPLATE

    return None


def format_record_for_sft(
    row: dict[str, Any],
    *,
    template: PromptTemplate,
) -> dict[str, str]:
    """
    Build a single training text field from one of:
    - Prolog-style rows: instruction/input/output
    - OpenAI GSM8K rows: question/answer (with a default instruction)
    """
    
    if {"instruction", "input", "output"}.issubset(row):
        instruction = str(row["instruction"]).strip()
        user_input = str(row["input"]).strip()
        output = str(row["output"]).rstrip()
    elif {"question", "answer"}.issubset(row):
        instruction = str(
            row.get("instruction", _OPENAI_GSM8K_DEFAULT_INSTRUCTION)
        ).strip()
        user_input = str(row["question"]).strip()
        output = str(row["answer"]).rstrip()
    else:
        raise ValueError(
            "Unsupported row schema. Expected either "
            "{instruction,input,output} or {question,answer}."
        )

    text = (
        f"{template.instruction_header}\n"
        f"{instruction}\n\n"
        f"{template.input_header}\n"
        f"{user_input}\n\n"
        f"{template.output_header}\n"
        f"{output}"
    )
    return {"text": text}


def to_text_dataset(
    split: Dataset,
    *,
    template: Optional[PromptTemplate] = None,
) -> Dataset:
    """
    Convert split rows into a plain text field consumed by SFT.
    """
    columns = set(split.column_names)
    has_prolog_schema = {"instruction", "input", "output"}.issubset(columns)
    has_openai_schema = {"question", "answer"}.issubset(columns)
    if not has_prolog_schema and not has_openai_schema:
        raise ValueError(
            "Unsupported split schema. Expected either columns "
            "{instruction,input,output} or {question,answer}."
        )

    resolved_template: PromptTemplate
    if template is not None:
        resolved_template = template
    elif has_prolog_schema:
        resolved_template = PROLOG_PROMPT_TEMPLATE
    else:
        resolved_template = OPENAI_GSM8K_PROMPT_TEMPLATE

    return split.map(
        lambda row: format_record_for_sft(row, template=resolved_template),
        remove_columns=split.column_names,
        desc="Formatting SFT text",
    )


def load_training_splits(
    dataset_dir: Path,
    *,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None
) -> tuple[Dataset, Dataset]:
    """
    Load + format train/eval splits and apply optional row caps.
    """
    ds = load_prepared_dataset(dataset_dir)
    
    if "train" in ds:
        resolved_train_split = "train"
    else:
        raise KeyError(
            "Could not infer train split. Ensure dataset has a 'train' split."
        )
    
    if "val" in ds:
        resolved_eval_split = "val"
    else:
        raise KeyError(
            "Could not infer eval split. Ensure dataset a 'val' split."
        )

    train_ds = cast(Dataset, ds[resolved_train_split])
    eval_ds = cast(Dataset, ds[resolved_eval_split])

    if max_train_samples is not None:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
    if max_eval_samples is not None:
        eval_ds = eval_ds.select(range(min(max_eval_samples, len(eval_ds))))

    resolved_template = _infer_template_from_dataset_dir(dataset_dir)

    train_text = to_text_dataset(train_ds, template=resolved_template)
    eval_text = to_text_dataset(eval_ds, template=resolved_template)
    return train_text, eval_text
