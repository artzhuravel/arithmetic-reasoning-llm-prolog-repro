from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Mapping, Optional, cast

from datasets import Dataset, DatasetDict, load_from_disk

_OPENAI_GSM8K_DEFAULT_INSTRUCTION = "Please solve the given math problem."
LOGGER = logging.getLogger(__name__)


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


def _resolve_template_from_columns(columns: set[str]) -> PromptTemplate:
    has_prolog_schema = {"instruction", "input", "output"}.issubset(columns)
    has_openai_schema = {"question", "answer"}.issubset(columns)
    if has_prolog_schema:
        return PROLOG_PROMPT_TEMPLATE
    if has_openai_schema:
        return OPENAI_GSM8K_PROMPT_TEMPLATE
    raise ValueError(
        "Unsupported split schema. Expected either columns "
        "{instruction,input,output} or {question,answer}."
    )


def resolve_prompt_template(dataset_dir: Path, split: Dataset) -> PromptTemplate:
    """
    Resolve prompt template using the same policy as training text formatting.
    """
    inferred = _infer_template_from_dataset_dir(dataset_dir)
    if inferred is not None:
        return inferred
    return _resolve_template_from_columns(set(split.column_names))


def _extract_prompt_fields(row: Mapping[str, Any]) -> tuple[str, str, str]:
    if {"instruction", "input", "output"}.issubset(row):
        instruction = str(row["instruction"]).strip()
        user_input = str(row["input"]).strip()
        output = str(row["output"]).rstrip()
        return instruction, user_input, output
    if {"question", "answer"}.issubset(row):
        instruction = str(
            row.get("instruction", _OPENAI_GSM8K_DEFAULT_INSTRUCTION)
        ).strip()
        user_input = str(row["question"]).strip()
        output = str(row["answer"]).rstrip()
        return instruction, user_input, output
    raise ValueError(
        "Unsupported row schema. Expected either "
        "{instruction,input,output} or {question,answer}."
    )


def build_prompt_text(
    row: Mapping[str, Any],
    *,
    template: PromptTemplate,
    include_output: bool = True,
) -> str:
    """
    Build text with the shared template used for both SFT formatting and eval prompts.
    """
    instruction, user_input, output = _extract_prompt_fields(row)
    prefix = (
        f"{template.instruction_header}\n"
        f"{instruction}\n\n"
        f"{template.input_header}\n"
        f"{user_input}\n\n"
        f"{template.output_header}\n"
    )
    if include_output:
        return prefix + output
    return prefix


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
    return {"text": build_prompt_text(row, template=template, include_output=True)}


def to_text_dataset(
    split: Dataset,
    *,
    template: Optional[PromptTemplate] = None,
) -> Dataset:
    """
    Convert split rows into a plain text field consumed by SFT.
    """
    columns = set(split.column_names)
    _resolve_template_from_columns(columns)

    resolved_template: PromptTemplate
    if template is not None:
        resolved_template = template
    else:
        resolved_template = _resolve_template_from_columns(columns)

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
    
    if "train" not in ds:
        available = ", ".join(str(k) for k in ds.keys())
        raise KeyError(
            "Could not infer train split. Strict policy requires a 'train' split. "
            f"Available splits: [{available}]"
        )
    resolved_train_split = "train"

    if "val" not in ds:
        available = ", ".join(str(k) for k in ds.keys())
        raise KeyError(
            "Could not infer eval split. Strict policy requires a 'val' split. "
            f"Available splits: [{available}]"
        )
    resolved_eval_split = "val"

    LOGGER.info(
        "Strict split policy active: using train='%s' and eval='%s'.",
        resolved_train_split,
        resolved_eval_split,
    )

    train_ds = cast(Dataset, ds[resolved_train_split])
    eval_ds = cast(Dataset, ds[resolved_eval_split])

    if max_train_samples is not None:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
    if max_eval_samples is not None:
        eval_ds = eval_ds.select(range(min(max_eval_samples, len(eval_ds))))

    resolved_template = resolve_prompt_template(dataset_dir, train_ds)
    LOGGER.info(
        "Resolved prompt template for dataset '%s': instruction='%s', input='%s', output='%s'.",
        dataset_dir,
        resolved_template.instruction_header,
        resolved_template.input_header,
        resolved_template.output_header,
    )

    train_text = to_text_dataset(train_ds, template=resolved_template)
    eval_text = to_text_dataset(eval_ds, template=resolved_template)
    return train_text, eval_text
