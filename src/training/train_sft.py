from __future__ import annotations
import json

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, cast

from src.training.data import load_training_splits

from src.training.data import load_prepared_dataset
from src.data.prepare_splits import get_default_splits_dir
from src.prolog.execute import normalize_prolog_answer_for_eval

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from datasets import Dataset, DatasetDict
import torch
try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

TRAINING_RESULTS_DIR = Path(__file__).resolve().parent / "training_results"

def load_ground_truth_map(dataset_dir: Path) -> dict[str, str]:
    """
    Load normalized ground-truth answers for any prepared dataset path in this repo.

    Works for:
    - .../gsm8k_proper/ratio_*/
    - .../gsm8k_prolog/
    - .../openai_gsm8k/
    - and nested split directories such as .../gsm8k_prolog/train
    """
    cur = dataset_dir.resolve()
    gt_path: Path | None = None
    for candidate_dir in (cur, *cur.parents):
        candidate = candidate_dir / "ground_truth_by_prompt.json"
        if candidate.exists():
            gt_path = candidate
            break

    if gt_path is None:
        raise FileNotFoundError(
            "Could not locate ground_truth_by_prompt.json from "
            f"path: {dataset_dir}"
        )

    payload = json.loads(gt_path.read_text(encoding="utf-8"))
    return {k.strip(): normalize_prolog_answer_for_eval(v) for k, v in payload["all"].items()}


@dataclass(frozen=True)
class TrainConfig:
    dataset_dir: Path
    model_name_or_path: str
    output_dir: Path = TRAINING_RESULTS_DIR
    train_split: str | None = None
    eval_split: str | None = None
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    seed: int = 42
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 1024
    quantization: str = "8bit"
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    torch_dtype: str = "bfloat16"
    device_map: str | None = "auto"
    dry_run: bool = False


@dataclass(frozen=True)
class RunContext:
    cfg: TrainConfig
    tokenizer: AutoTokenizer
    raw_dataset: DatasetDict


CallbackLike = TrainerCallback | Callable[
    [RunContext], TrainerCallback | Sequence[TrainerCallback]
]


def _normalize_ratio_dir_name(ratio: str) -> str:
    ratio = ratio.strip()
    if not ratio:
        raise ValueError("proper_ratio must not be empty.")
    if ratio.startswith("ratio_"):
        return ratio
    return f"ratio_{ratio}"


def _resolve_dataset_dir(
    *,
    dataset_dir: Path | None,
    splits_dir: Path | None,
    dataset_name: str | None,
    proper_ratio: str,
) -> Path:
    if dataset_dir is not None:
        return dataset_dir

    if dataset_name is None:
        raise ValueError(
            "Provide either --dataset-dir or --dataset-name "
            "(gsm8k_prolog, openai_gsm8k, gsm8k_proper)."
        )

    base_dir = splits_dir if splits_dir is not None else get_default_splits_dir()
    if dataset_name == "gsm8k_proper":
        return base_dir / dataset_name / _normalize_ratio_dir_name(proper_ratio)
    return base_dir / dataset_name


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="SFT scaffold for PROPER/GSM8K-Prolog data."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Direct path to a prepared dataset directory. Overrides dataset-name resolution.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        choices=("gsm8k_prolog", "openai_gsm8k", "gsm8k_proper"),
        help="Dataset under splits base directory.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        help="Base splits version directory. Defaults to prepare_splits.get_default_splits_dir().",
    )
    parser.add_argument(
        "--proper-ratio",
        type=str,
        default="1to2",
        help='Used with --dataset-name gsm8k_proper. Accepts "1to2" or "ratio_1to2".',
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)

    parser.add_argument("--train-split", type=str, default=None)
    parser.add_argument("--eval-split", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument(
        "--quantization",
        type=str,
        default="8bit",
        choices=("none", "8bit", "4bit"),
    )
    parser.add_argument("--use-lora", dest="use_lora", action="store_true")
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.set_defaults(use_lora=True)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,v_proj",
        help='Comma-separated module names, e.g. "q_proj,v_proj".',
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=("auto", "bfloat16", "float16", "float32"),
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help='Model placement strategy, e.g. "auto" or "none".',
    )
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    resolved_device_map: str | None = args.device_map
    if isinstance(resolved_device_map, str) and resolved_device_map.lower() == "none":
        resolved_device_map = None
    resolved_lora_targets = tuple(
        part.strip() for part in str(args.lora_target_modules).split(",") if part.strip()
    )
    if args.use_lora and not resolved_lora_targets:
        raise ValueError("When --use-lora is set, --lora-target-modules must not be empty.")

    resolved_dataset_dir = _resolve_dataset_dir(
        dataset_dir=args.dataset_dir,
        splits_dir=args.splits_dir,
        dataset_name=args.dataset_name,
        proper_ratio=args.proper_ratio,
    )

    return TrainConfig(
        dataset_dir=resolved_dataset_dir,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        train_split=args.train_split,
        eval_split=args.eval_split,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        seed=args.seed,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        quantization=args.quantization,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=resolved_lora_targets,
        torch_dtype=args.torch_dtype,
        device_map=resolved_device_map,
        dry_run=args.dry_run,
    )


def preview_formatted_examples(train_ds: Any, eval_ds: Any, *, n: int = 1) -> None:
    print(f"[data] train rows: {len(train_ds)}")
    print(f"[data] eval rows:  {len(eval_ds)}")
    for i in range(min(n, len(train_ds))):
        print(f"\n[data] train sample #{i}")
        print(train_ds[i]["text"][:800])


def build_tokenizer(cfg: TrainConfig) -> Any:
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return tokenizer


def _resolve_torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(
        f"Unsupported torch dtype '{name}'. Use one of: auto, bfloat16, float16, float32."
    )


def _build_quantization_config(cfg: TrainConfig) -> BitsAndBytesConfig | None:
    if cfg.quantization == "none":
        return None
    if cfg.quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if cfg.quantization == "4bit":
        compute_dtype = (
            torch.bfloat16 if cfg.torch_dtype == "bfloat16" else torch.float16
        )
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    raise ValueError(
        f"Unsupported quantization '{cfg.quantization}'. Use one of: none, 8bit, 4bit."
    )


def build_model(cfg: TrainConfig):
    if cfg.quantization != "none" and not cfg.use_lora:
        raise ValueError(
            "Quantized training in this script requires LoRA. "
            "Set --use-lora or use --quantization none."
        )

    if (cfg.use_lora or cfg.quantization != "none") and (
        LoraConfig is None
        or TaskType is None
        or get_peft_model is None
        or prepare_model_for_kbit_training is None
    ):
        raise ImportError(
            "PEFT is required for LoRA. Install with: pip install peft"
        )

    quant_config = _build_quantization_config(cfg)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": _resolve_torch_dtype(cfg.torch_dtype),
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    if cfg.device_map is not None:
        model_kwargs["device_map"] = cfg.device_map

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        **model_kwargs,
    )

    if quant_config is not None:
        # Required by PEFT for k-bit training preparation.
        model = cast(Callable[..., Any], prepare_model_for_kbit_training)(model)

    if cfg.use_lora:
        assert LoraConfig is not None and TaskType is not None and get_peft_model is not None
        lora_config = cast(Any, LoraConfig)(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            target_modules=list(cfg.lora_target_modules),
        )
        model = cast(Callable[..., Any], get_peft_model)(model, lora_config)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
    return model


def build_trainer(cfg: TrainConfig,
                  *,
                  model: AutoModelForCausalLM,
                  tokenizer: AutoTokenizer,
                  train_ds: Dataset,
                  eval_ds: Dataset,
                  callbacks: Sequence[TrainerCallback] | None = None,
                  eval_strategy: str = "steps",
                  eval_steps: int = 10,
                  save_steps: int = 10,
                  ) -> Trainer:
    
    tokenize = cast(Callable[..., Any], tokenizer)

    train_tok = train_ds.map(
        lambda x: tokenize(x["text"], truncation=True, max_length=cfg.max_seq_length),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    eval_tok = eval_ds.map(
        lambda x: tokenize(x["text"], truncation=True, max_length=cfg.max_seq_length),
        batched=True,
        remove_columns=eval_ds.column_names,
    )

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        save_steps=save_steps,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=cast(Any, tokenizer), mlm=False)

    return Trainer(
        model=cast(Any, model),
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        processing_class=cast(Any, tokenizer),
        data_collator=collator,
        callbacks=list(callbacks) if callbacks else None,
    )


def _resolve_callbacks(
    callbacks_input: Sequence[CallbackLike] | None,
    *,
    context: RunContext,
) -> list[TrainerCallback]:
    if not callbacks_input:
        return []

    callbacks: list[TrainerCallback] = []
    for callback in callbacks_input:
        if isinstance(callback, TrainerCallback):
            callbacks.append(callback)
            continue

        produced = callback(context)
        if isinstance(produced, TrainerCallback):
            callbacks.append(produced)
            continue
        if produced is None:
            continue

        for cb in produced:
            if not isinstance(cb, TrainerCallback):
                raise TypeError(
                    "Callback callable must return TrainerCallback or "
                    "a sequence of TrainerCallback instances."
                )
            callbacks.append(cb)
    return callbacks


def run(
    cfg: TrainConfig,
    *,
    callbacks: Sequence[CallbackLike] | None = None,
) -> None:
    """
    Execute SFT training with optional pluggable callbacks.

    `callbacks` accepts either:
    - a `TrainerCallback` instance
    - a factory callable taking `RunContext` and returning one or many callbacks
    """
    raw_ds = load_prepared_dataset(cfg.dataset_dir)
    train_ds, eval_ds = load_training_splits(
        cfg.dataset_dir,
        max_train_samples=cfg.max_train_samples,
        max_eval_samples=cfg.max_eval_samples,
    )
    preview_formatted_examples(train_ds, eval_ds, n=1)

    if cfg.dry_run:
        print("\n[dry-run] stopping before tokenizer/model/trainer setup.")
        return

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(cfg)
    model = cast(AutoModelForCausalLM, build_model(cfg))
    context = RunContext(cfg=cfg, tokenizer=tokenizer, raw_dataset=raw_ds)
    callbacks = _resolve_callbacks(callbacks, context=context)
    trainer = build_trainer(
        cfg,
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        eval_ds=eval_ds,
        callbacks=callbacks,
    )

    train_result = trainer.train()
    trainer.save_model(str(cfg.output_dir))
    trainer.save_state()

    train_metrics = dict(train_result.metrics)
    train_metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(eval_ds)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
