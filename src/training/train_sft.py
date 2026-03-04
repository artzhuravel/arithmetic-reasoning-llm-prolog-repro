from __future__ import annotations
import json
import logging

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence, cast

from src.training.data import (
    load_prepared_dataset,
    load_training_splits,
    resolve_prompt_template,
)
from src.data.prepare_splits import get_default_splits_dir
from src.training.callbacks import PrologAccuracyCallback
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
    from peft import (
        LoraConfig as PeftLoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except ImportError:
    PeftLoraConfig = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_RESULTS_DIR = REPO_ROOT / "outputs" / "training"
LOGGER = logging.getLogger(__name__)

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
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    seed: int = 42
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 1024
    enable_custom_callbacks: bool = True
    custom_callbacks_max_samples: int = 100
    custom_callbacks_eval_every_steps: int = 50
    torch_dtype: str = "bfloat16"
    device_map: str | None = "auto"
    dry_run: bool = False


@dataclass(frozen=True)
class RunContext:
    cfg: TrainConfig
    tokenizer: AutoTokenizer
    raw_dataset: DatasetDict


@dataclass(frozen=True)
class LoraStrategyConfig:
    quantization: str = "8bit"
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")


class ModelBuildStrategy(Protocol):
    def build_model(self, cfg: TrainConfig) -> Any:
        ...


@dataclass(frozen=True)
class FullFineTuneStrategy:
    quantization: str = "none"

    def build_model(self, cfg: TrainConfig) -> Any:
        if self.quantization != "none":
            raise ValueError(
                "Full fine-tuning strategy currently supports only --quantization none."
            )

        model_kwargs: dict[str, Any] = {
            "torch_dtype": _resolve_torch_dtype(cfg.torch_dtype),
        }
        if cfg.device_map is not None:
            model_kwargs["device_map"] = cfg.device_map
        return AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            **model_kwargs,
        )


@dataclass(frozen=True)
class LoraFineTuneStrategy:
    lora: LoraStrategyConfig

    def build_model(self, cfg: TrainConfig) -> Any:
        if (
            PeftLoraConfig is None
            or TaskType is None
            or get_peft_model is None
            or prepare_model_for_kbit_training is None
        ):
            raise ImportError(
                "PEFT is required for LoRA. Install with: pip install peft"
            )

        quant_config = _build_quantization_config(
            quantization=self.lora.quantization,
            torch_dtype=cfg.torch_dtype,
        )
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
            model = cast(Callable[..., Any], prepare_model_for_kbit_training)(model)

        lora_config = cast(Any, PeftLoraConfig)(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora.r,
            lora_alpha=self.lora.alpha,
            lora_dropout=self.lora.dropout,
            bias="none",
            target_modules=list(self.lora.target_modules),
        )
        model = cast(Callable[..., Any], get_peft_model)(model, lora_config)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
        return model


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


def parse_args() -> tuple[TrainConfig, ModelBuildStrategy]:
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
    parser.add_argument("--output-dir", type=Path, default=TRAINING_RESULTS_DIR)
    parser.add_argument("--model-name-or-path", type=str, required=True)

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
        "--training-strategy",
        type=str,
        default="lora",
        choices=("lora", "full"),
        help='Training pipeline type: "lora" (default) or "full" fine-tuning.',
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=("none", "8bit", "4bit"),
    )
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
        "--enable-custom-callbacks",
        dest="enable_custom_callbacks",
        action="store_true",
    )
    parser.add_argument(
        "--disable-custom-callbacks",
        dest="enable_custom_callbacks",
        action="store_false",
    )
    parser.set_defaults(enable_custom_callbacks=True)
    parser.add_argument("--custom-callbacks-max-samples", type=int, default=100)
    parser.add_argument("--custom-callbacks-eval-every-steps", type=int, default=50)
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
    resolved_quantization = (
        str(args.quantization)
        if args.quantization is not None
        else ("8bit" if args.training_strategy == "lora" else "none")
    )
    if args.training_strategy == "lora" and not resolved_lora_targets:
        raise ValueError(
            'When "--training-strategy lora" is used, --lora-target-modules must not be empty.'
        )
    if args.training_strategy == "full" and resolved_quantization != "none":
        raise ValueError(
            'Full fine-tuning currently requires "--quantization none".'
        )

    resolved_dataset_dir = _resolve_dataset_dir(
        dataset_dir=args.dataset_dir,
        splits_dir=args.splits_dir,
        dataset_name=args.dataset_name,
        proper_ratio=args.proper_ratio,
    )

    cfg = TrainConfig(
        dataset_dir=resolved_dataset_dir,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        seed=args.seed,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        enable_custom_callbacks=args.enable_custom_callbacks,
        custom_callbacks_max_samples=args.custom_callbacks_max_samples,
        custom_callbacks_eval_every_steps=args.custom_callbacks_eval_every_steps,
        torch_dtype=args.torch_dtype,
        device_map=resolved_device_map,
        dry_run=args.dry_run,
    )
    strategy: ModelBuildStrategy
    if args.training_strategy == "lora":
        strategy = LoraFineTuneStrategy(
            lora=LoraStrategyConfig(
                quantization=resolved_quantization,
                r=args.lora_r,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                target_modules=resolved_lora_targets,
            )
        )
    else:
        strategy = FullFineTuneStrategy(quantization=resolved_quantization)
    return cfg, strategy


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


def _build_quantization_config(
    *,
    quantization: str,
    torch_dtype: str,
) -> BitsAndBytesConfig | None:
    if quantization == "none":
        return None
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantization == "4bit":
        compute_dtype = (
            torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16
        )
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    raise ValueError(
        f"Unsupported quantization '{quantization}'. Use one of: none, 8bit, 4bit."
    )


def build_trainer(cfg: TrainConfig,
                  *,
                  model: AutoModelForCausalLM,
                  tokenizer: AutoTokenizer,
                  train_ds: Dataset,
                  eval_ds: Dataset,
                  callbacks: Sequence[TrainerCallback] | None = None,
                  eval_strategy: str = "steps",
                  eval_steps: int = 10,
                  save_steps: int = 200,
                  save_strategy: str = "steps",
                  save_total_limit: int = 2,
                  load_best_model_at_end: bool = True,
                  metric_for_best_model: str = "eval_loss",
                  greater_is_better: bool = False,
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
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
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


def _is_prolog_like_dataset(dataset_dir: Path) -> bool:
    for component in dataset_dir.parts:
        name = component.lower()
        if "gsm8k_prolog" in name or "gsm8k_proper" in name:
            return True
    return False


def _resolve_eval_rows(raw_ds: DatasetDict) -> Dataset:
    if "val" in raw_ds:
        LOGGER.info("Strict split policy active for callback rows: using eval split 'val'.")
        return cast(Dataset, raw_ds["val"])
    available = ", ".join(str(k) for k in raw_ds.keys())
    raise KeyError(
        "Could not infer eval split for callback rows. Strict policy requires 'val'. "
        f"Available splits: [{available}]"
    )


def _resolve_default_callbacks(context: RunContext) -> list[TrainerCallback]:
    cfg = context.cfg
    if not cfg.enable_custom_callbacks:
        LOGGER.info("Default custom callbacks disabled by config.")
        return []
    if not _is_prolog_like_dataset(cfg.dataset_dir):
        LOGGER.info(
            "Default PrologAccuracyCallback not attached: dataset is not prolog-like (%s).",
            cfg.dataset_dir,
        )
        return []

    try:
        eval_rows = _resolve_eval_rows(context.raw_dataset)
        template = resolve_prompt_template(cfg.dataset_dir, eval_rows)
        gt_map = load_ground_truth_map(cfg.dataset_dir)
    except Exception as e:
        LOGGER.warning("Could not attach default PrologAccuracyCallback: %s", e)
        return []

    LOGGER.info(
        "Attached default PrologAccuracyCallback (max_samples=%d, eval_every_steps=%d).",
        cfg.custom_callbacks_max_samples,
        cfg.custom_callbacks_eval_every_steps,
    )
    return [
        PrologAccuracyCallback(
            tokenizer=context.tokenizer,
            eval_rows=eval_rows,
            gt_map=gt_map,
            template=template,
            max_samples=cfg.custom_callbacks_max_samples,
            eval_every_steps=cfg.custom_callbacks_eval_every_steps,
        )
    ]


def run(
    cfg: TrainConfig,
    *,
    strategy: ModelBuildStrategy,
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
    model = cast(AutoModelForCausalLM, strategy.build_model(cfg))
    context = RunContext(cfg=cfg, tokenizer=tokenizer, raw_dataset=raw_ds)
    user_callbacks = _resolve_callbacks(callbacks, context=context)
    resolved_callbacks: list[TrainerCallback] = list(user_callbacks)
    if not any(isinstance(cb, PrologAccuracyCallback) for cb in resolved_callbacks):
        resolved_callbacks.extend(_resolve_default_callbacks(context))
    trainer = build_trainer(
        cfg,
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        eval_ds=eval_ds,
        callbacks=resolved_callbacks,
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    cfg, strategy = parse_args()
    run(cfg, strategy=strategy)


if __name__ == "__main__":
    main()
