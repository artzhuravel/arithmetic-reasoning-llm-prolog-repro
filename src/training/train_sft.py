from __future__ import annotations
import json

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

from src.training.data import load_training_splits

from src.training.data import load_prepared_dataset
from src.prolog.execute import normalize_prolog_answer_for_eval

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import torch

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
    dry_run: bool = False


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="SFT scaffold for PROPER/GSM8K-Prolog data."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
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
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    return TrainConfig(
        dataset_dir=args.dataset_dir,
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


def build_model(cfg: TrainConfig):
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=torch.bfloat16,   # or torch.float16
        device_map="auto",
    )
    return model


def build_trainer(cfg: TrainConfig,
                  *,
                  model: AutoModelForCausalLM,
                  tokenizer: AutoTokenizer,
                  train_ds: Dataset,
                  eval_ds: Dataset,
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
    )


def run(cfg: TrainConfig) -> None:
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
    trainer = build_trainer(
        cfg,
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        eval_ds=eval_ds,
    )
    
    #TODO
    ...


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
