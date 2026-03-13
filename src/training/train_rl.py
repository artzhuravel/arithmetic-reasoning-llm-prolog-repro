from __future__ import annotations

import argparse
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
import re

from datasets import DatasetDict
from transformers import AutoTokenizer
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
from src.prolog.execute import execute_solve
from src.training.callbacks import PrologAccuracyCallback

from src.training.data import (
    load_prepared_dataset,
    load_training_splits,
    preview_formatted_examples,
    load_ground_truth_map,
    resolve_eval_rows,
    resolve_prompt_template,
)
from src.training.helpers import (
    _resolve_dataset_dir,
    _resolve_hf_token_from_cfg,
    build_model,
    build_tokenizer,
)

TRAINING_RESULTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "training"
LOGGER = logging.getLogger(__name__)
_PROMPT_INPUT_RE = re.compile(r"### Input\n(.*?)\n\n### Output\n", re.DOTALL)

# Reward scaffold for Prolog execution outcomes. Tune these after a few runs.
_REWARD_CORRECT = 1.0
_REWARD_EXECUTABLE_WRONG = 0.1
_REWARD_NO_SOLUTION = -0.2
_REWARD_SYNTAX_ERROR = -0.4
_REWARD_EXECUTION_ERROR = -0.5
_REWARD_TIMEOUT = -0.75
_REWARD_EMPTY_COMPLETION = -1.0


def _configure_runtime_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"`get_control_token` is deprecated\. Use `get_special_token` instead\.",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`_control_tokens` is deprecated\..*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Could not estimate the number of tokens of the input, floating-point operations will not be computed",
        category=UserWarning,
    )


@dataclass(frozen=True)
class RLTrainConfig:
    dataset_dir: Path
    base_model_name_or_path: str
    sft_adapter_dir: Path
    output_dir: Path = TRAINING_RESULTS_DIR
    seed: int = 42
    dry_run: bool = False
    max_prompt_length: int = 1024
    max_completion_length: int = 1024
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    num_generations: int = 4
    temperature: float = 0.7
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 20
    per_device_eval_batch_size: int = 20
    validation_max_samples: int = 100
    validation_workers: int = 10
    validation_generation_num_beams: int = 4
    validation_generation_max_new_tokens: int = 256
    reward: str = "prolog_shaped_reward"
    reward_workers: int = 10
    vllm_gpu_memory_utilization: float = 0.7
    torch_dtype: str = "bfloat16"
    device_map: str | None = "auto"
    hf_token: str | None = None

    @property
    def model_name_or_path(self) -> str:
        return self.base_model_name_or_path


def _count_trainable_parameters(model) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _safe_output_component(value: str) -> str:
    cleaned = "".join(
        ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_"
        for ch in value.strip().lower()
    )
    return cleaned or "run"


def _infer_dataset_slug(
    *,
    dataset_name: str | None,
    dataset_dir: Path,
    proper_ratio: str | None,
) -> str:
    if dataset_name == "gsm8k_prolog":
        return "gsm8k_prolog"
    if dataset_name == "openai_gsm8k":
        return "openai_gsm8k"
    if dataset_name == "gsm8k_proper":
        if proper_ratio is None or not proper_ratio.strip():
            raise ValueError("For gsm8k_proper, --proper-ratio is required.")
        return f"gsm8k_proper_{_safe_output_component(proper_ratio.strip())}"

    lowered_parts = [part.lower() for part in dataset_dir.parts]
    if "gsm8k_prolog" in lowered_parts:
        return "gsm8k_prolog"
    if "openai_gsm8k" in lowered_parts:
        return "openai_gsm8k"
    if "gsm8k_proper" in lowered_parts:
        ratio_component = next(
            (part for part in dataset_dir.parts if part.lower().startswith("ratio_")),
            "ratio_unknown",
        )
        return f"gsm8k_proper_{_safe_output_component(ratio_component)}"

    return _safe_output_component(dataset_dir.name)


def _resolve_output_dir(
    *,
    output_dir: Path | None,
    dataset_name: str | None,
    dataset_dir: Path,
    proper_ratio: str | None,
) -> Path:
    if output_dir is not None:
        return output_dir

    dataset_slug = _infer_dataset_slug(
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        proper_ratio=proper_ratio,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = TRAINING_RESULTS_DIR / f"{dataset_slug}_{timestamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = TRAINING_RESULTS_DIR / f"{dataset_slug}_{timestamp}_{suffix}"
        suffix += 1
    return candidate


def _resolve_steps_per_generation(cfg: RLTrainConfig) -> int:
    return cfg.gradient_accumulation_steps


def _validate_grpo_config(cfg: RLTrainConfig) -> int:
    if cfg.per_device_train_batch_size < 1:
        raise ValueError("per_device_train_batch_size must be >= 1.")
    if cfg.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1.")
    if cfg.num_generations < 2:
        raise ValueError("num_generations must be >= 2 for GRPO.")
    if cfg.eval_strategy not in {"no", "steps", "epoch"}:
        raise ValueError("eval_strategy must be one of: no, steps, epoch.")
    if cfg.eval_strategy == "steps" and cfg.eval_steps < 1:
        raise ValueError("eval_steps must be >= 1 when eval_strategy='steps'.")
    if cfg.per_device_eval_batch_size < 1:
        raise ValueError("per_device_eval_batch_size must be >= 1.")
    if cfg.validation_max_samples < 1:
        raise ValueError("validation_max_samples must be >= 1.")
    if cfg.validation_workers < 1:
        raise ValueError("validation_workers must be >= 1.")
    if cfg.validation_generation_num_beams < 1:
        raise ValueError("validation_generation_num_beams must be >= 1.")
    if cfg.validation_generation_max_new_tokens < 1:
        raise ValueError("validation_generation_max_new_tokens must be >= 1.")
    if cfg.reward_workers < 1:
        raise ValueError("reward_workers must be >= 1.")
    if not (0.0 < cfg.vllm_gpu_memory_utilization < 1.0):
        raise ValueError("vllm_gpu_memory_utilization must be between 0 and 1.")

    steps_per_generation = _resolve_steps_per_generation(cfg)
    generation_batch_size = cfg.per_device_train_batch_size * steps_per_generation
    if generation_batch_size % cfg.num_generations != 0:
        raise ValueError(
            "Invalid GRPO batch geometry for single-GPU training: "
            f"per_device_train_batch_size={cfg.per_device_train_batch_size}, "
            f"num_generations={cfg.num_generations}, "
            f"steps_per_generation={steps_per_generation}, "
            f"generation_batch_size={generation_batch_size}. "
            "generation_batch_size must be divisible by num_generations."
        )

    return steps_per_generation


def compute_single_prolog_shaped_reward(completion: str, expected_answer: str) -> float:
    """
    Scaffold reward for one generated Prolog program.

    Current policy:
    - executable and correct: highest reward
    - executable but wrong answer: small positive reward
    - no solution / syntax / runtime / timeout: increasingly negative rewards
    - dependency errors fail fast because the environment is unusable for RL
    """
    if not completion.strip():
        return _REWARD_EMPTY_COMPLETION

    result = execute_solve(completion)

    if result.ok:
        if result.normalized_answer == expected_answer:
            return _REWARD_CORRECT
        return _REWARD_EXECUTABLE_WRONG

    error_type = result.error_type or "execution_error"
    if error_type == "no_solution":
        return _REWARD_NO_SOLUTION
    if error_type == "syntax_error":
        return _REWARD_SYNTAX_ERROR
    if error_type == "timeout":
        return _REWARD_TIMEOUT
    if error_type == "dependency_error":
        raise RuntimeError(
            f"SWI-Prolog dependency error while scoring completion: {result.error}"
        )
    if error_type == "execution_error":
        return _REWARD_EXECUTION_ERROR

    # Unknown executor failure type.
    LOGGER.warning("Unhandled Prolog error_type=%r. Falling back to execution penalty.", error_type)
    return _REWARD_EXECUTION_ERROR


def _extract_prompt_key(prompt: str) -> str:
    match = _PROMPT_INPUT_RE.search(prompt)
    if match is None:
        raise ValueError("Prompt does not match expected PROLOG prompt template.")
    return match.group(1).strip()


def _resolve_expected_answers(
    *,
    prompts: list[str],
    gt_map: dict[str, str] | None,
    expected_answer: list[str] | None,
    prompt_key: list[str] | None,
) -> list[str]:
    if expected_answer is not None:
        return [str(answer) for answer in expected_answer]

    if gt_map is None:
        raise ValueError("Either expected_answer or gt_map must be provided to the reward function.")

    if prompt_key is not None:
        return [str(gt_map[str(key)]) for key in prompt_key]

    return [str(gt_map[_extract_prompt_key(prompt)]) for prompt in prompts]

def prolog_shaped_reward(*,
                         prompts: list[str],
                         completions: list[str],
                         gt_map: dict[str, str] | None = None,
                         expected_answer: list[str] | None = None,
                         prompt_key: list[str] | None = None,
                         workers: int = 10,
                         **kwargs) -> list[float]:
    """
    TRL calls this with batched prompts/completions and forwards extra dataset columns.
    Prefer using `expected_answer` from the dataset; `gt_map` + prompt parsing is fallback-only.
    """
    expected_answers = _resolve_expected_answers(
        prompts=prompts,
        gt_map=gt_map,
        expected_answer=expected_answer,
        prompt_key=prompt_key,
    )

    if len(completions) != len(expected_answers):
        raise ValueError(
            f"Completions/expected answer size mismatch: {len(completions)} vs {len(expected_answers)}"
        )

    if not completions:
        return []

    workers = min(workers, len(completions))
    if workers == 1:
        return [
            compute_single_prolog_shaped_reward(completion, target)
            for completion, target in zip(completions, expected_answers)
        ]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            executor.map(
                compute_single_prolog_shaped_reward,
                completions,
                expected_answers,
            )
        )
    


def _resolve_reward_function(reward: str,
                             workers: int = 10,
                             gt_map: dict[str, str] | None = None) -> Callable[..., list[float]]:
    if reward == "prolog_shaped_reward":
        return lambda *args, **kwargs: prolog_shaped_reward(workers=workers, gt_map=gt_map, *args, **kwargs)
    else:
        raise ValueError(f"Unsupported reward function: {reward}")


def _build_validation_callback(
    *,
    cfg: RLTrainConfig,
    tokenizer: AutoTokenizer,
    raw_ds: DatasetDict,
    gt_map: dict[str, str],
) -> PrologAccuracyCallback:
    eval_rows = resolve_eval_rows(raw_ds)
    if cfg.max_eval_samples is not None:
        eval_rows = eval_rows.select(range(min(cfg.max_eval_samples, len(eval_rows))))
    template = resolve_prompt_template(cfg.dataset_dir, eval_rows)
    return PrologAccuracyCallback(
        tokenizer=tokenizer,
        eval_rows=eval_rows,
        gt_map=gt_map,
        template=template,
        max_samples=cfg.validation_max_samples,
        eval_every_steps=cfg.eval_steps,
        eval_strategy=cfg.eval_strategy,
        workers=cfg.validation_workers,
        generation_batch_size=cfg.per_device_eval_batch_size,
        generation_num_beams=cfg.validation_generation_num_beams,
        generation_max_new_tokens=cfg.validation_generation_max_new_tokens,
        prompt_max_length=cfg.max_prompt_length,
    )




def run(cfg: RLTrainConfig) -> None:
    _configure_runtime_warning_filters()
    raw_ds = load_prepared_dataset(cfg.dataset_dir)
    train_ds, eval_ds = load_training_splits(
        cfg.dataset_dir,
        mode="rl",
        max_train_samples=cfg.max_train_samples,
        max_eval_samples=cfg.max_eval_samples,
    )
    preview_formatted_examples(train_ds, eval_ds, n=1)

    if cfg.dry_run:
        print("\n[dry-run] stopping before tokenizer/model/trainer setup.")
        return

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Training outputs directory: %s", cfg.output_dir)
    resolved_hf_token = _resolve_hf_token_from_cfg(cfg)
    if resolved_hf_token is None:
        LOGGER.warning(
            "HF token not detected. Downloads will be unauthenticated and may be rate-limited."
        )
    else:
        LOGGER.info("HF token detected. Using authenticated Hugging Face Hub requests.")

    steps_per_generation = _validate_grpo_config(cfg)

    tokenizer = build_tokenizer(
        model_name_or_path=cfg.model_name_or_path,
        hf_token=resolved_hf_token,
        padding="left",
    )
    tokenizer.truncation_side = "left"
    model = build_model(
        model_name_or_path=cfg.model_name_or_path,
        torch_dtype=cfg.torch_dtype,
        quantization="none",
        device_map=cfg.device_map,
        hf_token=resolved_hf_token,
        attach_adapter=True,
        adapter_dir=cfg.sft_adapter_dir,
        adapter_trainable=True,
    )
    trainable_params = _count_trainable_parameters(model)
    if trainable_params <= 0:
        raise RuntimeError(
            "Loaded RL model has no trainable parameters. "
            "Check adapter loading and PEFT trainability configuration."
        )
    LOGGER.info("RL model trainable parameters: %d", trainable_params)
    
    gt_map = load_ground_truth_map(cfg.dataset_dir)
    reward_function = _resolve_reward_function(cfg.reward, workers=cfg.reward_workers, gt_map=gt_map)
    validation_callback = _build_validation_callback(
        cfg=cfg,
        tokenizer=tokenizer,
        raw_ds=raw_ds,
        gt_map=gt_map,
    )
    save_strategy = cfg.eval_strategy if cfg.eval_strategy in {"steps", "epoch"} else "steps"
    load_best_model_at_end = cfg.eval_strategy != "no"
    
    # pyright: ignore[reportCallIssue]
    training_args = GRPOConfig(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_train_epochs,
        num_generations=cfg.num_generations,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps if cfg.eval_strategy == "steps" else None,
        save_strategy=save_strategy,
        save_steps=cfg.eval_steps if save_strategy == "steps" else 200,
        save_total_limit=2,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=(
            "eval_prolog_answer_accuracy" if load_best_model_at_end else None
        ),
        greater_is_better=True if load_best_model_at_end else None,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        seed=cfg.seed,
        bf16=(cfg.torch_dtype == "bfloat16"),
        fp16=(cfg.torch_dtype == "float16"),
        max_prompt_length=cfg.max_prompt_length,  # pyright: ignore[reportCallIssue]
        max_completion_length=cfg.max_completion_length,  # pyright: ignore[reportCallIssue]
        temperature=cfg.temperature,
        steps_per_generation=steps_per_generation,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=1,
        vllm_enable_sleep_mode=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        reward_funcs=reward_function,
        callbacks=[validation_callback],
    )
    
    train_result = trainer.train()
    if trainer.state.best_model_checkpoint:
        LOGGER.info(
            "Best checkpoint by eval_prolog_answer_accuracy: %s",
            trainer.state.best_model_checkpoint,
        )
    trainer.save_model(str(cfg.output_dir))
    trainer.save_state()

    train_metrics = dict(train_result.metrics)
    train_metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    if cfg.eval_strategy != "no":
        eval_metrics = trainer.evaluate()
        eval_metrics["eval_samples"] = len(eval_ds)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)


def parse_args() -> RLTrainConfig:
    parser = argparse.ArgumentParser(
        description="RL training for PROPER/GSM8K-Prolog data."
    )
    parser.add_argument("--dataset-dir", type=Path, required=False)
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=False,
        choices=("gsm8k_prolog", "openai_gsm8k", "gsm8k_proper"),
    )
    parser.add_argument("--proper-ratio", type=str, required=False)
    parser.add_argument("--splits-dir", type=Path, required=False)
    parser.add_argument("--base-model-name-or-path", type=str, required=True)
    parser.add_argument("--sft-adapter-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Run output directory. If omitted, defaults to "
            "outputs/training/<dataset_or_ratio>_<timestamp>."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--max-train-samples", type=int, required=False)
    parser.add_argument("--max-eval-samples", type=int, required=False)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--eval-strategy",
        type=str,
        default="steps",
        choices=("no", "steps", "epoch"),
    )
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=20)
    parser.add_argument("--validation-max-samples", type=int, default=100)
    parser.add_argument("--validation-workers", type=int, default=10)
    parser.add_argument("--validation-generation-num-beams", type=int, default=4)
    parser.add_argument("--validation-generation-max-new-tokens", type=int, default=256)
    parser.add_argument("--reward-workers", type=int, default=10)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--hf-token", type=str, required=False)

    args = parser.parse_args()

    resolved_device_map: str | None = args.device_map
    if isinstance(resolved_device_map, str) and resolved_device_map.lower() == "none":
        resolved_device_map = None

    resolved_dataset_dir = _resolve_dataset_dir(
        dataset_dir=args.dataset_dir,
        splits_dir=args.splits_dir,
        dataset_name=args.dataset_name,
        proper_ratio=args.proper_ratio,
    )
    resolved_output_dir = _resolve_output_dir(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        dataset_dir=resolved_dataset_dir,
        proper_ratio=args.proper_ratio,
    )

    return RLTrainConfig(
        dataset_dir=resolved_dataset_dir,
        base_model_name_or_path=args.base_model_name_or_path,
        sft_adapter_dir=args.sft_adapter_dir,
        output_dir=resolved_output_dir,
        seed=args.seed,
        dry_run=args.dry_run,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        num_generations=args.num_generations,
        temperature=args.temperature,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        validation_max_samples=args.validation_max_samples,
        validation_workers=args.validation_workers,
        validation_generation_num_beams=args.validation_generation_num_beams,
        validation_generation_max_new_tokens=args.validation_generation_max_new_tokens,
        reward_workers=args.reward_workers,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        torch_dtype=args.torch_dtype,
        device_map=resolved_device_map,
        hf_token=args.hf_token,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
