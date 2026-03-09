from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Optional, cast

from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

from src.data.prepare_splits import get_default_splits_dir
from src.prolog.execute import normalize_prolog_answer_for_eval
from src.training.callbacks import score_predicted_prolog_batch, tqdm
from src.training.data import PromptTemplate, build_prompt_text, load_prepared_dataset, resolve_prompt_template


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_RESULTS_DIR = REPO_ROOT / "outputs" / "eval"


@dataclass(frozen=True)
class TestSuiteSpec:
    dataset_name: str
    split_name: str
    use_proper_ratio: bool = False


@dataclass(frozen=True)
class EvalConfig:
    model_mode: str
    base_model_name_or_path: str | None
    adapter_dir: Path | None
    merged_model_dir: Path | None
    test_suite: str
    splits_dir: Path
    proper_ratio: str
    output_dir: Path
    save_trace: bool
    max_samples: int | None
    generation_batch_size: int
    generation_num_beams: int
    generation_max_new_tokens: int
    workers: int
    torch_dtype: str
    device_map: str | None
    hf_token: str | None


TEST_SUITES: dict[str, TestSuiteSpec] = {
    "gsm8k_prolog_val": TestSuiteSpec(dataset_name="gsm8k_prolog", split_name="val"),
    "gsm8k_prolog_test": TestSuiteSpec(dataset_name="gsm8k_prolog", split_name="test"),
    "gsm8k_proper_val": TestSuiteSpec(
        dataset_name="gsm8k_proper", split_name="val", use_proper_ratio=True
    ),
    "gsm8k_proper_test": TestSuiteSpec(
        dataset_name="gsm8k_proper", split_name="test", use_proper_ratio=True
    ),
    "openai_gsm8k_val": TestSuiteSpec(dataset_name="openai_gsm8k", split_name="val"),
    "openai_gsm8k_test": TestSuiteSpec(dataset_name="openai_gsm8k", split_name="test"),
}


def _normalize_ratio_dir_name(ratio: str) -> str:
    clean = ratio.strip()
    if not clean:
        raise ValueError("proper_ratio must not be empty.")
    if clean.startswith("ratio_"):
        return clean
    return f"ratio_{clean}"


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


def _resolve_hf_token(explicit_token: str | None) -> str | None:
    if explicit_token is not None and explicit_token.strip():
        return explicit_token.strip()

    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.getenv(env_name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _resolve_test_dataset_dir(
    *,
    splits_dir: Path,
    test_suite: str,
    proper_ratio: str,
) -> tuple[Path, str]:
    if test_suite not in TEST_SUITES:
        choices = ", ".join(sorted(TEST_SUITES.keys()))
        raise ValueError(f"Unknown test suite: {test_suite}. Choices: {choices}")

    spec = TEST_SUITES[test_suite]
    if spec.use_proper_ratio:
        dataset_dir = (
            splits_dir
            / spec.dataset_name
            / _normalize_ratio_dir_name(proper_ratio)
        )
    else:
        dataset_dir = splits_dir / spec.dataset_name
    return dataset_dir, spec.split_name


def _load_ground_truth_map(dataset_dir: Path) -> dict[str, str]:
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
    raw_map = payload.get("all")
    if not isinstance(raw_map, dict):
        raise ValueError(
            f"Invalid ground truth format in {gt_path}: expected top-level 'all' dict."
        )
    return {
        str(k).strip(): normalize_prolog_answer_for_eval(v)
        for k, v in raw_map.items()
    }


def _resolve_prompt_key(row: Mapping[str, Any]) -> str:
    if "input" in row:
        return str(row["input"]).strip()
    if "question" in row:
        return str(row["question"]).strip()
    raise ValueError("Unsupported row schema: expected either 'input' or 'question'.")


def _resolve_base_model_from_adapter(adapter_dir: Path) -> str:
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(
            f"Could not infer base model: missing {adapter_config_path}"
        )

    payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    candidate = payload.get("base_model_name_or_path")
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError(
            "Could not resolve base model from adapter_config.json. "
            "Pass --base-model-name-or-path explicitly."
        )
    return candidate.strip()


def _build_output_dir(
    *,
    output_dir: Path | None,
    test_suite: str,
    model_mode: str,
) -> Path:
    if output_dir is not None:
        return output_dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return EVAL_RESULTS_DIR / f"{test_suite}_{model_mode}_{ts}"


def _load_model_and_tokenizer(cfg: EvalConfig) -> tuple[Any, Any, str]:
    resolved_token = _resolve_hf_token(cfg.hf_token)

    if cfg.model_mode == "merged":
        if cfg.merged_model_dir is None:
            raise ValueError("--merged-model-dir is required when --model-mode merged.")
        if not cfg.merged_model_dir.exists():
            raise FileNotFoundError(
                f"Merged model directory not found: {cfg.merged_model_dir}"
            )
        load_path = str(cfg.merged_model_dir)
        tokenizer_kwargs: dict[str, Any] = {}
        model_kwargs: dict[str, Any] = {
            "torch_dtype": _resolve_torch_dtype(cfg.torch_dtype)
        }
        if cfg.device_map is not None:
            model_kwargs["device_map"] = cfg.device_map
        if resolved_token is not None:
            tokenizer_kwargs["token"] = resolved_token
            model_kwargs["token"] = resolved_token

        tokenizer = AutoTokenizer.from_pretrained(load_path, **tokenizer_kwargs)
        model = AutoModelForCausalLM.from_pretrained(load_path, **model_kwargs)
        return tokenizer, model, load_path

    if cfg.model_mode == "vanilla":
        if cfg.base_model_name_or_path is None or not cfg.base_model_name_or_path.strip():
            raise ValueError(
                "--base-model-name-or-path is required when --model-mode vanilla."
            )
        resolved_base = cfg.base_model_name_or_path.strip()
        tokenizer_kwargs = {}
        model_kwargs = {"torch_dtype": _resolve_torch_dtype(cfg.torch_dtype)}
        if cfg.device_map is not None:
            model_kwargs["device_map"] = cfg.device_map
        if resolved_token is not None:
            tokenizer_kwargs["token"] = resolved_token
            model_kwargs["token"] = resolved_token

        tokenizer = AutoTokenizer.from_pretrained(resolved_base, **tokenizer_kwargs)
        model = AutoModelForCausalLM.from_pretrained(resolved_base, **model_kwargs)
        return tokenizer, model, resolved_base

    if cfg.model_mode == "adapter":
        if cfg.adapter_dir is None:
            raise ValueError("--adapter-dir is required when --model-mode adapter.")
        if not cfg.adapter_dir.exists():
            raise FileNotFoundError(f"Adapter directory not found: {cfg.adapter_dir}")
        if PeftModel is None:
            raise ImportError("PEFT is required for adapter mode. Install with: pip install peft")

        resolved_base = (
            cfg.base_model_name_or_path.strip()
            if cfg.base_model_name_or_path is not None and cfg.base_model_name_or_path.strip()
            else _resolve_base_model_from_adapter(cfg.adapter_dir)
        )
        tokenizer_kwargs = {}
        model_kwargs = {"torch_dtype": _resolve_torch_dtype(cfg.torch_dtype)}
        if cfg.device_map is not None:
            model_kwargs["device_map"] = cfg.device_map
        if resolved_token is not None:
            tokenizer_kwargs["token"] = resolved_token
            model_kwargs["token"] = resolved_token

        tokenizer = AutoTokenizer.from_pretrained(resolved_base, **tokenizer_kwargs)
        base_model = AutoModelForCausalLM.from_pretrained(resolved_base, **model_kwargs)
        peft_kwargs: dict[str, Any] = {}
        if resolved_token is not None:
            peft_kwargs["token"] = resolved_token
        model = cast(Any, PeftModel).from_pretrained(
            base_model,
            str(cfg.adapter_dir),
            **peft_kwargs,
        )
        return tokenizer, model, resolved_base

    raise ValueError(
        f"Unsupported model mode: {cfg.model_mode}. Use one of: vanilla, adapter, merged."
    )


def _generate_batch(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    num_beams: int,
) -> list[str]:
    if not prompts:
        return []

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    if "attention_mask" in inputs:
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    else:
        input_len = int(inputs["input_ids"].shape[1])
        input_lengths = [input_len for _ in prompts]

    completions: list[str] = []
    for i, prompt_len in enumerate(input_lengths):
        gen_ids = out[i][int(prompt_len):]
        completions.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return completions


def run_evaluation(cfg: EvalConfig) -> dict[str, Any]:
    dataset_dir, split_name = _resolve_test_dataset_dir(
        splits_dir=cfg.splits_dir,
        test_suite=cfg.test_suite,
        proper_ratio=cfg.proper_ratio,
    )
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    ds = load_prepared_dataset(dataset_dir)
    if split_name not in ds:
        available = ", ".join(str(k) for k in ds.keys())
        raise KeyError(
            f"Requested split '{split_name}' not found. Available splits: [{available}]"
        )
    split_ds = cast(Dataset, ds[split_name])

    if cfg.max_samples is not None:
        split_ds = split_ds.select(range(min(cfg.max_samples, len(split_ds))))

    template: PromptTemplate = resolve_prompt_template(dataset_dir, split_ds)
    ground_truth_by_prompt = _load_ground_truth_map(dataset_dir)

    tokenizer, model, resolved_model_ref = _load_model_and_tokenizer(cfg)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    model.eval()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path: Path | None = None
    trace_file = None
    if cfg.save_trace:
        trace_path = cfg.output_dir / "trace.jsonl"
        trace_file = trace_path.open("w", encoding="utf-8")

    total_rows = len(split_ds)
    rows_with_ground_truth = 0
    exec_ok_count = 0
    correct_count = 0
    correct_when_exec_ok_count = 0
    missing_ground_truth_count = 0

    score_executor: ThreadPoolExecutor | None = None
    if cfg.workers > 1:
        score_executor = ThreadPoolExecutor(max_workers=cfg.workers)

    started = perf_counter()
    try:
        with tqdm(
            total=total_rows,
            desc="Eval generate",
            unit="sample",
            leave=False,
        ) as gen_pbar, tqdm(
            total=total_rows,
            desc="Eval execute",
            unit="sample",
            leave=False,
        ) as exec_pbar:
            for batch_start in range(0, total_rows, cfg.generation_batch_size):
                batch_end = min(batch_start + cfg.generation_batch_size, total_rows)
                batch_rows = [cast(dict[str, Any], split_ds[i]) for i in range(batch_start, batch_end)]

                prompts: list[str] = [
                    build_prompt_text(row, template=template, include_output=False)
                    for row in batch_rows
                ]
                generated = _generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_new_tokens=cfg.generation_max_new_tokens,
                    num_beams=cfg.generation_num_beams,
                )
                gen_pbar.update(len(generated))

                prompt_keys: list[str] = []
                expected_batch: list[str] = []
                for local_idx, row in enumerate(batch_rows):
                    prompt_key = _resolve_prompt_key(row)
                    prompt_keys.append(prompt_key)
                    expected = ground_truth_by_prompt.get(prompt_key)
                    if expected is None:
                        missing_ground_truth_count += 1
                        expected = ""
                    else:
                        rows_with_ground_truth += 1
                    expected_batch.append(expected)

                scored_batch = score_predicted_prolog_batch(
                    [(generated[i], expected_batch[i]) for i in range(len(generated))],
                    workers=cfg.workers,
                    executor=score_executor,
                )
                exec_pbar.update(len(scored_batch))

                for local_idx in range(len(batch_rows)):
                    global_idx = batch_start + local_idx
                    prompt_key = prompt_keys[local_idx]
                    expected = expected_batch[local_idx]
                    model_input = prompts[local_idx]
                    model_output = generated[local_idx]
                    exec_ok_inc, raw_correct_inc, exec_result = scored_batch[local_idx]

                    exec_ok_count += exec_ok_inc
                    has_expected = bool(expected)
                    correct_inc = raw_correct_inc if has_expected else 0
                    correct_count += correct_inc
                    correct_when_exec_ok_count += correct_inc
                    is_correct = bool(correct_inc)

                    if trace_file is not None:
                        trace_record = {
                            "index": global_idx,
                            "suite": cfg.test_suite,
                            "dataset_dir": str(dataset_dir),
                            "split": split_name,
                            "prompt_key": prompt_key,
                            "model_input": model_input,
                            "model_output": model_output,
                            "expected_answer": expected,
                            "exec_ok": exec_result.ok,
                            "exec_error_type": exec_result.error_type,
                            "exec_error": exec_result.error,
                            "exec_normalized_answer": exec_result.normalized_answer,
                            "is_correct": is_correct,
                        }
                        trace_file.write(json.dumps(trace_record) + "\n")
    finally:
        if score_executor is not None:
            score_executor.shutdown(wait=True)
        if trace_file is not None:
            trace_file.close()

    elapsed_s = perf_counter() - started
    exec_ok_rate = exec_ok_count / total_rows if total_rows else 0.0
    answer_accuracy = correct_count / total_rows if total_rows else 0.0
    answer_accuracy_on_exec_ok = (
        correct_when_exec_ok_count / exec_ok_count if exec_ok_count else 0.0
    )
    ground_truth_coverage = rows_with_ground_truth / total_rows if total_rows else 0.0

    summary: dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "config": {
            "model_mode": cfg.model_mode,
            "base_model_name_or_path": cfg.base_model_name_or_path,
            "adapter_dir": str(cfg.adapter_dir) if cfg.adapter_dir is not None else None,
            "merged_model_dir": (
                str(cfg.merged_model_dir) if cfg.merged_model_dir is not None else None
            ),
            "resolved_model_ref": resolved_model_ref,
            "test_suite": cfg.test_suite,
            "dataset_dir": str(dataset_dir),
            "split": split_name,
            "proper_ratio": cfg.proper_ratio,
            "generation_batch_size": cfg.generation_batch_size,
            "generation_num_beams": cfg.generation_num_beams,
            "generation_max_new_tokens": cfg.generation_max_new_tokens,
            "workers": cfg.workers,
            "torch_dtype": cfg.torch_dtype,
            "device_map": cfg.device_map,
            "max_samples": cfg.max_samples,
            "save_trace": cfg.save_trace,
        },
        "counts": {
            "checked_rows": total_rows,
            "rows_with_ground_truth": rows_with_ground_truth,
            "missing_ground_truth": missing_ground_truth_count,
            "exec_ok": exec_ok_count,
            "answer_correct": correct_count,
            "answer_correct_when_exec_ok": correct_when_exec_ok_count,
        },
        "rates": {
            "ground_truth_coverage": ground_truth_coverage,
            "exec_ok_rate": exec_ok_rate,
            "answer_accuracy": answer_accuracy,
            "answer_accuracy_on_exec_ok": answer_accuracy_on_exec_ok,
        },
        "runtime_seconds": elapsed_s,
        "artifacts": {
            "summary_json": str(cfg.output_dir / "summary.json"),
            "trace_jsonl": str(trace_path) if trace_path is not None else None,
        },
    }

    summary_path = cfg.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        (
            "[Eval] suite={suite} mode={mode} checked={checked} exec_ok={exec_ok} "
            "correct={correct} exec_ok_rate={exec_ok_rate:.4f} answer_acc={answer_acc:.4f} "
            "answer_acc_on_exec_ok={answer_acc_exec:.4f} summary={summary_path}"
        ).format(
            suite=cfg.test_suite,
            mode=cfg.model_mode,
            checked=total_rows,
            exec_ok=exec_ok_count,
            correct=correct_count,
            exec_ok_rate=exec_ok_rate,
            answer_acc=answer_accuracy,
            answer_acc_exec=answer_accuracy_on_exec_ok,
            summary_path=summary_path,
        )
    )
    if trace_path is not None:
        print(f"[Eval] trace={trace_path}")

    return summary


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run evaluation test suites with either vanilla base model, base+adapter, "
            "or merged model weights."
        )
    )
    parser.add_argument(
        "--model-mode",
        type=str,
        default="vanilla",
        choices=("vanilla", "adapter", "merged"),
    )
    parser.add_argument("--base-model-name-or-path", type=str, default=None)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--merged-model-dir", type=Path, default=None)

    parser.add_argument(
        "--test-suite",
        type=str,
        required=True,
        choices=tuple(sorted(TEST_SUITES.keys())),
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        help="Base versioned splits directory. Defaults to prepare_splits.get_default_splits_dir().",
    )
    parser.add_argument(
        "--proper-ratio",
        type=str,
        default="1to2",
        help='Used by gsm8k_proper suites. Accepts "1to2" or "ratio_1to2".',
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--save-trace", action="store_true")

    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--generation-batch-size", type=int, default=6)
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of worker threads for Prolog execute+compare scoring.",
    )
    parser.add_argument("--generation-num-beams", type=int, default=4)
    parser.add_argument("--generation-max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=("auto", "bfloat16", "float16", "float32"),
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help='Model placement strategy, e.g. "auto" or "none".',
    )
    parser.add_argument("--hf-token", type=str, default=None)

    args = parser.parse_args()

    resolved_device_map: str | None = args.device_map
    if isinstance(resolved_device_map, str) and resolved_device_map.lower() == "none":
        resolved_device_map = None

    if args.generation_batch_size < 1:
        raise ValueError("--generation-batch-size must be >= 1.")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")
    if args.generation_num_beams < 1:
        raise ValueError("--generation-num-beams must be >= 1.")
    if args.generation_max_new_tokens < 1:
        raise ValueError("--generation-max-new-tokens must be >= 1.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be > 0 when provided.")

    resolved_splits_dir = args.splits_dir if args.splits_dir is not None else get_default_splits_dir()
    resolved_output_dir = _build_output_dir(
        output_dir=args.output_dir,
        test_suite=args.test_suite,
        model_mode=args.model_mode,
    )

    return EvalConfig(
        model_mode=args.model_mode,
        base_model_name_or_path=args.base_model_name_or_path,
        adapter_dir=args.adapter_dir,
        merged_model_dir=args.merged_model_dir,
        test_suite=args.test_suite,
        splits_dir=resolved_splits_dir,
        proper_ratio=args.proper_ratio,
        output_dir=resolved_output_dir,
        save_trace=args.save_trace,
        max_samples=args.max_samples,
        generation_batch_size=args.generation_batch_size,
        generation_num_beams=args.generation_num_beams,
        generation_max_new_tokens=args.generation_max_new_tokens,
        workers=args.workers,
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
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
