from __future__ import annotations
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
from pathlib import Path
from typing import Any, Sequence, Tuple
import warnings

from src.prolog.execute import PrologExecutionResult, execute_solve
from src.training.data import PromptTemplate, build_prompt_text
from transformers import TrainerCallback
import torch
import logging

def _load_tqdm() -> Any:
    try:
        from tqdm.auto import tqdm
        return tqdm
    except ImportError:
        class _NoOpTqdm:  # pragma: no cover - exercised only when tqdm is unavailable
            def __init__(self, iterable=None, **kwargs):
                self._iterable = iterable

            def __iter__(self):
                if self._iterable is None:
                    return iter(())
                return iter(self._iterable)

            def update(self, n=1):
                return None

            def close(self):
                return None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        def _noop(iterable=None, **kwargs):
            return _NoOpTqdm(iterable=iterable, **kwargs)
        return _noop


tqdm = _load_tqdm()

_KNOWN_EXECUTION_OUTCOMES: tuple[str, ...] = (
    "ok",
    "no_solution",
    "syntax_error",
    "execution_error",
    "timeout",
    "dependency_error",
    "unknown_error",
)


def _resolve_execution_outcome_name(result: PrologExecutionResult) -> str:
    if result.ok:
        return "ok"
    if result.error_type is None or not result.error_type.strip():
        return "unknown_error"
    return result.error_type.strip()


def _build_execution_outcome_summary(
    outcome_counts: Counter[str],
    *,
    total_samples: int,
) -> tuple[dict[str, int], dict[str, float]]:
    ordered_counts: dict[str, int] = {
        key: int(outcome_counts.get(key, 0)) for key in _KNOWN_EXECUTION_OUTCOMES
    }
    for key in sorted(outcome_counts):
        if key not in ordered_counts:
            ordered_counts[key] = int(outcome_counts[key])

    ordered_rates = {
        key: (value / total_samples if total_samples else 0.0)
        for key, value in ordered_counts.items()
    }
    return ordered_counts, ordered_rates


def score_predicted_prolog(scorable_item: tuple[str, str]) -> tuple[int, int, PrologExecutionResult]:
    """
    Execute predicted Prolog code and compare normalized answer to expected.

    Returns:
    - exec_ok_inc: 1 when execution succeeded else 0
    - correct_inc: 1 when execution succeeded and answer matches expected else 0
    - result: full PrologExecutionResult
    """
    pred_code, expected = scorable_item
    got = execute_solve(pred_code)
    if not got.ok:
        return 0, 0, got
    return 1, 1 if got.normalized_answer == expected else 0, got


def score_predicted_prolog_batch(
    items: Sequence[tuple[str, str]],
    *,
    workers: int = 1,
    executor: ThreadPoolExecutor | None = None,
) -> list[tuple[int, int, PrologExecutionResult]]:
    """
    Batch variant of score_predicted_prolog with optional thread workers.

    Input item format: (predicted_prolog_code, expected_normalized_answer).
    When `executor` is provided, it is reused instead of creating a new pool.
    """
    if workers < 1:
        raise ValueError("workers must be >= 1")

    if executor is not None:
        return list(executor.map(score_predicted_prolog, items))

    if workers == 1:
        return [score_predicted_prolog(item) for item in items]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(score_predicted_prolog, items))


class PrologAccuracyCallback(TrainerCallback):
    def __init__(
        self,
        *,
        tokenizer,
        eval_rows,
        gt_map,
        template: PromptTemplate,
        max_samples: int = 100,
        eval_every_steps: int = 1,
        eval_strategy: str = "steps",
        workers: int = 10,
        generation_batch_size: int = 8,
        generation_num_beams: int = 4,
        generation_max_new_tokens: int = 256,
        prompt_max_length: int = 1024,
    ):
        if eval_every_steps < 1:
            raise ValueError("eval_every_steps must be >= 1")
        if eval_strategy not in {"no", "steps", "epoch"}:
            raise ValueError("eval_strategy must be one of: no, steps, epoch")
        if workers < 1:
            raise ValueError("workers must be >= 1")
        if generation_batch_size < 1:
            raise ValueError("generation_batch_size must be >= 1")
        if generation_num_beams < 1:
            raise ValueError("generation_num_beams must be >= 1")
        if generation_max_new_tokens < 1:
            raise ValueError("generation_max_new_tokens must be >= 1")
        if prompt_max_length < 1:
            raise ValueError("prompt_max_length must be >= 1")

        self.tokenizer = tokenizer
        self.eval_rows = eval_rows
        self.gt_map = gt_map
        self.template = template
        self.max_samples = max_samples
        self.eval_every_steps = eval_every_steps
        self.eval_strategy = eval_strategy
        self.workers = workers
        self.generation_batch_size = generation_batch_size
        self.generation_num_beams = generation_num_beams
        self.generation_max_new_tokens = generation_max_new_tokens
        self.prompt_max_length = prompt_max_length
        self.last_result: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []

    def _append_run_metrics(self, output_dir: str, result: dict[str, Any]) -> None:
        output_path = Path(output_dir) / "prolog_accuracy_metrics.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

    def _write_checkpoint_metrics(self, output_dir: str, step: int) -> None:
        if self.last_result is None:
            return
        checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
        if not checkpoint_dir.exists():
            return
        checkpoint_payload = {
            "latest": self.last_result,
            "history": self.history,
        }
        checkpoint_path = checkpoint_dir / "prolog_accuracy_metrics.json"
        with checkpoint_path.open("w", encoding="utf-8") as f:
            json.dump(checkpoint_payload, f, indent=2)

    def on_save(self, args, state, control, **kwargs) -> None:
        if state.is_world_process_zero:
            self._write_checkpoint_metrics(str(args.output_dir), int(state.global_step))

    def on_evaluate(self,
                    args,
                    state,
                    control,
                    model=None,
                    metrics=None,
                    **kwargs) -> Any:

        if model is None:
            logging.warning(
                "[%s] Model is None at step=%d epoch=%.4f; skipping evaluation.",
                self.__class__.__name__,
                int(state.global_step),
                float(state.epoch or 0.0),
            )
            return control

        if not state.is_world_process_zero:
            return control
        global_step = int(state.global_step)
        max_steps = int(state.max_steps or 0)
        is_final_eval = max_steps > 0 and global_step >= max_steps
        should_skip = (
            self.eval_strategy == "steps"
            and global_step % self.eval_every_steps != 0
            and not is_final_eval
        )
        if should_skip:
            return control

        model.eval()
        n = min(self.max_samples, len(self.eval_rows))
        step = global_step
        epoch = float(state.epoch or 0.0)
        progress_disabled = bool(getattr(args, "disable_tqdm", False))

        logging.info(
            (
                "[%s] Running Prolog accuracy check at step=%d epoch=%.4f on %d sample(s) "
                "(generation_batch_size=%d, workers=%d, generation_num_beams=%d, "
                "generation_max_new_tokens=%d)."
            ),
            self.__class__.__name__,
            step,
            epoch,
            n,
            self.generation_batch_size,
            self.workers,
            self.generation_num_beams,
            self.generation_max_new_tokens,
        )

        exec_ok = 0
        correct = 0
        execution_outcome_counts: Counter[str] = Counter()
        batch_size = self.generation_batch_size
        executor: ThreadPoolExecutor | None = None
        padding_side_original = getattr(self.tokenizer, "padding_side", None)
        if padding_side_original != "left":
            self.tokenizer.padding_side = "left"
        if self.workers > 1:
            executor = ThreadPoolExecutor(max_workers=self.workers)
        try:
            with tqdm(
                total=n,
                desc=f"Prolog eval generate (step {step})",
                unit="sample",
                leave=False,
                disable=progress_disabled,
            ) as gen_pbar, tqdm(
                total=n,
                desc=f"Prolog eval execute (step {step})",
                unit="sample",
                leave=False,
                disable=progress_disabled,
            ) as exec_pbar:
                pending_scores: set[Future[tuple[int, int, PrologExecutionResult]]] = set()

                def _consume_done_scores(*, block: bool) -> None:
                    nonlocal exec_ok, correct, execution_outcome_counts, pending_scores
                    if not pending_scores:
                        return
                    if block:
                        done, not_done = wait(pending_scores)
                    else:
                        done, not_done = wait(
                            pending_scores,
                            timeout=0.0,
                            return_when=FIRST_COMPLETED,
                        )
                    pending_scores = set(not_done)
                    for future in done:
                        exec_ok_inc, correct_inc, exec_result = future.result()
                        exec_ok += exec_ok_inc
                        correct += correct_inc
                        execution_outcome_counts[_resolve_execution_outcome_name(exec_result)] += 1
                        exec_pbar.update(1)

                for batch_start in range(0, n, batch_size):
                    batch_end = min(batch_start + batch_size, n)
                    batch_rows = [
                        self.eval_rows[i]
                        for i in range(batch_start, batch_end)
                    ]

                    prompts: list[str] = []
                    expected_batch: list[str] = []
                    for row in batch_rows:
                        prompts.append(
                            build_prompt_text(
                                row, template=self.template, include_output=False
                            )
                        )
                        expected_key = (
                            str(row["input"]).strip()
                            if "input" in row
                            else str(row.get("question", "")).strip()
                        )
                        expected_batch.append(self.gt_map.get(expected_key, ""))

                    inputs = self.tokenizer(
                        prompts,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.prompt_max_length,
                        padding=True,
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=(
                                r"MatMul8bitLt: inputs will be cast from "
                                r"torch\.float32 to float16 during quantization"
                            ),
                            category=UserWarning,
                        )
                        with torch.no_grad():
                            out = model.generate(
                                **inputs,
                                max_new_tokens=self.generation_max_new_tokens,
                                num_beams=self.generation_num_beams,
                                do_sample=False,
                                pad_token_id=self.tokenizer.pad_token_id,
                            )

                    # For decoder-only generation, returned sequences include
                    # the full padded input width. With left padding, slicing
                    # by attention_mask.sum would keep part of the prompt.
                    input_len = int(inputs["input_ids"].shape[1])
                    batch_preds: list[tuple[str, str]] = []
                    for idx, expected in enumerate(expected_batch):
                        gen_ids = out[idx][input_len:]
                        pred_code = self.tokenizer.decode(
                            gen_ids,
                            skip_special_tokens=True,
                        )
                        batch_preds.append((pred_code, expected))
                    gen_pbar.update(len(batch_preds))

                    if executor is None:
                        for exec_ok_inc, correct_inc, exec_result in map(score_predicted_prolog, batch_preds):
                            exec_ok += exec_ok_inc
                            correct += correct_inc
                            execution_outcome_counts[_resolve_execution_outcome_name(exec_result)] += 1
                            exec_pbar.update(1)
                    else:
                        for item in batch_preds:
                            pending_scores.add(executor.submit(score_predicted_prolog, item))
                        _consume_done_scores(block=False)

                _consume_done_scores(block=True)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
            if padding_side_original is not None:
                self.tokenizer.padding_side = padding_side_original

        acc = correct / n if n else 0.0
        exec_rate = exec_ok / n if n else 0.0
        execution_outcome_count_summary, execution_outcome_rate_summary = (
            _build_execution_outcome_summary(
                execution_outcome_counts,
                total_samples=n,
            )
        )
        result = {
            "step": step,
            "epoch": epoch,
            "samples": n,
            "exec_ok_rate": exec_rate,
            "answer_accuracy": acc,
            "execution_outcomes": {
                "counts": execution_outcome_count_summary,
                "rates": execution_outcome_rate_summary,
            },
        }
        self.last_result = result
        self.history.append(result)
        self._append_run_metrics(str(args.output_dir), result)

        logging.info(
            (
                "[%s] Prolog accuracy done at step=%d epoch=%.4f: "
                "exec_ok_rate=%.4f answer_accuracy=%.4f execution_outcomes=%s."
            ),
            self.__class__.__name__,
            step,
            epoch,
            exec_rate,
            acc,
            json.dumps(execution_outcome_count_summary, sort_keys=True),
        )
        print(
            (
                f"[PrologAccuracyCallback] step={step} epoch={epoch:.4f} "
                f"exec_ok_rate={exec_rate:.4f} answer_accuracy={acc:.4f} "
                f"execution_outcomes={json.dumps(execution_outcome_count_summary, sort_keys=True)}"
            ),
            flush=True,
        )
        
        if isinstance(metrics, dict):
            metrics["eval_prolog_exec_ok_rate"] = exec_rate
            metrics["eval_prolog_answer_accuracy"] = acc
            for outcome_name, outcome_count in execution_outcome_count_summary.items():
                metrics[f"eval_prolog_{outcome_name}_count"] = outcome_count
                metrics[f"eval_prolog_{outcome_name}_rate"] = execution_outcome_rate_summary[
                    outcome_name
                ]

        return control
