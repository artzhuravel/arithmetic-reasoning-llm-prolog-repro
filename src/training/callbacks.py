from __future__ import annotations
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any

from src.prolog.execute import execute_solve
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
        workers: int = 10,
        generation_batch_size: int = 8,
        generation_num_beams: int = 4,
        generation_max_new_tokens: int = 256,
    ):
        if eval_every_steps < 1:
            raise ValueError("eval_every_steps must be >= 1")
        if workers < 1:
            raise ValueError("workers must be >= 1")
        if generation_batch_size < 1:
            raise ValueError("generation_batch_size must be >= 1")
        if generation_num_beams < 1:
            raise ValueError("generation_num_beams must be >= 1")
        if generation_max_new_tokens < 1:
            raise ValueError("generation_max_new_tokens must be >= 1")

        self.tokenizer = tokenizer
        self.eval_rows = eval_rows
        self.gt_map = gt_map
        self.template = template
        self.max_samples = max_samples
        self.eval_every_steps = eval_every_steps
        self.workers = workers
        self.generation_batch_size = generation_batch_size
        self.generation_num_beams = generation_num_beams
        self.generation_max_new_tokens = generation_max_new_tokens

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
        if global_step % self.eval_every_steps != 0 and not is_final_eval:
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

        def _score_single(item: tuple[str, str]) -> tuple[int, int]:
            pred_code, expected = item
            got = execute_solve(pred_code)
            if not got.ok:
                return 0, 0
            return 1, 1 if got.normalized_answer == expected else 0

        exec_ok = 0
        correct = 0
        batch_size = self.generation_batch_size
        executor: ThreadPoolExecutor | None = None
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
                pending_scores: set[Future[tuple[int, int]]] = set()

                def _consume_done_scores(*, block: bool) -> None:
                    nonlocal exec_ok, correct, pending_scores
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
                        exec_ok_inc, correct_inc = future.result()
                        exec_ok += exec_ok_inc
                        correct += correct_inc
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
                        max_length=1024,
                        padding=True,
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        out = model.generate(
                            **inputs,
                            max_new_tokens=self.generation_max_new_tokens,
                            num_beams=self.generation_num_beams,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )

                    if "attention_mask" in inputs:
                        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
                    else:
                        input_len = int(inputs["input_ids"].shape[1])
                        input_lengths = [input_len for _ in batch_rows]

                    batch_preds: list[tuple[str, str]] = []
                    for idx, expected in enumerate(expected_batch):
                        prompt_len = int(input_lengths[idx])
                        gen_ids = out[idx][prompt_len:]
                        pred_code = self.tokenizer.decode(
                            gen_ids,
                            skip_special_tokens=True,
                        )
                        batch_preds.append((pred_code, expected))
                    gen_pbar.update(len(batch_preds))

                    if executor is None:
                        for exec_ok_inc, correct_inc in map(_score_single, batch_preds):
                            exec_ok += exec_ok_inc
                            correct += correct_inc
                            exec_pbar.update(1)
                    else:
                        for item in batch_preds:
                            pending_scores.add(executor.submit(_score_single, item))
                        # Do a non-blocking drain so Prolog execution overlaps
                        # with subsequent generation batches.
                        _consume_done_scores(block=False)

                _consume_done_scores(block=True)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        acc = correct / n if n else 0.0
        exec_rate = exec_ok / n if n else 0.0

        logging.info(
            (
                "[%s] Prolog accuracy done at step=%d epoch=%.4f: "
                "exec_ok_rate=%.4f answer_accuracy=%.4f."
            ),
            self.__class__.__name__,
            step,
            epoch,
            exec_rate,
            acc,
        )
        
        if isinstance(metrics, dict):
            metrics["eval_prolog_exec_ok_rate"] = exec_rate
            metrics["eval_prolog_answer_accuracy"] = acc

        return control
