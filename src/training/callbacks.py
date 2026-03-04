from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.prolog.execute import execute_solve
from src.training.data import PromptTemplate, build_prompt_text
from transformers import TrainerCallback
import torch
import logging

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable


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
    ):
        if eval_every_steps < 1:
            raise ValueError("eval_every_steps must be >= 1")
        if workers < 1:
            raise ValueError("workers must be >= 1")

        self.tokenizer = tokenizer
        self.eval_rows = eval_rows
        self.gt_map = gt_map
        self.template = template
        self.max_samples = max_samples
        self.eval_every_steps = eval_every_steps
        self.workers = workers

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
        if int(state.global_step) % self.eval_every_steps != 0:
            return control

        model.eval()
        n = min(self.max_samples, len(self.eval_rows))
        step = int(state.global_step)
        epoch = float(state.epoch or 0.0)
        progress_disabled = bool(getattr(args, "disable_tqdm", False))
        generated_preds: list[tuple[str, str]] = []

        logging.info(
            "[%s] Running Prolog accuracy check at step=%d epoch=%.4f on %d sample(s).",
            self.__class__.__name__,
            step,
            epoch,
            n,
        )

        for i in tqdm(
            range(n),
            total=n,
            desc=f"Prolog eval generate (step {step})",
            unit="sample",
            leave=False,
            disable=progress_disabled,
        ):
            row = self.eval_rows[i]
            prompt = build_prompt_text(
                row, template=self.template, include_output=False
            )

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_beams=4,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            pred_code = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

            expected_key = (
                str(row["input"]).strip()
                if "input" in row
                else str(row.get("question", "")).strip()
            )
            expected = self.gt_map.get(expected_key, "")
            generated_preds.append((pred_code, expected))

        def _score_single(item: tuple[str, str]) -> tuple[int, int]:
            pred_code, expected = item
            got = execute_solve(pred_code)
            if not got.ok:
                return 0, 0
            return 1, 1 if got.normalized_answer == expected else 0

        exec_ok = 0
        correct = 0
        if self.workers == 1:
            score_iter = map(_score_single, generated_preds)
            for exec_ok_inc, correct_inc in tqdm(
                score_iter,
                total=n,
                desc=f"Prolog eval execute (step {step})",
                unit="sample",
                leave=False,
                disable=progress_disabled,
            ):
                exec_ok += exec_ok_inc
                correct += correct_inc
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                score_iter = executor.map(_score_single, generated_preds)
                for exec_ok_inc, correct_inc in tqdm(
                    score_iter,
                    total=n,
                    desc=f"Prolog eval execute (step {step})",
                    unit="sample",
                    leave=False,
                    disable=progress_disabled,
                ):
                    exec_ok += exec_ok_inc
                    correct += correct_inc

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
