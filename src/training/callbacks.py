from __future__ import annotations
from typing import Any

from src.prolog.execute import execute_solve
from src.training.data import PromptTemplate, build_prompt_text
from transformers import TrainerCallback
import torch
import logging


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
    ):
        if eval_every_steps < 1:
            raise ValueError("eval_every_steps must be >= 1")

        self.tokenizer = tokenizer
        self.eval_rows = eval_rows
        self.gt_map = gt_map
        self.template = template
        self.max_samples = max_samples
        self.eval_every_steps = eval_every_steps

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
        correct, exec_ok = 0, 0

        for i in range(n):
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

            got = execute_solve(pred_code)
            expected = self.gt_map.get(str(row["input"]).strip(), "")
            if got.ok:
                exec_ok += 1
                if got.normalized_answer == expected:
                    correct += 1

        acc = correct / n if n else 0.0
        exec_rate = exec_ok / n if n else 0.0
        
        if isinstance(metrics, dict):
            metrics["eval_prolog_exec_ok_rate"] = exec_rate
            metrics["eval_prolog_answer_accuracy"] = acc

        return control
