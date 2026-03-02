from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Callable, cast

from src.prolog.execute import execute_solve
from transformers import TrainerCallback
import torch


class PrologAccuracyCallback(TrainerCallback):
    def __init__(self, *, tokenizer, eval_rows, gt_map, out_dir: Path, max_samples: int = 100):
        self.tokenizer = tokenizer
        self.eval_rows = eval_rows
        self.gt_map = gt_map
        self.out_file = out_dir / "prolog_eval_metrics.jsonl"
        self.max_samples = max_samples
        self.history = []
        self.last_result = None

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if model is None:
            return control

        model.eval()
        n = min(self.max_samples, len(self.eval_rows))
        correct, exec_ok = 0, 0

        for i in range(n):
            row = self.eval_rows[i]
            prompt = (
                "### Instruction\n"
                f"{row['instruction']}\n\n"
                "### Input\n"
                f"{row['input']}\n\n"
                "### Output\n"
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
        
        result = {
            "step": int(state.global_step),
            "epoch": float(state.epoch or 0.0),
            "prolog_eval_samples": n,
            "prolog_exec_ok_rate": exec_rate,
            "prolog_answer_accuracy": acc,
        }

        self.last_result = result
        self.history.append(result)

        metrics = kwargs.get("metrics")
        if isinstance(metrics, dict):
            metrics["eval_prolog_exec_ok_rate"] = exec_rate
            metrics["eval_prolog_answer_accuracy"] = acc

        return control