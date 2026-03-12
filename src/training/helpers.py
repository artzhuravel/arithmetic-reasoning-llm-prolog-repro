from __future__ import annotations

import os
from typing import Any, Protocol
from transformers import AutoTokenizer
from pathlib import Path

from src.data.prepare_splits import get_default_splits_dir

import torch
import logging

LOGGER = logging.getLogger(__name__)

class TrainConfigLike(Protocol):
    @property
    def model_name_or_path(self) -> str:
        ...
    
    @property
    def hf_token(self) -> str | None:
        ...

def _resolve_dataset_dir(
    dataset_dir: Path | None = None,
    splits_dir: Path | None = None,
    dataset_name: str | None = None,
    proper_ratio: str | None = None,
) -> Path:
    if dataset_dir is not None:
        return dataset_dir

    if dataset_name is None:
        raise ValueError(
            "Provide either --dataset-dir or --dataset-name "
            "(gsm8k_prolog, openai_gsm8k, gsm8k_proper)."
        )

    splits_dir = splits_dir if splits_dir is not None else get_default_splits_dir()

    if dataset_name == "gsm8k_proper":
        if proper_ratio is None or not proper_ratio.strip():
            raise ValueError("For gsm8k_proper, --proper-ratio is required.")
        return splits_dir / dataset_name / proper_ratio.strip()

    if proper_ratio is not None:
        LOGGER.warning(
            "--proper-ratio was provided for non-gsm8k_proper dataset and will be ignored."
        )

    return splits_dir / dataset_name


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


def _resolve_hf_token_from_cfg(cfg: TrainConfigLike) -> str | None:
    if cfg.hf_token is not None and cfg.hf_token.strip():
        return cfg.hf_token.strip()

    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.getenv(env_name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _maybe_add_hf_token_from_cfg(
    cfg: TrainConfigLike,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    token = _resolve_hf_token_from_cfg(cfg)
    if token is not None:
        kwargs["token"] = token
    return kwargs


def build_tokenizer(cfg: TrainConfigLike) -> Any:
    tokenizer_kwargs: dict[str, Any] = {"use_fast": True}
    tokenizer_kwargs = _maybe_add_hf_token_from_cfg(cfg, tokenizer_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        use_fast=True,
        **tokenizer_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return tokenizer