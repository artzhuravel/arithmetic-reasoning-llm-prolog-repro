from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

from src.data.prepare_splits import get_default_splits_dir

LOGGER = logging.getLogger(__name__)


class HFTokenConfigLike(Protocol):
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
        resolved_ratio = proper_ratio.strip()
        if not resolved_ratio.startswith("ratio_"):
            resolved_ratio = f"ratio_{resolved_ratio}"
        return splits_dir / dataset_name / resolved_ratio

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


def _resolve_hf_token(explicit_token: str | None = None) -> str | None:
    if explicit_token is not None and explicit_token.strip():
        return explicit_token.strip()

    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.getenv(env_name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _resolve_hf_token_from_cfg(cfg: HFTokenConfigLike) -> str | None:
    return _resolve_hf_token(cfg.hf_token)


def _maybe_add_hf_token(hf_token: str | None, kwargs: dict[str, Any]) -> dict[str, Any]:
    token = _resolve_hf_token(hf_token)
    if token is not None:
        kwargs["token"] = token
    return kwargs


def _maybe_add_hf_token_from_cfg(
    cfg: HFTokenConfigLike,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    return _maybe_add_hf_token(_resolve_hf_token_from_cfg(cfg), kwargs)


def build_tokenizer(
    *,
    model_name_or_path: str,
    hf_token: str | None = None,
    padding: str = "right",
) -> Any:
    tokenizer_kwargs: dict[str, Any] = {}
    tokenizer_kwargs = _maybe_add_hf_token(hf_token, tokenizer_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        **tokenizer_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding
    return tokenizer


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


def build_model(
    *,
    model_name_or_path: str,
    torch_dtype: str,
    quantization: str | None = None,
    device_map: str | None = "auto",
    hf_token: str | None = None,
    attach_adapter: bool = False,
    adapter_dir: Path | None = None,
    adapter_trainable: bool = False,
) -> Any:
    if attach_adapter and adapter_dir is None:
        raise ValueError("--adapter-dir is required when attach_adapter is True.")
    if not attach_adapter and adapter_dir is not None:
        raise ValueError("adapter_dir requires attach_adapter=True.")
    if adapter_trainable and not attach_adapter:
        raise ValueError("adapter_trainable requires attach_adapter=True.")
    if adapter_dir is not None and not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    if attach_adapter and PeftModel is None:
        raise ImportError("PEFT is required for adapter mode. Install with: pip install peft")

    resolved_quantization = quantization or "none"
    quantization_config = _build_quantization_config(
        quantization=resolved_quantization,
        torch_dtype=torch_dtype,
    )
    model_kwargs: dict[str, Any] = {
        "torch_dtype": _resolve_torch_dtype(torch_dtype),
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    model_kwargs = _maybe_add_hf_token(hf_token, model_kwargs)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )

    if attach_adapter:
        peft_kwargs: dict[str, Any] = {}
        peft_kwargs = _maybe_add_hf_token(hf_token, peft_kwargs)
        model = cast(Any, PeftModel).from_pretrained(
            model,
            str(adapter_dir),
            is_trainable=adapter_trainable,
            **peft_kwargs,
        )

    return model
