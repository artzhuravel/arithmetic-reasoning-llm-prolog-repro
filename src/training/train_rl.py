import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

from transformers import AutoTokenizer
from datasets import DatasetDict
from src.data.prepare_splits import get_default_splits_dir
from src.training.data import load_prepared_dataset, load_training_splits, preview_formatted_examples, resolve_eval_rows, resolve_prompt_template, load_ground_truth_map
from src.training.helpers import _resolve_hf_token_from_cfg, build_tokenizer, _resolve_dataset_dir

TRAINING_RESULTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "training"
LOGGER = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    pass


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
    num_generations: int = 10
    temperature: float = 0.7
    reward: RewardConfig = field(default_factory=RewardConfig)
    torch_dtype: str = "bfloat16"
    device_map: str | None = "auto"  # allow None
    hf_token: str | None = None

    @property
    def model_name_or_path(self) -> str:
        return self.base_model_name_or_path

@dataclass(frozen=True)
class RunContext:
    cfg: RLTrainConfig
    tokenizer: AutoTokenizer
    raw_dataset: DatasetDict



def run(cfg: RLTrainConfig) -> None:
    
    raw_ds = load_prepared_dataset(cfg.dataset_dir)
    
    train_ds, eval_ds = load_training_splits(
        cfg.dataset_dir,
        mode="rl",
        max_train_samples=cfg.max_train_samples,
        max_eval_samples=cfg.max_eval_samples
    )
    
    preview_formatted_examples(train_ds, eval_ds, n=1)
    
    if cfg.dry_run:
        print("\n[dry-run] stopping before tokenizer/model/trainer setup.")
        return
    
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Training outputs directory: %s", cfg.output_dir)
    if _resolve_hf_token_from_cfg(cfg) is None:
        LOGGER.warning("HF token not detected. Downloads will be unauthenticated and may be rate-limited.")
    else:
        LOGGER.info("HF token detected. Using authenticated Hugging Face Hub requests.")
    
    train_text, eval_text = load_training_splits(
        cfg.dataset_dir,
        mode="rl",
        max_train_samples=cfg.max_train_samples,
        max_eval_samples=cfg.max_eval_samples
    )
    
    tokenizer = build_tokenizer(cfg)
    
    pass























def parse_args() -> RLTrainConfig:
    parser = argparse.ArgumentParser(description="RL training for PROPER/GSM8K-Prolog data.")
    parser.add_argument("--dataset-dir", type=Path, required=False)
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=False,
        choices=("gsm8k_prolog", "openai_gsm8k", "gsm8k_proper"),  # fail fast
    )
    parser.add_argument("--proper-ratio", type=str, required=False)
    parser.add_argument("--splits-dir", type=Path, required=False)
    parser.add_argument("--base-model-name-or-path", type=str, required=True)
    parser.add_argument("--sft-adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--max-train-samples", type=int, required=False)
    parser.add_argument("--max-eval-samples", type=int, required=False)
    parser.add_argument("--num-generations", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--hf-token", type=str, required=False)

    args = parser.parse_args()

    resolved_device_map: str | None = args.device_map
    if isinstance(resolved_device_map, str) and resolved_device_map.lower() == "none":
        resolved_device_map = None

    return RLTrainConfig(
        dataset_dir=_resolve_dataset_dir(
            dataset_dir=args.dataset_dir,
            splits_dir=args.splits_dir,
            dataset_name=args.dataset_name,
            proper_ratio=args.proper_ratio,
        ),
        base_model_name_or_path=args.base_model_name_or_path,
        sft_adapter_dir=args.sft_adapter_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        dry_run=args.dry_run,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        num_generations=args.num_generations,
        temperature=args.temperature,
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
    print(cfg)


if __name__ == "__main__":
    main()
