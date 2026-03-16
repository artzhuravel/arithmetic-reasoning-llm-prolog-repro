# arithmetic-reasoning-llm-prolog-repro

Independent reproduction of:
- Xiaocheng Yang, Bingsen Chen, Yik-Cheung Tam. 2024. *Arithmetic Reasoning with LLM: Prolog Generation & Permutation*. NAACL-HLT 2024 (Short), pp. 699–710. DOI: 10.18653/v1/2024.naacl-short.61
  - Paper: https://aclanthology.org/2024.naacl-short.61/

This repository reproduces an arithmetic-reasoning pipeline in which a language model is trained to generate Prolog programs for GSM8K-style word problems, execute them with SWI-Prolog, and compare the resulting answers against curated ground truth. The retained experiments cover ordinary Prolog SFT (that is, supervised fine-tuning on the original Prolog-formatted dataset without PROPER-style permutation augmentation), a paperstyle-close PROPER SFT track (that is, supervised fine-tuning on a training set augmented with execution-validated permutations of correct Prolog programs), and an additional GRPO reinforcement-learning stage built on top of the best supervised checkpoint.

In the retained artifact set, the strongest ordinary SFT model reaches 54.30% answer accuracy on `gsm8k_prolog_test`, and the best retained RL run improves that to 58.71% while also slightly increasing execution success. The PROPER run did not beat the corresponding ordinary SFT run in this repository.

## Reproduction Report

The full write-up for the retained experiments lives in [paper/reproduction_report.md](paper/reproduction_report.md).

## Retained Results

The table below summarizes the final held-out results for the retained SFT runs and the retained RL extension.

| Model | Evaluation split | Answer accuracy | Exec OK rate | Answer accuracy on exec OK |
| --- | --- | ---: | ---: | ---: |
| Best ordinary SFT | `gsm8k_prolog_test` | 0.5430 | 0.7871 | 0.6899 |
| Paperstyle-close ordinary SFT | `gsm8k_prolog_test` | 0.5308 | 0.7977 | 0.6654 |
| Paperstyle-close PROPER SFT | `gsm8k_proper_test` | 0.5202 | 0.7977 | 0.6520 |
| Best ordinary RL | `gsm8k_prolog_test` | 0.5871 | 0.8046 | 0.7297 |

Briefly:
- The strongest retained supervised baseline is the repo-tuned ordinary Prolog SFT run.
- The retained paperstyle-close PROPER run did not outperform the corresponding ordinary paperstyle-close SFT run in this repository.
- The retained RL extension improved over the best ordinary SFT baseline on the full `gsm8k_prolog_test` split, although the gain is modest.

## Training Dynamics

### Ordinary SFT Validation Loss

![Ordinary SFT validation loss](paper/figures/sft_ordinary_eval_loss.png)

### PROPER SFT Validation Loss

![PROPER SFT validation loss](paper/figures/sft_proper_eval_loss.png)

### RL Training Progress

![RL training progress](paper/figures/rl_training_progress.png)

## Docker (GPU)

This project includes a CUDA-enabled Docker setup for training on Linux + NVIDIA GPUs.

### 1) Build image

```bash
docker compose build trainer
```

### 2) Start container shell

```bash
docker compose run --rm trainer
```

### 3) Run training inside container

```bash
python -m src.training.train_sft \
  --dataset-name gsm8k_proper \
  --proper-ratio 1to2 \
  --output-dir outputs/training/run2 \
  --model-name-or-path mistralai/Mistral-7B-v0.3
```

If you already have a specific prepared dataset path, use `--dataset-dir` instead of `--dataset-name`.
