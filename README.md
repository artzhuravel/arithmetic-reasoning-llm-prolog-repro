# arithmetic-reasoning-llm-prolog-repro

Independent reproduction of:
- Xiaocheng Yang, Bingsen Chen, Yik-Cheung Tam. 2024. *Arithmetic Reasoning with LLM: Prolog Generation & Permutation*. NAACL-HLT 2024 (Short), pp. 699–710. DOI: 10.18653/v1/2024.naacl-short.61
  - Paper: https://aclanthology.org/2024.naacl-short.61/

This repository is not affiliated with the paper authors.

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
