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

## Preliminary Results (SFT, Prolog Evaluation)

Primary metric: **Answer Accuracy** on `gsm8k_prolog_test`  
Secondary metric: **Execution OK Rate**

| Suite | Mode | Adapter | N | Exec OK % | Answer Acc % | Acc \| Exec OK % | Source |
|---|---|---|---:|---:|---:|---:|---|
| gsm8k_prolog_test | vanilla | - | 1315 | 0.00 | 0.00 | 0.00 | `outputs/eval/gsm8k_prolog_test_vanilla_20260309_152610/summary.json` |
| gsm8k_prolog_test | adapter | prolog_2e-4_no_8_3_callback/checkpoint-600 | 1315 | 79.32 | 55.21 | 69.61 | `outputs/eval/gsm8k_prolog_test_adapter_20260309_160209/summary.json` |
| gsm8k_prolog_test | adapter | proper_2e-4_no_8_3_no_callback_correct1/checkpoint-200 | 1315 | 77.72 | 49.43 | 63.60 | `outputs/eval/gsm8k_prolog_test_adapter_proper_correct1_ckpt200/summary.json` |

### Training Metrics (SFT Runs)

| Run | Train Samples | Epochs | Train Loss | Eval Loss | Train Runtime (s) | Source |
|---|---:|---:|---:|---:|---:|---|
| prolog_2e-4_no_8_3_callback | 7358 | 4.0 | 0.3951 | 0.4363 | 2860.57 | `outputs/training/prolog_2e-4_no_8_3_callback/all_results.json` |
| proper_2e-4_no_8_3_no_callback_correct1 | 21176 | 4.0 | 0.2784 | 0.4540 | 8168.99 | `outputs/training/proper_2e-4_no_8_3_no_callback_correct1/all_results.json` |
