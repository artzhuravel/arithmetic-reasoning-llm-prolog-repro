# Reproduction Report: Arithmetic Reasoning via Prolog Code Generation, Supervised Fine-Tuning, and GRPO Reinforcement Learning

## 1. Scope

This document presents a reproduction report for the arithmetic reasoning pipeline implemented in this repository. The reproduced system targets math word-problem solving by generating Prolog programs, executing them with SWI-Prolog, and scoring the resulting answers against ground-truth GSM8K targets. The report is attached directly to the repository and is intended to document what was actually implemented, trained, and evaluated here, rather than to restate the full claims of the original paper line in abstract form.

The original paper being reproduced is Yang et al. (2024), *Arithmetic Reasoning with LLM: Prolog Generation & Permutation*. That paper studies supervised fine-tuning with LoRA on three 7B model families—Llama-2, CodeLlama, and Mistral—compares Chain-of-Thought (CoT), ordinary Prolog generation, and PROPER augmentation, and reports results on both GSM8K and GSM-HARD. It also studies PROPER permutation ratios `1:0`, `1:1`, and `1:2`. The current repository reproduces only part of that setup. In particular, the cleaned experiments retained in this repository center on a Mistral-family track, ordinary Prolog SFT, one PROPER SFT setting, and an additional RL stage that is not part of the original paper.

The reproduction focuses on three experiment tracks that are present in the current repository state:

- an ordinary `gsm8k_prolog` supervised fine-tuning (SFT) track, used as the primary baseline;
- a `gsm8k_prolog` GRPO reinforcement-learning track, initialized from the best ordinary SFT checkpoint; this RL track is an extension of the paper rather than part of the original Yang et al. setup;
- a `gsm8k_proper` SFT track, included as a secondary comparison rather than the main narrative result.

Accordingly, this report reproduces the paper’s central Prolog-generation idea and its PROPER-style data augmentation in a repository-specific Mistral setting, but it does not reproduce the full three-model sweep from the paper, does not include the paper’s CoT training baseline as a canonical retained experiment, and does not report GSM-HARD results as part of the cleaned artifact set. It also evaluates an RL extension that goes beyond the original paper.

The core experimental comparison in this report is therefore:

1. vanilla `mistralai/Mistral-7B-v0.3`;
2. the best ordinary Prolog SFT adapter in [`outputs/training/gsm8k_prolog_best_sft`](outputs/training/gsm8k_prolog_best_sft);
3. the best ordinary Prolog RL adapter in [`outputs/training/gsm8k_prolog_best_rl`](outputs/training/gsm8k_prolog_best_rl).

Evaluation is performed with the repository’s execution-based evaluator on the `gsm8k_prolog_test` split and reports not only answer accuracy, but also Prolog execution success and detailed execution-error categories. The cleaned evaluation artifacts used by this report are stored in:

- [`outputs/eval/gsm8k_prolog_test_vanilla`](outputs/eval/gsm8k_prolog_test_vanilla)
- [`outputs/eval/gsm8k_prolog_test_best_sft`](outputs/eval/gsm8k_prolog_test_best_sft)
- [`outputs/eval/gsm8k_prolog_test_best_rl`](outputs/eval/gsm8k_prolog_test_best_rl)
- [`outputs/eval/gsm8k_proper_test_best_sft`](outputs/eval/gsm8k_proper_test_best_sft)

Accordingly, this report should be read as a concrete reproduction record for this codebase: what data pipeline was built, how the models were trained, what evaluation protocol was used, and what resulting runs exhibited.

## 2. Objective

The objective of this reproduction is to test whether the repository’s implementation can recover the main practical behavior associated with Prolog-mediated arithmetic reasoning:

- a decoder-only language model can be adapted to emit executable Prolog programs for GSM8K-style math word problems;
- execution-based supervised fine-tuning yields a meaningful baseline on the ordinary `gsm8k_prolog` benchmark;
- reinforcement learning with a Prolog-execution reward can improve over the best SFT baseline on the same task.

Relative to the original paper, the present report has a narrower scope. The paper evaluates LoRA-SFT on Llama-2 7B, CodeLlama 7B, and Mistral 7B, uses both GSM8K and GSM-HARD for evaluation, and reports PROPER ratios `1:0`, `1:1`, and `1:2`. In contrast, the current repository’s cleaned reproduction centers on a Mistral-family model (`mistralai/Mistral-7B-v0.3`), versioned aligned GSM8K-Prolog splits prepared inside the repository, and a single retained PROPER experiment track corresponding to the `1:2` setting. The repository also introduces a GRPO reinforcement-learning stage, which should be understood as an extension built on top of the paper’s SFT-focused methodology rather than a direct paper replication target.

The split construction in this repository is likewise repo-specific. The original paper reports a GSM8K-Prolog corpus of 8792 samples, with 7473 for training and 1319 for test, plus 100 validation examples drawn from training. The current reproduction uses versioned aligned splits generated from fixed dataset revisions and retains the ordinary `gsm8k_prolog` split layout used by the codebase, namely 7358 train, 100 validation, and 1315 test examples. This difference means the present report should be treated as a faithful reproduction of the repository implementation, not as an exact recreation of every corpus detail reported in the paper.

In this repository, the reproduction objective therefore becomes a set of narrower questions:

1. Can ordinary SFT on versioned `gsm8k_prolog` splits produce Prolog programs that execute successfully and return correct numeric answers at nontrivial rates?
2. Does the repository’s retained PROPER-style `1:2` augmentation track improve over ordinary Prolog SFT in this Mistral-based setting?
3. What failure modes dominate when generation fails: syntax errors, runtime execution errors, no-solution outputs, or other categories?
4. As an extension beyond the original paper, does GRPO fine-tuning from the best ordinary SFT adapter improve full-test performance on `gsm8k_prolog_test` relative to that SFT baseline?

The report will use the following model hierarchy as its narrative backbone:

- `vanilla` model: the unadapted base model, included to show that raw generation is almost entirely non-functional for this task;
- `best ordinary SFT` model: the strongest supervised baseline and the starting point for RL;
- `best ordinary RL` model: the main extension result for reinforcement learning, included because this repository goes beyond the original paper;
- `best PROPER SFT` model: an auxiliary comparison used to contextualize the ordinary-track results.

The central success criterion for this reproduction is not merely whether generated text looks plausible, but whether generated Prolog code executes and yields correct answers under a uniform SWI-Prolog evaluation backend. For that reason, the main outcome measures in later sections will emphasize:

- `exec_ok_rate`
- `answer_accuracy`
- `answer_accuracy_on_exec_ok`
- execution-outcome breakdowns such as `syntax_error`, `execution_error`, and `no_solution`
