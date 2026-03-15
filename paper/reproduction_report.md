# Reproduction Report: Arithmetic Reasoning via Prolog Code Generation, Supervised Fine-Tuning, and GRPO Reinforcement Learning

## 1. Scope

This document presents a reproduction report for the arithmetic reasoning pipeline implemented in this repository. The reproduced system targets math word-problem solving by generating Prolog programs, executing them with SWI-Prolog, and scoring the resulting answers against ground-truth GSM8K targets. The report is attached directly to the repository and is intended to document what was actually implemented, trained, and evaluated here, rather than to restate the full claims of the original paper line in abstract form.

The original paper being reproduced is Yang et al. (2024), *Arithmetic Reasoning with LLM: Prolog Generation & Permutation*. That paper studies supervised fine-tuning with LoRA on three 7B model families—Llama-2, CodeLlama, and Mistral—compares Chain-of-Thought (CoT), ordinary Prolog generation, and PROPER augmentation, and reports results on both GSM8K and GSM-HARD. It also studies PROPER permutation ratios `1:0`, `1:1`, and `1:2`. The current repository reproduces only part of that setup. In particular, the cleaned experiments retained in this repository center on a Mistral-family track, ordinary Prolog SFT, a paperstyle-close PROPER SFT run, and an additional RL stage that is not part of the original paper.

The reproduction focuses on three experiment tracks that are present in the current repository state:

- an ordinary `gsm8k_prolog` supervised fine-tuning (SFT) track, used as the primary baseline;
- a `gsm8k_prolog` GRPO reinforcement-learning track, initialized from the best ordinary SFT checkpoint; this RL track is an extension of the paper rather than part of the original Yang et al. setup;
- a `gsm8k_proper` SFT track, retained in the current artifact set as a paperstyle-close comparison rather than as a separate canonical “best PROPER” run.

Accordingly, this report reproduces the paper’s central Prolog-generation idea and its PROPER-style data augmentation in a repository-specific Mistral setting, but it does not reproduce the full three-model sweep, does not include the paper’s CoT training baseline, and does not report GSM-HARD results as part of the cleaned artifact set. It also evaluates an RL extension that goes beyond the original paper.

The core experimental comparison in this report is therefore:

1. vanilla `mistralai/Mistral-7B-v0.3`;
2. the best ordinary Prolog SFT adapter in [`outputs/training/gsm8k_prolog_best_sft`](outputs/training/gsm8k_prolog_best_sft);
3. the best ordinary Prolog RL adapter in [`outputs/training/gsm8k_prolog_best_rl`](outputs/training/gsm8k_prolog_best_rl).

Evaluation is performed with the repository’s execution-based evaluator on the `gsm8k_prolog_test` split and reports not only answer accuracy, but also Prolog execution success and detailed execution-error categories. The cleaned evaluation artifacts used by this report are stored in:

- [`outputs/eval/gsm8k_prolog_test_vanilla`](outputs/eval/gsm8k_prolog_test_vanilla)
- [`outputs/eval/gsm8k_prolog_test_adapter_best_sft`](outputs/eval/gsm8k_prolog_test_adapter_best_sft)
- [`outputs/eval/gsm8k_prolog_test_adapter_best_rl`](outputs/eval/gsm8k_prolog_test_adapter_best_rl)
- [`outputs/eval/gsm8k_prolog_test_adapter_paperstyle_close`](outputs/eval/gsm8k_prolog_test_adapter_paperstyle_close)
- [`outputs/eval/gsm8k_proper_test_adapter_paperstyle_close`](outputs/eval/gsm8k_proper_test_adapter_paperstyle_close)

Accordingly, this report should be read as a reproduction record for this codebase.

## 2. Objective

The objective of this reproduction is to test whether the repository’s implementation can recover the main practical behavior associated with Prolog-mediated arithmetic reasoning:

- a decoder-only language model can be adapted to emit executable Prolog programs for GSM8K-style math word problems;
- execution-based supervised fine-tuning yields a meaningful baseline on the ordinary `gsm8k_prolog` benchmark;
- reinforcement learning with a Prolog-execution reward can improve over the best SFT baseline on the same task.

Relative to the original paper, the present report has a narrower scope. The paper evaluates LoRA-SFT on Llama-2 7B, CodeLlama 7B, and Mistral 7B, uses both GSM8K and GSM-HARD for evaluation, and reports PROPER ratios `1:0`, `1:1`, and `1:2`. In contrast, the current repository’s cleaned reproduction centers on a Mistral-family model (`mistralai/Mistral-7B-v0.3`), versioned aligned GSM8K-Prolog splits prepared inside the repository, and a retained PROPER comparison corresponding to the `1:2` setting in the form of a paperstyle-close SFT run. The repository also introduces a GRPO reinforcement-learning stage, which should be understood as an extension built on top of the paper’s SFT-focused methodology rather than a direct paper replication target.

The split construction in this repository is likewise repo-specific. The original paper reports a GSM8K-Prolog corpus of 8792 samples, with 7473 for training and 1319 for test, plus 100 validation examples drawn from training. The current reproduction uses versioned aligned splits generated from fixed dataset revisions and retains the ordinary `gsm8k_prolog` split layout used by the codebase, namely 7358 train, 100 validation, and 1315 test examples. Note that the training and test splits in this reproduction are slightly smaller than those used in the original paper. This is because rows that did not align with the original OpenAI GSM8K dataset were removed to preserve the possibility of a more faithful future performance comparison on the Prolog and the OpenAI datasets, although that comparison was not conducted here. Regardless, this difference means the present report should be treated as a faithful reproduction of the repository implementation, not as an exact recreation of every corpus detail reported in the paper.

In this repository, the reproduction objective therefore becomes a set of narrower questions:

1. Can ordinary SFT on versioned `gsm8k_prolog` splits produce Prolog programs that execute successfully and return correct numeric answers at nontrivial rates?
2. Does the repository’s retained PROPER-style `1:2` augmentation track improve over ordinary Prolog SFT in this Mistral-based setting?
3. What failure modes dominate when generation fails: syntax errors, runtime execution errors, no-solution outputs, or other categories?
4. As an extension beyond the original paper, does GRPO fine-tuning from the best ordinary SFT adapter improve full-test performance on `gsm8k_prolog_test` relative to that SFT baseline?

The report will use the following model hierarchy as its narrative backbone:

- `vanilla` model: the unadapted base model, included to show that raw generation is almost entirely non-functional for this task;
- `best ordinary SFT` model: the strongest supervised baseline and the starting point for RL;
- `best ordinary RL` model: the main extension result for reinforcement learning, included because this repository goes beyond the original paper;
- `paperstyle-close PROPER SFT` model: an auxiliary comparison used to contextualize the ordinary-track results.

The central success criterion for this reproduction is whether generated Prolog code executes and yields correct answers under a uniform SWI-Prolog evaluation backend. For that reason, the main outcome measures in later sections will emphasize:

- `exec_ok_rate`
- `answer_accuracy`
- `answer_accuracy_on_exec_ok`
- execution-outcome breakdowns such as `syntax_error`, `execution_error`, and `no_solution`

## 3. Data Preparation and Ground-Truth Curation

The split builder in [`src/data/prepare_splits.py`](src/data/prepare_splits.py) loads two raw datasets from pinned revisions: `Thomas-X-Yang/gsm8k-prolog` and `openai/gsm8k`. These are then aligned by index and checked row by row. For each paired example, the code verifies that the Prolog prompt text and the OpenAI GSM8K question match after minimal whitespace normalization, extracts the final OpenAI answer from the `#### ...` marker, executes the Prolog program with SWI-Prolog through [`execute_solve`](src/prolog/execute.py), and keeps the pair only if the executed Prolog answer matches the normalized OpenAI answer. If any of these checks fail, the corresponding row is removed from both datasets.

This filtering step is applied to the raw `train` and `test` splits before validation is created. The raw paired source revisions each contain `7473` train examples and `1319` test examples. After filtering, the retained ordinary split stored in this repository becomes `7458` cleaned train examples and `1315` cleaned test examples. The validation split is then sampled from the cleaned train split rather than being inherited from the raw source. With the default settings the builder shuffles the cleaned train split with `seed=42`, takes the first `100` examples as validation, and leaves the remaining `7358` examples as training data. The result is the `7358/100/1315` ordinary split layout discussed earlier in this report.

Ground truth is also curated through this alignment procedure. After the cleaned train, validation, and test splits are formed, [`prepare_splits.py`](src/data/prepare_splits.py) writes a repository-level [`ground_truth_by_prompt.json`](data/splits/prolog_rev_49fe7b81d6fb3d6e96f39dce4f9b3afc3462e309__openai_cfg_main__openai_rev_cc7b047b6e5bb11b4f1af84efc572db110a51b3c/ground_truth_by_prompt.json) file. The keys are prompt strings and the values are normalized numeric answers extracted from the aligned OpenAI GSM8K solutions.

The PROPER dataset is generated from these already-cleaned ordinary Prolog splits in [`src/data/proper_permute.py`](src/data/proper_permute.py). Only the train split is augmented; validation and test are copied unchanged from the ordinary `gsm8k_prolog` dataset. For each original Prolog program, the repository parses the output into directives, facts, and predicates, then produces candidate reordered variants by permuting fact order, predicate order, and goal order inside predicates. Directive permutation is supported in code but is disabled by default. Every retained PROPER variant is execution-validated against the curated ground truth. Each candidate reordered program is executed with SWI-Prolog, and it is kept only if execution succeeds and the normalized answer matches the answer stored in `ground_truth_by_prompt.json` for the corresponding prompt. Note, that the PROPER dataset was generated after the validation split was extracted from the original training dataset. This prevents the leakage problem, reported in the original paper when training and validation on PROPER.

The retained PROPER artifact used in this repository is the `ratio_1to2` variant, whose manifest is stored at [`data/splits/prolog_rev_49fe7b81d6fb3d6e96f39dce4f9b3afc3462e309__openai_cfg_main__openai_rev_cc7b047b6e5bb11b4f1af84efc572db110a51b3c/gsm8k_proper/ratio_1to2/manifest.json`](data/splits/prolog_rev_49fe7b81d6fb3d6e96f39dce4f9b3afc3462e309__openai_cfg_main__openai_rev_cc7b047b6e5bb11b4f1af84efc572db110a51b3c/gsm8k_proper/ratio_1to2/manifest.json). Starting from `7358` original train examples, the augmentation process adds `13818` validated permuted rows, producing a final PROPER train split of `21176` rows while leaving validation and test at `100` and `1315`, respectively.

Finally, the repository includes an explicit validation utility for the saved PROPER dataset in [`tests/proper_dataset_validator.py`](tests/proper_dataset_validator.py). This script re-executes rows from a saved PROPER dataset, reloads the corresponding `ground_truth_by_prompt.json`, and checks again that every row either executes to the expected answer or is reported as a failure.

## 4. Program Execution Backend

Program execution is handled by the subprocess backend in [`src/prolog/execute.py`](src/prolog/execute.py). The repository writes generated code to a temporary `.pl` file, invokes SWI-Prolog with a one-shot query under the `solve(Result)` contract, and normalizes the returned answer before comparison against ground truth. The same backend is used for data cleaning, PROPER validation, offline evaluation, and RL reward computation. Execution failures are typed into categories such as `syntax_error`, `execution_error`, `no_solution`, `timeout`, and `dependency_error`, which makes it possible to analyze not only whether a program failed, but how it failed.

This execution contract also explains why the retained vanilla evaluation should not be interpreted as a meaningful direct comparison against the adapted models. The original dataset provides only a brief instruction to generate Prolog code and does not include one-shot or few-shot demonstrations that would force a base model to follow the repository's exact `solve(Result)` interface. As a result, the vanilla model has substantial freedom in how it formats its response, and many of its outputs contain answer text, repeated prompt scaffolding, or Prolog-like fragments that do not satisfy the strict executor contract. The vanilla run was therefore kept mainly for qualitative interest: its score is effectively zero under the strict evaluator for this reason, but it still provides a useful view of how the unadapted model behaves under these conditions. A more permissive tool-using or ReAct-style loop could have been attempted, but that would move the current reproduction substantially away from the original paper design and away from the one-shot dataset construction used in this repository.

## 5. Observations on Vanilla Model Outputs

The retained vanilla run in [`outputs/eval/gsm8k_prolog_test_vanilla`](outputs/eval/gsm8k_prolog_test_vanilla) confirms the mismatch described above. The model often produced Prolog-like text, but it rarely produced a complete program that could be executed under the repository's `solve(Result)` interface. In many cases the output mixed a direct answer with tutorial-style scaffolding such as `### Solution`, `### Explanation`, or `### Test`, and in other cases it produced REPL-style fragments such as `?- hotdogs(X).` followed by an answer trace rather than a final standalone program.

The vanilla run is therefore retained not as a competitive baseline, but as a qualitative reference point. It shows that the base model recognizes the broad format of “Prolog-like” responses, yet without adaptation it does not reliably map the repository prompt into the specific executable program contract required by the downstream evaluator.

Here are some of the patterns noticed in the vanilla model output. First, the vanilla model frequently produced outputs that looked superficially appropriate for the task: many generations contained fenced ` ```prolog ` blocks, facts, or simple predicate definitions. However, these outputs often drifted into surrounding scaffolding such as `### Solution`, `### Explanation`, `### Test`, repeated prompt fragments, or direct answer text. Second, the model often behaved as if it were demonstrating Prolog usage rather than returning a final program, producing REPL-style snippets such as `?- predicate(X).` together with an example answer trace.

These patterns suggest that the base model is not completely oblivious to the task format. Rather, it appears to recognize that the target output should be related to Prolog, but it does not reliably infer the stricter repository-specific requirement that the answer must be packaged as a one-shot executable program under the `solve(Result)` interface. This is consistent with the fact that the prompt gives only a brief instruction and does not provide the base model with demonstrations that would force that exact output structure.

## 6. Supervised Fine-Tuning

### 6.1 Brief SFT Methodology

The main reproduced training stage in this repository is supervised fine-tuning with LoRA adapters on top of `mistralai/Mistral-7B-v0.3`. All retained training runs discussed in this report were carried out on an NVIDIA A100 GPU with 80 GB of VRAM. In the ordinary track, the model is trained to map the repository prompt template directly to a complete Prolog program; in the PROPER track, the same objective is used, but the training split is augmented with execution-validated non-sequential Prolog variants. Model selection for the retained SFT runs is based on validation `eval_loss`, and the strongest ordinary SFT checkpoint is later used as the initialization point for the RL extension.

### 6.2 Paperstyle-Close SFT Runs

The retained paperstyle-close SFT runs were designed to remain reasonably close to the paper's reported Mistral setup while still fitting the repository's actual implementation. In particular, they keep the Mistral-family backbone, use LoRA adapters on `q_proj` and `v_proj`, retain `r=32` and `alpha=64`, keep the paper's learning rate of `3e-4`, and match the paper's effective batch size of `128` through `per_device_train_batch_size=8` and `gradient_accumulation_steps=16`. The paperstyle-close ordinary run is stored in [`outputs/training/gsm8k_prolog_paperstyle_close_sft`](outputs/training/gsm8k_prolog_paperstyle_close_sft), and the retained PROPER `1:2` run is stored in [`outputs/training/gsm8k_proper_paperstyle_close_sft`](outputs/training/gsm8k_proper_paperstyle_close_sft).

These runs are nevertheless not exact copies of the paper's setup. First, they use `mistralai/Mistral-7B-v0.3`, whereas the paper reports only "Mistral 7B" without pinning the exact released checkpoint. Second, the retained runs here use a token limit of `2048`, whereas the paper states that examples longer than `512` tokens were removed. Third, the retained paperstyle-close runs were capped at `4` epochs rather than the `6` epochs reported in the paper. This choice is justified by the fact that in the original paper, validation loss did not show meaningful improvement beyond epoch 4 and often peaked earlier. Finally, the retained repository runs select checkpoints by validation loss, whereas the paper's later discussion emphasizes that loss and execution accuracy do not always move together.

### 6.3 Difference in Outcomes Relative to the Paper

The paperstyle-close runs in this repository show earlier overfitting than the paper would suggest, especially for PROPER. For the ordinary Prolog paperstyle-close run, validation loss reached its best value of `0.43997` at [`checkpoint-110`](outputs/training/gsm8k_prolog_paperstyle_close_sft/checkpoint-110), around epoch `1.90`, and the remainder of training stayed slightly worse than that minimum. For the PROPER paperstyle-close run, the effect was much stronger: validation loss reached its best value of `0.44562` at [`checkpoint-130`](outputs/training/gsm8k_proper_paperstyle_close_sft/checkpoint-130), around epoch `0.79`, and then deteriorated steadily to approximately `0.60` by the end of epoch 4.

This behavior differs from the paper's discussion of Mistral on PROPER `1:2`, where overfitting was not reported to emerge that quickly and the validation-loss trend remained favorable much later into training. In the present reproduction, both ordinary Prolog and PROPER converged early under validation loss, and the PROPER run in particular overfit much faster than the paper's Mistral results would lead one to expect. On held-out test evaluation, the retained paperstyle-close ordinary SFT run reached `answer_accuracy = 0.5308` and `exec_ok_rate = 0.7977` in [`outputs/eval/gsm8k_prolog_test_adapter_paperstyle_close/summary.json`](outputs/eval/gsm8k_prolog_test_adapter_paperstyle_close/summary.json), while the retained paperstyle-close PROPER run reached `answer_accuracy = 0.5202` and `exec_ok_rate = 0.7977` in [`outputs/eval/gsm8k_proper_test_adapter_paperstyle_close/summary.json`](outputs/eval/gsm8k_proper_test_adapter_paperstyle_close/summary.json). In this repository, therefore, the retained paperstyle-close PROPER run did not surpass the corresponding ordinary paperstyle-close run.

### 6.4 Best Ordinary SFT Run

The strongest retained ordinary SFT model in this repository is stored in [`outputs/training/gsm8k_prolog_best_sft`](outputs/training/gsm8k_prolog_best_sft). Relative to the paperstyle-close runs, this best SFT configuration changes several parameters: it lowers the learning rate from `3e-4` to `2e-4`, reduces gradient accumulation from `16` to `3`, and uses `per_device_eval_batch_size=4` instead of `8`. The resulting effective batch size is therefore `24` rather than `128`.

This retuned setup yielded the best ordinary supervised model retained in the repository. Its best validation loss was `0.43629` at [`checkpoint-600`](outputs/training/gsm8k_prolog_best_sft/checkpoint-600), and on held-out test evaluation it reached `answer_accuracy = 0.5430`, `exec_ok_rate = 0.7871`, and `answer_accuracy_on_exec_ok = 0.6899` in [`outputs/eval/gsm8k_prolog_test_adapter_best_sft/summary.json`](outputs/eval/gsm8k_prolog_test_adapter_best_sft/summary.json). Relative to the paperstyle-close ordinary SFT run, this best ordinary SFT model improved answer accuracy and correctness conditional on execution, although its raw execution-success rate was slightly lower.

### 6.5 Validation-Loss Figures

![Ordinary SFT validation-loss dynamics](figures/sft_ordinary_eval_loss.png)

*Figure 1. Validation loss over epoch for the retained ordinary SFT runs. The repo-tuned best ordinary SFT run reaches a lower minimum validation loss than the paperstyle-close ordinary run.*

![PROPER SFT validation-loss dynamics](figures/sft_proper_eval_loss.png)

*Figure 2. Validation loss over epoch for the retained paperstyle-close PROPER SFT run. The best checkpoint is reached early, after which validation loss degrades steadily.*

## 7. Reinforcement Learning Extension

### 7.1 RL Setup

The repository extends the paper's supervised setup with a GRPO reinforcement-learning stage implemented in [`src/training/train_rl.py`](src/training/train_rl.py). This stage is not part of the original paper and should therefore be read as an additional experiment built on top of the reproduced SFT stack. Rather than training from the base Mistral model, the RL stage initializes from the best ordinary SFT adapter and continues training only the adapter parameters. Rollouts are generated with vLLM, multiple completions are sampled per prompt, and the resulting Prolog programs are executed with the same SWI-Prolog backend used elsewhere in the repository.

The retained best RL run is stored in [`outputs/training/gsm8k_prolog_best_rl`](outputs/training/gsm8k_prolog_best_rl). Its saved configuration corresponds to `learning_rate = 1e-5`, `num_train_epochs = 1`, `per_device_train_batch_size = 4`, `gradient_accumulation_steps = 6`, `per_device_eval_batch_size = 20`, `num_generations = 4`, `max_prompt_length = 1024`, `max_completion_length = 512`, and `vllm_gpu_memory_utilization = 0.7`. Checkpoint selection is based on validation `eval_prolog_answer_accuracy` rather than on loss alone. Note, that even though the ideal selection of `max_prompt_length` would be 2048 to ensure that all of the exepcted Prolog codes fit, there was only a single test in the training set that was longer than 1024; thus, the expected harm should be negligible.

### 7.2 Reward Design

The RL reward is shaped by both executability and final answer correctness. Correct solutions receive the strongest positive signal, executable-but-wrong programs are treated neutrally, and broken outputs receive negative rewards depending on failure type. In particular, syntax errors, runtime execution errors, timeouts, and empty outputs are penalized, while infrastructure failures such as missing SWI-Prolog are treated as environment problems rather than as valid model outcomes.

### 7.3 Retained RL Outcome

The retained best checkpoint for the RL run is [`checkpoint-400`](outputs/training/gsm8k_prolog_best_rl/checkpoint-400), selected at validation `eval_prolog_answer_accuracy = 0.65` with `eval_prolog_exec_ok_rate = 0.87` on the repository's 100-example validation slice. This best checkpoint emerged relatively early in the single-epoch run, while later checkpoints did not improve on the same validation criterion. The final training artifact reloads this best checkpoint into the root run directory.

On the full held-out `gsm8k_prolog_test` split, the best RL model improved over the best ordinary SFT baseline. The RL model reached `answer_accuracy = 0.5871`, `exec_ok_rate = 0.8046`, and `answer_accuracy_on_exec_ok = 0.7297` in [`outputs/eval/gsm8k_prolog_test_adapter_best_rl/summary.json`](outputs/eval/gsm8k_prolog_test_adapter_best_rl/summary.json). The best ordinary SFT model, by comparison, reached `answer_accuracy = 0.5430`, `exec_ok_rate = 0.7871`, and `answer_accuracy_on_exec_ok = 0.6899` in [`outputs/eval/gsm8k_prolog_test_adapter_best_sft/summary.json`](outputs/eval/gsm8k_prolog_test_adapter_best_sft/summary.json). 

The error breakdown remains consistent with the earlier SFT analysis: syntax errors are still the dominant non-`ok` failure mode. However, the RL run still reduces the syntax-error rate slightly relative to the best ordinary SFT baseline while also improving the final answer metrics.

Overall, the improvement is modest and suggests that RL was not able to dramatically improve model's performance, but this should be interpreted cautiously. First, the retained RL experiment set is small, and the repository does not yet include a broad sweep over reward design or RL hyperparameters. Second, the RL stage updates only the LoRA adapter parameters rather than the full model. Third, it is plausible that a 7B-scale model in this strict one-shot executable-Prolog setting is already closer to its attainable limit than the original SFT results alone might suggest. These possibilities are not mutually exclusive, and the present artifact set is not sufficient to distinguish cleanly between them.

### 7.4 RL Training Figure

![RL training progress](figures/rl_training_progress.png)

*Figure 3. Training dynamics for the retained RL run. The top panel shows held-out validation metrics from the Prolog callback, while the middle and bottom panels show reward and loss trends during training. The dashed line marks the selected best checkpoint at step 400.*

## 8. Conclusion

This repository provides a successful but repository-specific reproduction of the core idea behind arithmetic reasoning via Prolog generation and execution-grounded training. In the retained artifact set, ordinary supervised fine-tuning on `gsm8k_prolog` clearly produces a nontrivial executable-Prolog baseline, and the data-preparation pipeline ensures that both the ordinary and PROPER datasets are grounded in execution-validated answers aligned with the original OpenAI GSM8K targets.

At the same time, the present reproduction does not fully match the original paper. It retains only a Mistral-family track, omits the paper's broader multi-model and GSM-HARD comparisons, and uses a repository-specific cleaned split construction. The retained paperstyle-close SFT runs also show faster overfitting than the paper would suggest, especially for PROPER `1:2`, where the repository results did not reproduce the stronger Mistral behavior described in the original work.

Within this narrower scope, the most important empirical result is that the repository's best ordinary SFT run yields a strong supervised baseline, and the additional GRPO extension improves over that baseline on the full `gsm8k_prolog_test` split, even thought modestly. The vanilla base model, by contrast, remains qualitatively useful only as a reference point for failure analysis under the strict one-shot `solve(Result)` execution contract. Taken together, these results support the main claim that arithmetic word problems can be approached through generated Prolog programs and execution-grounded model adaptation, while also showing that the exact training dynamics and gains are sensitive to the concrete implementation and retained dataset construction used in this repository.
