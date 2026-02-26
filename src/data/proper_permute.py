from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, List, Tuple, Optional, TypeVar, cast

from itertools import permutations, product
import random

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from src.data.prepare_splits import DATA_SPLITS_DIR, prepare_splits
from src.prolog.execute import execute_solve, normalize_answer_for_eval


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_SPLITS_DIR = REPO_ROOT / "data" / "splits"
GSM8K_PROPER_DIR = DATA_SPLITS_DIR / "gsm8k_proper"

PROPER_PERMUTED_INSTRUCTION = (
    "Please generate a piece of Prolog code in non-sequential order "
    "to solve the given math problem."
)
_GSM8K_FINAL_ANSWER_RE = re.compile(r"####\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)")

T = TypeVar("T")


def get_directives_facts_and_predicates(output: str) -> Tuple[List[str], List[str], List[List[str]]]:
    directives: List[str] = []
    facts: List[str] = []
    predicates: List[List[str]] = []

    cur_predicate: List[str] = []
    in_predicate = False

    for raw in output.split("\n"):
        l = raw.rstrip()
        if not l.strip():
            continue

        # Directives
        if l.lstrip().startswith(":-"):
            directives.append(l)
            continue
        
        # Predicate body lines
        if in_predicate:
            cur_predicate.append(l)
            if l.endswith("."):
                predicates.append(cur_predicate)
                cur_predicate = []
                in_predicate = False
            continue

        # Start of a rule/predicate
        if ":-" in l:
            in_predicate = True
            cur_predicate = [l]
            # one-line rule ending with "."
            if l.endswith("."):
                predicates.append(cur_predicate)
                cur_predicate = []
                in_predicate = False
            continue

        # Facts: top-level, ends with ".", not indented, and not a rule head
        if l.endswith("."):
            facts.append(l)
            continue

    return directives, facts, predicates


def _sample_permutations(
    items: List[T],
    *,
    max_count: int | float = float("inf"),
    rng: random.Random = random.Random(42),
) -> List[Tuple[T, ...]]:
    """
    Return up to `max_count` permutations without materializing the full factorial set.
    When capped, we stop early after collecting enough permutations.
    """
    items_for_perm = list(items)
    # Shuffle the base order to vary which first-k permutations we observe.
    rng.shuffle(items_for_perm)

    out: List[Tuple[T, ...]] = []
    for perm in permutations(items_for_perm):
        if len(out) >= max_count:
            return out
        out.append(perm)
    return out


def permute_directives(
    directives: List[str],
    *,
    max_count: int | float = float("inf"),
    rng: random.Random = random.Random(42),
) -> List[Tuple[str, ...]]:
    return _sample_permutations(directives, max_count=max_count, rng=rng)


def permute_facts(
    facts: List[str],
    *,
    max_count: int | float = float("inf"),
    rng: random.Random = random.Random(42),
) -> List[Tuple[str, ...]]:
    return _sample_permutations(facts, max_count=max_count, rng=rng)


def _strip_goal_terminator(line: str) -> str:
    if line.endswith(",") or line.endswith("."):
        return line[:-1].rstrip()
    return line


def get_predicate_variations(
    predicate: List[str],
    *,
    max_variations: int | float = float("inf"),
    rng: random.Random = random.Random(42),
) -> List[str]:
    if not predicate:
        return []

    head = predicate[0].rstrip()
    goals = [_strip_goal_terminator(p) for p in predicate[1:]]

    # If parsing yields no goals (e.g. malformed/unsupported clause layout), return original clause.
    if not goals:
        return ["\n".join(predicate).rstrip() + "\n"]

    predicate_variations: List[str] = []
    for perm in _sample_permutations(goals, max_count=max_variations, rng=rng):
        predicate_variations.append(head + "\n    " + ",\n    ".join(perm) + ".\n")
    return predicate_variations


def get_all_output_variations(
    output: str,
    *,
    max_outputs: int | float  = float("inf"),
    max_component_perms: int | float = float("inf"),
    max_predicate_variations: int | float = float("inf"),
    seed: int = 42,
    permute_directives_too: bool = False,
    correct_answer: Optional[Any] = None,
) -> List[str]:
    """
    Generate up to `max_outputs` semantically equivalent Prolog outputs by permuting:
      - directives order (optional; disabled by default)
      - facts order
      - predicate clause order (if multiple predicates)

    Uses and builds on:
      - get_directives_facts_and_predicates
      - permute_facts
      - permute_predicates

    Notes:
      - The cartesian product can explode, so we cap each component pool to
        `max_component_perms` and then sample combinations until we have `max_outputs`.
      - Returned outputs are deduplicated by exact string.
      - If `correct_answer` is provided, candidate permutations are executed and
        only kept when the result matches the expected answer.
    """
    if max_outputs <= 0 or max_component_perms <= 0:
        return []

    rng = random.Random(seed)
    original_canonical = output.strip()
    expected_answer_normalized = (
        normalize_answer_for_eval(correct_answer) if correct_answer is not None else None
    )

    directives, facts, predicates = get_directives_facts_and_predicates(output)

    # Directives pool (optional)
    if permute_directives_too and len(directives) > 1:
        directive_perms = permute_directives(
            directives, max_count=max_component_perms, rng=rng
        )
    else:
        directive_perms = [tuple(directives)]

    # Facts pool
    fact_perms = permute_facts(facts, max_count=max_component_perms, rng=rng)

    # Per-predicate goal-variation pools (each is a list[str] of full predicate clauses)
    predicate_perms = _sample_permutations(list(range(len(predicates))), max_count=max_component_perms, rng=rng)
    
    predicate_vars: List[List[str]] = []
    for predicate in predicates:
        predicate_vars.append(get_predicate_variations(predicate, max_variations=max_predicate_variations, rng=rng))

    def assemble_output(directives: Tuple[str, ...], facts: Tuple[str, ...], predicates: Tuple[str, ...]) -> str:
        parts: List[str] = []

        if directives:
            parts.append("\n".join(directives) + "\n")
        if facts:
            parts.append("\n".join(facts) + "\n")
        if predicates:
            parts.append("\n".join(predicates))

        out = "\n".join(parts).rstrip() + "\n"
        return out


    
    # Precompute predicate variation combos once per predicate order
    predicate_combo_sets: List[Tuple[Tuple[str, ...], ...]] = []
    for order in predicate_perms:
        predicate_vars_for_perm = [predicate_vars[k] for k in order]
        all_combos = tuple(product(*predicate_vars_for_perm))
        predicate_combo_sets.append(all_combos)

    # Then build full combinations
    full_perm_pairs: List[Tuple[int, int, Tuple[str, ...]]] = []
    for d in range(len(directive_perms)):
        for i in range(len(fact_perms)):
            for all_combos in predicate_combo_sets:
                for predicate_perm in all_combos:
                    full_perm_pairs.append((d, i, predicate_perm))
            
    rng.shuffle(full_perm_pairs)
    
    results: List[str] = []
    canonical_output = output.strip()
    for d, i, predicates in full_perm_pairs:
        if len(results) >= max_outputs:
            return results
        directives = directive_perms[d]
        facts = fact_perms[i]
        s = assemble_output(tuple(directives), tuple(facts), tuple(predicates))
        if s.strip() == canonical_output:
            continue
        if expected_answer_normalized is not None:
            exec_result = execute_solve(s)
            if (not exec_result.ok or 
                exec_result.normalized_answer != expected_answer_normalized):
                continue
        results.append(s)

    return results


def _canonicalize_output(output: str) -> str:
    return output.strip()


def _extract_openai_gsm8k_final_answer(answer_text: str) -> str:
    """
    Extract the final GSM8K answer from the OpenAI format (`...\\n#### <number>`)
    and normalize it to the Prolog-eval comparison format.
    """
    m = _GSM8K_FINAL_ANSWER_RE.search(answer_text)
    if m is None:
        raise ValueError("Could not find final GSM8K answer marker ('#### <number>').")
    raw = m.group(1).replace(",", "")
    return normalize_answer_for_eval(raw)


def get_gsm8k_prolog_train_correct_answers() -> List[str]:
    """
    Return normalized gold answers aligned by index to `gsm8k_prolog/train`,
    extracted from `openai_gsm8k/train`.
    """
    prolog_train_path = DATA_SPLITS_DIR / "gsm8k_prolog" / "train"
    openai_train_path = DATA_SPLITS_DIR / "openai_gsm8k" / "train"

    if not (prolog_train_path.exists() and openai_train_path.exists()):
        # Ensure prepared aligned splits exist, then read them from disk.
        prepare_splits()

    prolog_train = cast(Dataset, load_from_disk(str(prolog_train_path)))
    openai_train = cast(Dataset, load_from_disk(str(openai_train_path)))

    if len(prolog_train) != len(openai_train):
        raise ValueError(
            "Prepared train splits are not aligned by length: "
            f"gsm8k_prolog/train={len(prolog_train)} vs openai_gsm8k/train={len(openai_train)}"
        )

    answers: List[str] = []
    for i in range(len(prolog_train)):
        # Lightweight alignment guard (prepare_splits should already guarantee this).
        if str(prolog_train[i]["input"]).strip() != str(openai_train[i]["question"]).strip():
            raise ValueError(f"Train split question mismatch at index {i}.")
        answers.append(_extract_openai_gsm8k_final_answer(str(openai_train[i]["answer"])))
    return answers


def _ratio_dir_name(permutations_per_sample: int) -> str:
    return f"ratio_1to{permutations_per_sample}"


def _ensure_prepared_gsm8k_prolog_splits(
    *,
    train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    validation_size: Optional[int] = None,
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Load prepared splits from disk if present; otherwise create them via prepare_splits().

    Returns only the Prolog splits: (train, val, test).
    """
    train_path = DATA_SPLITS_DIR / "gsm8k_prolog" / "train"
    val_path = DATA_SPLITS_DIR / "gsm8k_prolog" / "val"
    test_path = DATA_SPLITS_DIR / "gsm8k_prolog" / "test"

    if train_path.exists() and val_path.exists() and test_path.exists():
        if (
            train_size is not None
            or test_size is not None
            or validation_size is not None
            or seed != 42
        ):
            raise ValueError(
                "Prepared splits already exist in a non-parameterized cache path "
                f"({DATA_SPLITS_DIR}). Custom train/test/validation sizes or seed cannot be "
                "applied safely from here. Delete/rebuild `data/splits` first, or version your split paths."
            )
        return (
            cast(Dataset, load_from_disk(str(train_path))),
            cast(Dataset, load_from_disk(str(val_path))),
            cast(Dataset, load_from_disk(str(test_path))),
        )

    gsm8k_prolog_splits, _openai_gsm8k_splits = prepare_splits(
        train_size=train_size,
        test_size=test_size,
        validation_size=validation_size,
        seed=seed,
    )
    return (
        cast(Dataset, gsm8k_prolog_splits["train"]),
        cast(Dataset, gsm8k_prolog_splits["val"]),
        cast(Dataset, gsm8k_prolog_splits["test"]),
    )


def _build_permuted_rows(
    train_split: Dataset,
    *,
    permutations_per_sample: int,
    max_component_perms: int,
    max_outputs_per_sample: Optional[int],
    seed: int,
    permute_directives_too: bool,
) -> list[dict[str, Any]]:
    """
    Generate PROPER rows nd return them as plain dict rows.
    Original rows are NOT included here.
    """
    generated_rows: list[dict[str, Any]] = []

    if permutations_per_sample <= 0:
        return generated_rows

    # We ask for a few extra candidates to survive dedup/original filtering.
    requested_candidates = max_outputs_per_sample or max(
        permutations_per_sample * 4, permutations_per_sample + 2
    )

    for idx in range(len(train_split)):
        row = train_split[idx]
        original_output = row["output"]

        variations = get_all_output_variations(
            original_output,
            max_outputs=requested_candidates,
            max_component_perms=max_component_perms,
            seed=seed + idx,
            permute_directives_too=permute_directives_too,
        )

        if len(variations) > permutations_per_sample:
            local_rng = random.Random(seed + idx)
            chosen_outputs = local_rng.sample(variations, k=permutations_per_sample)
        else:
            chosen_outputs = variations

        for permuted_output in chosen_outputs:
            new_row = dict(row)
            new_row["instruction"] = PROPER_PERMUTED_INSTRUCTION
            new_row["output"] = permuted_output
            generated_rows.append(new_row)

    return generated_rows


def build_gsm8k_proper_dataset(
    *,
    permutations_per_sample: int = 1,
    train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    validation_size: Optional[int] = None,
    seed: int = 42,
    max_component_perms: int = 10,
    max_outputs_per_sample: Optional[int] = None,
    permute_directives_too: bool = False,
    shuffle_augmented_train: bool = True,
    overwrite: bool = False,
) -> DatasetDict:
    """
    Build a PROPER dataset variant and save it under data/splits/gsm8k_proper/.

    - Augments only the train split with permuted Prolog outputs.
    - Keeps val/test unchanged for leakage-safe evaluation.
    - Uses the paper's distinct instruction text for permuted samples.
    """
    ratio_dir = GSM8K_PROPER_DIR / _ratio_dir_name(permutations_per_sample)
    manifest_path = ratio_dir / "manifest.json"

    if ratio_dir.exists() and not overwrite:
        print(f"Loading existing PROPER dataset from {ratio_dir}")
        return cast(DatasetDict, load_from_disk(str(ratio_dir)))

    prolog_train, prolog_val, prolog_test = _ensure_prepared_gsm8k_prolog_splits(
        train_size=train_size,
        test_size=test_size,
        validation_size=validation_size,
        seed=seed,
    )

    permuted_rows = _build_permuted_rows(
        prolog_train,
        permutations_per_sample=permutations_per_sample,
        max_component_perms=max_component_perms,
        max_outputs_per_sample=max_outputs_per_sample,
        seed=seed,
        permute_directives_too=permute_directives_too,
    )

    if permuted_rows:
        permuted_train = Dataset.from_list(permuted_rows)
        augmented_train = cast(
            Dataset, concatenate_datasets([prolog_train, permuted_train])
        )
    else:
        augmented_train = prolog_train

    if shuffle_augmented_train:
        augmented_train = cast(Dataset, augmented_train.shuffle(seed=seed))

    proper_ds = DatasetDict(
        {
            "train": augmented_train,
            "val": prolog_val,
            "test": prolog_test,
        }
    )

    ratio_dir.parent.mkdir(parents=True, exist_ok=True)
    if ratio_dir.exists() and overwrite:
        # Datasets save_to_disk expects a non-existing directory in many cases.
        import shutil

        shutil.rmtree(ratio_dir)

    proper_ds.save_to_disk(str(ratio_dir))

    manifest = {
        "dataset_name": "gsm8k_proper",
        "ratio": _ratio_dir_name(permutations_per_sample),
        "permutations_per_sample": permutations_per_sample,
        "seed": seed,
        "max_component_perms": max_component_perms,
        "max_outputs_per_sample": max_outputs_per_sample,
        "permute_directives_too": permute_directives_too,
        "shuffle_augmented_train": shuffle_augmented_train,
        "counts": {
            "train_original": len(prolog_train),
            "train_permuted_added": len(permuted_rows),
            "train_final": len(augmented_train),
            "val": len(prolog_val),
            "test": len(prolog_test),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        "Saved PROPER dataset to "
        f"{ratio_dir} (train {len(prolog_train)} + {len(permuted_rows)} -> {len(augmented_train)})"
    )
    return proper_ds
