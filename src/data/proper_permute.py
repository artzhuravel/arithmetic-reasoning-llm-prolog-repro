from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Tuple, Optional, TypeVar, cast
from itertools import permutations
import random

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from src.data.prepare_splits import DATA_SPLITS_DIR, prepare_splits


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_SPLITS_DIR = REPO_ROOT / "data" / "splits"
GSM8K_PROPER_DIR = DATA_SPLITS_DIR / "gsm8k_proper"

PROPER_PERMUTED_INSTRUCTION = (
    "Please generate a piece of Prolog code in non-sequential order "
    "to solve the given math problem."
)


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

        # Start of a rule/predicate
        if (not in_predicate) and (":-" in l) and (not l.lstrip().startswith(":-")):
            in_predicate = True
            cur_predicate = [l]
            # one-line rule ending with "."
            if l.endswith("."):
                predicates.append(cur_predicate)
                cur_predicate = []
                in_predicate = False
            continue

        # Facts: top-level, ends with ".", not indented, and not a rule head
        if (not in_predicate) and (not l.startswith(" ")) and l.endswith(".") and (":-" not in l):
            facts.append(l)
            continue

        # Predicate body lines
        if in_predicate:
            cur_predicate.append(l)
            if l.endswith("."):
                predicates.append(cur_predicate)
                cur_predicate = []
                in_predicate = False
            continue

        # Fallback: treat as fact if it looks like a top-level clause
        if (not l.startswith(" ")) and l.endswith("."):
            facts.append(l)

    return directives, facts, predicates

T = TypeVar("T")


def _sample_permutations(
    items: List[T],
    *,
    max_count: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> List[Tuple[T, ...]]:
    """
    Return up to `max_count` permutations without materializing the full factorial set.
    When capped, we stop early after collecting enough permutations (paper-aligned iterator behavior).
    """
    items_for_perm = list(items)
    if rng is not None and len(items_for_perm) > 1:
        # Shuffle the base order to vary which first-k permutations we observe while staying reproducible.
        rng.shuffle(items_for_perm)

    perm_iter = permutations(items_for_perm)
    if max_count is None:
        return list(perm_iter)

    if max_count <= 0:
        return []

    out: List[Tuple[T, ...]] = []
    for perm in perm_iter:
        out.append(perm)
        if len(out) >= max_count:
            break
    return out


def permute_directives(
    directives: List[str],
    *,
    max_count: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> List[Tuple[str, ...]]:
    return _sample_permutations(directives, max_count=max_count, rng=rng)


def permute_facts(
    facts: List[str],
    *,
    max_count: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> List[Tuple[str, ...]]:
    return _sample_permutations(facts, max_count=max_count, rng=rng)

def _strip_goal_terminator(line: str) -> str:
    s = line.strip()
    if s.endswith(",") or s.endswith("."):
        s = s[:-1].rstrip()
    return s

def _split_goals_top_level(body: str) -> List[str]:
    """Split an inline rule body on top-level commas only."""
    parts: List[str] = []
    cur: List[str] = []
    depth_paren = depth_brace = depth_bracket = 0

    for ch in body:
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket = max(0, depth_bracket - 1)

        if (
            ch == ","
            and depth_paren == 0
            and depth_brace == 0
            and depth_bracket == 0
        ):
            part = "".join(cur).strip()
            if part:
                parts.append(part)
            cur = []
            continue
        cur.append(ch)

    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def get_predicate_variations(
    predicate: List[str],
    *,
    max_variations: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> List[str]:
    if not predicate:
        return []

    head = predicate[0].rstrip()

    # If head already contains ":-", keep it; otherwise add it.
    head_has_colon_dash = ":-" in head and (not head.lstrip().startswith(":-"))
    head_prefix = head if head_has_colon_dash else (head + " :-")

    goals: List[str] = []
    if len(predicate) == 1 and head_has_colon_dash:
        # One-line rule, e.g. "solve(X) :- a(X), b(X)."
        head_part, body_part = head.split(":-", 1)
        head_prefix = head_part.rstrip() + " :-"
        body = _strip_goal_terminator(body_part)
        goals = [g for g in _split_goals_top_level(body) if g]
    else:
        goals = [_strip_goal_terminator(p) for p in predicate[1:]]
        goals = [g for g in goals if g]

    # If parsing yields no goals (e.g. malformed/unsupported clause layout), return original clause.
    if not goals:
        return ["\n".join(predicate).rstrip() + "\n"]

    predicate_variations: List[str] = []
    for perm in _sample_permutations(goals, max_count=max_variations, rng=rng):
        predicate_variations.append(head_prefix + "\n    " + ",\n    ".join(perm) + ".\n")
    return predicate_variations

def get_all_output_variations(
    output: str,
    *,
    max_outputs: int = 10,
    max_component_perms: int = 10,
    seed: Optional[int] = 42,
    permute_directives_too: bool = False,
) -> List[str]:
    """
    Generate up to `max_outputs` semantically equivalent Prolog outputs by permuting:
      - directives order (optional)
      - facts order
      - goal order within each predicate body
      - predicate clause order (if multiple predicates)

    Uses and builds on:
      - get_directives_facts_and_predicates
      - permute_directives
      - permute_facts
      - get_predicate_variations

    Notes:
      - The cartesian product can explode, so we cap each component pool to
        `max_component_perms` and then sample combinations until we have `max_outputs`.
      - Returned outputs are deduplicated by exact string.
    """
    rng = random.Random(seed)

    directives, facts, predicates = get_directives_facts_and_predicates(output)

    # Directives pool
    if permute_directives_too and len(directives) > 1:
        dir_perms = permute_directives(
            directives, max_count=max_component_perms, rng=rng
        )
    else:
        dir_perms = [tuple(directives)]

    # Facts pool
    if len(facts) > 1:
        fact_perms = permute_facts(facts, max_count=max_component_perms, rng=rng)
    else:
        fact_perms = [tuple(facts)]

    # Per-predicate goal-variation pools (each is a list[str] of full predicate clauses)
    predicate_variation_pools: List[List[str]] = []
    for p in predicates:
        vars_for_p = get_predicate_variations(
            p, max_variations=max_component_perms, rng=rng
        )
        predicate_variation_pools.append(vars_for_p)

    # Predicate-order pool (permutations of predicate indices)
    pred_count = len(predicate_variation_pools)
    if pred_count <= 1:
        pred_orders = [tuple(range(pred_count))]
    else:
        pred_orders = _sample_permutations(
            list(range(pred_count)),
            max_count=max_component_perms,
            rng=rng,
        )

    def assemble_one(dirs: Tuple[str, ...], fcts: Tuple[str, ...], pred_clauses_in_order: List[str]) -> str:
        parts: List[str] = []

        if dirs:
            parts.append("\n".join(dirs))
        if fcts:
            # separate directives and facts by a blank line if both exist
            if parts:
                parts.append("")
            parts.append("\n".join(fcts))
        if pred_clauses_in_order:
            if parts:
                parts.append("")
            # predicates themselves already end with "\n" in get_predicate_variations,
            # but we normalize spacing: separate clauses by one blank line.
            norm_preds = [pc.rstrip("\n") for pc in pred_clauses_in_order]
            parts.append("\n\n".join(norm_preds))

        out = "\n".join(parts).rstrip() + "\n"
        return out

    # If there are no predicates, we can still return directive/fact permutations
    results: List[str] = []
    seen: set[str] = set()
    if pred_count == 0:
        for dirs in dir_perms:
            for fcts in fact_perms:
                s = assemble_one(tuple(dirs), tuple(fcts), [])
                if s not in seen:
                    seen.add(s)
                    results.append(s)
                if len(results) >= max_outputs:
                    return results
        return results

    # Sample full combinations without exploding.
    # Put a conservative cap on attempts to avoid pathological loops in high-duplication cases.
    max_attempts = max_outputs * 500

    attempts = 0
    while len(results) < max_outputs and attempts < max_attempts:
        attempts += 1

        dirs = rng.choice(dir_perms)
        fcts = rng.choice(fact_perms)
        order = rng.choice(pred_orders)

        chosen_preds: List[str] = []
        for idx in order:
            pool = predicate_variation_pools[idx]
            chosen_preds.append(rng.choice(pool))

        s = assemble_one(tuple(dirs), tuple(fcts), chosen_preds)
        if s in seen:
            continue
        seen.add(s)
        results.append(s)

    # If sampling didn't fill, do a small deterministic sweep as a fallback.
    if len(results) < max_outputs:
        for dirs in dir_perms:
            for fcts in fact_perms:
                for order in pred_orders:
                    # choose first variation per predicate for deterministic coverage
                    chosen_preds = [predicate_variation_pools[i][0] for i in order]
                    s = assemble_one(tuple(dirs), tuple(fcts), chosen_preds)
                    if s not in seen:
                        seen.add(s)
                        results.append(s)
                    if len(results) >= max_outputs:
                        return results

    return results


def _canonicalize_output(output: str) -> str:
    return output.strip()


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
    Load leakage-safe prepared splits from disk if present; otherwise create them via prepare_splits().

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
    Generate PROPER rows (train-only) and return them as plain dict rows.
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
        original_canonical = _canonicalize_output(original_output)

        variations = get_all_output_variations(
            original_output,
            max_outputs=requested_candidates,
            max_component_perms=max_component_perms,
            seed=seed + idx,
            permute_directives_too=permute_directives_too,
        )

        # Remove duplicates and the identity/original sample.
        unique_candidates: list[str] = []
        seen: set[str] = set()
        for v in variations:
            canon = _canonicalize_output(v)
            if canon == original_canonical or canon in seen:
                continue
            seen.add(canon)
            unique_candidates.append(v)

        if not unique_candidates:
            continue

        local_rng = random.Random(seed + idx)
        if len(unique_candidates) > permutations_per_sample:
            chosen_outputs = local_rng.sample(unique_candidates, k=permutations_per_sample)
        else:
            chosen_outputs = unique_candidates

        for perm_i, permuted_output in enumerate(chosen_outputs):
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
