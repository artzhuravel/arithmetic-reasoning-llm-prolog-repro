from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
from pathlib import Path
from typing import Any, List, Tuple, Optional, TypeVar, cast

from itertools import permutations, product
import random

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from src.data.prepare_splits import get_default_splits_dir, prepare_splits
from src.prolog.execute import execute_solve, normalize_answer_for_eval

def _noop_tqdm(iterable: Any, **_: Any) -> Any:
    return iterable

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = _noop_tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_SPLITS_DIR = get_default_splits_dir()
GSM8K_PROPER_DIR = DATA_SPLITS_DIR / "gsm8k_proper"

PROPER_PERMUTED_INSTRUCTION = (
    "Please generate a piece of Prolog code in non-sequential order "
    "to solve the given math problem."
)
_GSM8K_FINAL_ANSWER_RE = re.compile(r"####\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)")

T = TypeVar("T")
GROUND_TRUTH_FILENAME = "ground_truth_by_prompt.json"


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
    max_outputs: Optional[int | float] = float("inf"),
    max_intra_component_perms: int | float = float("inf"),
    max_intra_predicate_perms: int | float = float("inf"),
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
        `max_intra_component_perms` and then sample combinations until we have `max_outputs`.
      - Returned outputs are deduplicated by exact string.
      - If `correct_answer` is provided, candidate permutations are executed and
        only kept when the result matches the expected answer.
    """
    
    if max_intra_component_perms <= 0 or max_intra_predicate_perms <= 0:
        return []
    if max_outputs is not None and max_outputs <= 0:
        return []

    rng = random.Random(seed)
    expected_answer_normalized = (
        normalize_answer_for_eval(correct_answer) if correct_answer is not None else None
    )

    directives, facts, predicates_clauses = get_directives_facts_and_predicates(output)

    # Directives pool (optional)
    if permute_directives_too and len(directives) > 1:
        directive_perms = permute_directives(
            directives, max_count=max_intra_component_perms, rng=rng
        )
    else:
        directive_perms = [tuple(directives)]

    # Facts pool
    fact_perms = permute_facts(facts, max_count=max_intra_component_perms, rng=rng)

    # Get inter-predicate permutations
    predicate_perms = _sample_permutations(list(range(len(predicates_clauses))), max_count=max_intra_component_perms, rng=rng)
    
    predicate_vars: List[List[str]] = [] # Per each predicate clause, we have a list of variations
    for predicate in predicates_clauses:
        predicate_vars.append(get_predicate_variations(predicate, max_variations=max_intra_predicate_perms, rng=rng))

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
    
    # Precompute all possible predicate combinations
    predicates_perms: List[Tuple[str, ...]] = []
    for predicate_order in predicate_perms:
        predicate_vars_for_perm = [predicate_vars[i] for i in predicate_order]
        predicates_perms.extend(product(*predicate_vars_for_perm))

    # Then build full combinations
    n_facts_permutations = len(fact_perms)
    n_predicates_permutations = len(predicates_perms)
    
    full_perm_pairs: set[Tuple[int, int, int]] = set()
    for d in range(len(directive_perms)):
        for f in range(n_facts_permutations):
            for p in range(n_predicates_permutations):
                full_perm_pairs.add((d, f, p))
    
    results: List[str] = []
    canonical_output = output.strip()
    
    if max_outputs is None or max_outputs == float("inf"):
        need_outputs = len(full_perm_pairs)
    else:
        need_outputs = max_outputs

    while full_perm_pairs and need_outputs > 0:
        k = int(min(need_outputs, len(full_perm_pairs)))
        picked = rng.sample(list(full_perm_pairs), k=k)
        
        for d, f, p in picked:
            s = assemble_output(tuple(directive_perms[d]), tuple(fact_perms[f]), tuple(predicates_perms[p]))
            if s.strip() == canonical_output:
                continue
            if expected_answer_normalized is not None:
                exec_result = execute_solve(s)
                if (not exec_result.ok or 
                    exec_result.normalized_answer != expected_answer_normalized):
                    continue
            need_outputs -= 1
            results.append(s)
        
        full_perm_pairs.difference_update(picked)

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


def _load_ground_truth_by_prompt(
    *,
    splits_dir: Optional[Path] = None,
) -> dict[str, float]:
    base_dir = splits_dir if splits_dir is not None else DATA_SPLITS_DIR
    ground_truth_path = base_dir / GROUND_TRUTH_FILENAME

    if not ground_truth_path.exists():
        if splits_dir is not None:
            raise FileNotFoundError(
                f"Ground-truth file not found in {splits_dir}: {GROUND_TRUTH_FILENAME}"
            )
        prepare_splits()
        if not ground_truth_path.exists():
            raise FileNotFoundError(
                f"Ground-truth file was not created at {ground_truth_path}."
            )

    payload = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    all_map = payload.get("all")
    if not isinstance(all_map, dict):
        raise ValueError(
            f"Invalid ground-truth format in {ground_truth_path}: expected top-level 'all' mapping."
        )

    normalized: dict[str, float] = {}
    for prompt, value in all_map.items():
        if not isinstance(prompt, str):
            raise ValueError("Ground-truth keys must be prompt strings.")
        try:
            normalized[prompt.strip()] = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Ground-truth value must be numeric for prompt {prompt[:80]!r}."
            ) from e
    return normalized


def get_gsm8k_prolog_train_correct_answers(
    splits_dir: Optional[Path] = None,
) -> List[str]:
    """
    Return normalized gold answers aligned by index to `gsm8k_prolog/train`,
    extracted from `openai_gsm8k/train`.

    Args:
        splits_dir: Directory containing gsm8k_prolog/ and openai_gsm8k/.
            If None, uses default versioned splits dir.
    """
    base_dir = splits_dir if splits_dir is not None else DATA_SPLITS_DIR
    prolog_train_path = base_dir / "gsm8k_prolog" / "train"
    if not prolog_train_path.exists():
        if splits_dir is not None:
            raise FileNotFoundError(
                f"Train split not found in {splits_dir}. Expected gsm8k_prolog/train."
            )
        prepare_splits()

    prolog_train = cast(Dataset, load_from_disk(str(prolog_train_path)))
    ground_truth_by_prompt = _load_ground_truth_by_prompt(splits_dir=splits_dir)

    answers: List[str] = []
    for i in range(len(prolog_train)):
        prompt = str(prolog_train[i]["input"]).strip()
        if prompt not in ground_truth_by_prompt:
            raise ValueError(f"Missing ground-truth answer for train prompt at index {i}.")
        answers.append(normalize_answer_for_eval(ground_truth_by_prompt[prompt]))
    return answers


def _ratio_dir_name(permutations_per_sample: int) -> str:
    return f"ratio_1to{permutations_per_sample}"


def _ensure_prepared_gsm8k_prolog_splits(
    *,
    splits_dir: Optional[Path] = None,
    train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    validation_size: Optional[int] = None,
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Load prepared splits from disk if present; otherwise create them via prepare_splits().

    Args:
        splits_dir: Directory containing gsm8k_prolog/ and openai_gsm8k/. If None, uses default.
        When provided, loads only from disk (no prepare_splits fallback).

    Returns only the Prolog splits: (train, val, test).
    """
    base_dir = splits_dir if splits_dir is not None else DATA_SPLITS_DIR
    train_path = base_dir / "gsm8k_prolog" / "train"
    val_path = base_dir / "gsm8k_prolog" / "val"
    test_path = base_dir / "gsm8k_prolog" / "test"

    if train_path.exists() and val_path.exists() and test_path.exists():
        if splits_dir is not None:
            # Explicit splits_dir: load from disk only, no custom sizes/seed
            if train_size is not None or test_size is not None or validation_size is not None or seed != 42:
                raise ValueError(
                    "When splits_dir is provided, train_size/test_size/validation_size/seed "
                    "cannot be customized. Use the splits as-is."
                )
        elif train_size is not None or test_size is not None or validation_size is not None or seed != 42:
            raise ValueError(
                "Prepared splits already exist in a non-parameterized cache path "
                f"({base_dir}). Custom train/test/validation sizes or seed cannot be "
                "applied safely from here. Delete/rebuild or use a different splits_dir."
            )
        return (
            cast(Dataset, load_from_disk(str(train_path))),
            cast(Dataset, load_from_disk(str(val_path))),
            cast(Dataset, load_from_disk(str(test_path))),
        )

    if splits_dir is not None:
        raise FileNotFoundError(
            f"Splits not found in {splits_dir}. Expected gsm8k_prolog/{{train,val,test}}. "
            "Run prepare_splits first for the desired versions."
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
    ground_truth_by_prompt: dict[str, float],
    permutations_per_sample: int,
    max_intra_component_perms: Optional[int],
    max_intra_predicate_perms: Optional[int],
    max_outputs_per_sample: Optional[int],
    seed: int,
    permute_directives_too: bool,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """
    Generate PROPER rows nd return them as plain dict rows.
    Original rows are NOT included here.
    """
    generated_rows: list[dict[str, Any]] = []

    if permutations_per_sample <= 0:
        return generated_rows
    if workers <= 0:
        raise ValueError("workers must be greater than 0")

    indexed_rows: list[tuple[int, dict[str, Any]]] = [
        (idx, dict(train_split[idx])) for idx in range(len(train_split))
    ]

    def _build_row_variants(item: tuple[int, dict[str, Any]]) -> list[dict[str, Any]]:
        idx, row = item
        prompt = str(row["input"]).strip()
        if prompt not in ground_truth_by_prompt:
            raise ValueError(
                f"Missing ground-truth answer for train prompt at index {idx}."
            )
        correct_answer = ground_truth_by_prompt[prompt]
        original_output = row["output"]

        variations = get_all_output_variations(
            original_output,
            max_outputs=max_outputs_per_sample if max_outputs_per_sample is not None else float("inf"),
            max_intra_component_perms=max_intra_component_perms if max_intra_component_perms is not None else float("inf"),
            max_intra_predicate_perms=max_intra_predicate_perms if max_intra_predicate_perms is not None else float("inf"),
            seed=seed + idx,
            permute_directives_too=permute_directives_too,
            correct_answer=correct_answer,
        )

        if len(variations) > permutations_per_sample:
            local_rng = random.Random(seed + idx)
            chosen_outputs = local_rng.sample(variations, k=permutations_per_sample)
        else:
            chosen_outputs = variations

        out_rows: list[dict[str, Any]] = []
        for permuted_output in chosen_outputs:
            new_row = dict(row)
            new_row["instruction"] = PROPER_PERMUTED_INSTRUCTION
            new_row["output"] = permuted_output
            out_rows.append(new_row)

        return out_rows

    if workers == 1:
        for item in tqdm(indexed_rows, total=len(indexed_rows), desc="Build PROPER rows", unit="row"):
            generated_rows.extend(_build_row_variants(item))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            mapped = executor.map(_build_row_variants, indexed_rows)
            for row_variants in tqdm(
                mapped, total=len(indexed_rows), desc="Build PROPER rows", unit="row"
            ):
                generated_rows.extend(row_variants)

    return generated_rows


def build_gsm8k_proper_dataset(
    *,
    permutations_per_sample: int = 2,
    train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    validation_size: Optional[int] = None,
    seed: int = 42,
    max_intra_component_perms: Optional[int] = 10,
    max_intra_predicate_perms: Optional[int] = 10,
    max_outputs_per_sample: Optional[int] = 100,
    permute_directives_too: bool = False,
    shuffle_augmented_train: bool = True,
    workers: int = 10,
    overwrite: bool = False,
    splits_dir: Optional[Path] = None,
) -> DatasetDict:
    """
    Build a PROPER dataset variant and save it under splits_dir/gsm8k_proper/.

    - Augments only the train split with permuted Prolog outputs.
    - Keeps val/test unchanged for leakage-safe evaluation.
    - Uses the paper's distinct instruction text for permuted samples.

    Args:
        splits_dir: Directory containing gsm8k_prolog/ and openai_gsm8k/ splits
            (versioned by prolog/openai revisions). If None, uses default.
            PROPER output is saved under splits_dir/gsm8k_proper/ratio_1toN/.
    """
    base_dir = splits_dir if splits_dir is not None else DATA_SPLITS_DIR
    proper_dir = base_dir / "gsm8k_proper"
    ratio_dir = proper_dir / _ratio_dir_name(permutations_per_sample)
    manifest_path = ratio_dir / "manifest.json"

    if ratio_dir.exists() and not overwrite:
        print(f"Loading existing PROPER dataset from {ratio_dir}")
        return cast(DatasetDict, load_from_disk(str(ratio_dir)))

    prolog_train, prolog_val, prolog_test = _ensure_prepared_gsm8k_prolog_splits(
        splits_dir=splits_dir,
        train_size=train_size,
        test_size=test_size,
        validation_size=validation_size,
        seed=seed,
    )
    ground_truth_by_prompt = _load_ground_truth_by_prompt(splits_dir=splits_dir)

    permuted_rows = _build_permuted_rows(
        prolog_train,
        ground_truth_by_prompt=ground_truth_by_prompt,
        permutations_per_sample=permutations_per_sample,
        max_intra_component_perms=max_intra_component_perms,
        max_intra_predicate_perms=max_intra_predicate_perms,
        max_outputs_per_sample=max_outputs_per_sample,
        seed=seed,
        permute_directives_too=permute_directives_too,
        workers=workers,
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
        "splits_dir": str(base_dir),
        "ratio": _ratio_dir_name(permutations_per_sample),
        "permutations_per_sample": permutations_per_sample,
        "seed": seed,
        "max_intra_component_perms": max_intra_component_perms,
        "max_intra_predicate_perms": max_intra_predicate_perms,
        "max_outputs_per_sample": max_outputs_per_sample,
        "permute_directives_too": permute_directives_too,
        "shuffle_augmented_train": shuffle_augmented_train,
        "workers": workers,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build PROPER-augmented GSM8K-Prolog dataset from prepared splits."
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        help="Directory with gsm8k_prolog/ and openai_gsm8k/ (versioned). Default: default version.",
    )
    parser.add_argument("--permutations-per-sample", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--validation-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-intra-component-perms", type=int, default=10)
    parser.add_argument("--max-intra-predicate-perms", type=int, default=10)
    parser.add_argument("--max-outputs-per-sample", type=int, default=100)
    parser.add_argument("--permute-directives-too", action="store_true")
    parser.add_argument("--no-shuffle-augmented-train", action="store_true")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    build_gsm8k_proper_dataset(
        permutations_per_sample=args.permutations_per_sample,
        train_size=args.train_size,
        test_size=args.test_size,
        validation_size=args.validation_size,
        seed=args.seed,
        max_intra_component_perms=args.max_intra_component_perms,
        max_intra_predicate_perms=args.max_intra_predicate_perms,
        max_outputs_per_sample=args.max_outputs_per_sample,
        permute_directives_too=args.permute_directives_too,
        shuffle_augmented_train=not args.no_shuffle_augmented_train,
        workers=args.workers,
        overwrite=args.overwrite,
        splits_dir=args.splits_dir,
    )
