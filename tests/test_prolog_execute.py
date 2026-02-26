"""
Test Prolog execution and float/decimal handling via `swipl` subprocess.

Paper (Yang et al., NAACL 2024) states: "since we noticed that the PySwip library
cannot handle decimal answers, we only considered the samples with an integer answer."

This test verifies:
1. Basic Prolog execution works
2. Integer results normalization (14 vs 14.0 handling for eval)
3. Decimal/division with clpq now works via subprocess backend
4. Answer normalization for evaluation (14 vs "14" vs 14.0)
"""

import shutil
import pytest

from src.prolog.execute import execute_solve, execute_prolog_string, PrologExecutionResult


# Simple integer result (paper Figure 1 example)
CODE_INTEGER = """
:- use_module(library(clpq)).
age_diff(raymond, samantha, 6).
age(raymond, 23).
age(samantha, 31).
solve(Years_ago) :-
    age_diff(raymond, samantha, Age_diff),
    age(raymond, Raymond_age),
    age(samantha, Samantha_age),
    {Raymond_son_age = Raymond_age - Age_diff},
    {Years_ago = Samantha_age - Raymond_son_age}.
"""

# Division that yields integer (12 * 50 / 60 = 10)
CODE_DIVISION_INTEGER = """
:- use_module(library(clpq)).
earn(weng, 12).
work(weng, 50).
solve(Total_salary) :-
    earn(weng, Salary_per_hour),
    work(weng, Working_minutes),
    {Total_salary = Salary_per_hour * Working_minutes / 60}.
"""

# Division that yields non-integer (e.g. 10 / 3 = 3.333...)
CODE_DECIMAL = """
:- use_module(library(clpq)).
solve(Result) :-
    {Result = 10 / 3}.
"""

# Simple integer without clpq
CODE_SIMPLE_INT = """
solve(Result) :-
    Result is 14 + 0.
"""

CODE_SYNTAX_ERROR = """
:- use_module(library(clpq)).
solve(X) :-
    {X = 1 + 2}
"""


def _answer_str(result: PrologExecutionResult) -> str:
    """Normalize answer for comparison (as evaluation might do)."""
    if not result.ok:
        return ""
    return result.normalized_answer or ""


def _swipl_available() -> bool:
    return shutil.which("swipl") is not None


@pytest.mark.skipif(not _swipl_available(), reason="SWI-Prolog not installed")
def test_basic_execution() -> None:
    r = execute_solve(CODE_INTEGER)
    assert r.ok, f"Expected success: {r.error}"
    assert r.answer is not None


@pytest.mark.skipif(not _swipl_available(), reason="SWI-Prolog not installed")
def test_integer_result_type_and_normalization() -> None:
    """Integer-like results are standardized to float / float-string format."""
    r = execute_solve(CODE_INTEGER)
    assert r.answer == 14.0
    assert isinstance(r.answer, float)
    assert _answer_str(r) == "14.0"


@pytest.mark.skipif(not _swipl_available(), reason="SWI-Prolog not installed")
def test_decimal_clpq_succeeds_via_subprocess() -> None:
    """Subprocess backend should handle clpq decimal outputs as text."""
    r = execute_solve(CODE_DECIMAL)
    assert r.ok, f"Expected decimal (10/3) to succeed via swipl subprocess: {r.error}"
    assert r.answer is not None
    assert _answer_str(r) != ""
    assert "r" not in _answer_str(r)
    assert "." in _answer_str(r)


@pytest.mark.skipif(not _swipl_available(), reason="SWI-Prolog not installed")
def test_syntax_error_classification() -> None:
    r = execute_solve(CODE_SYNTAX_ERROR)
    assert not r.ok
    assert r.error_type == "syntax_error"
    assert r.stderr is not None
    assert "Syntax error" in r.stderr


def test_answer_normalization_for_eval() -> None:
    """Verify standardized float-string normalization."""
    from src.prolog.execute import normalize_answer_for_eval

    for val in (14, 14.0):
        s = normalize_answer_for_eval(val)
        assert s == "14.0", f"val={val!r} -> {s!r}"

    assert normalize_answer_for_eval(3.5) == "3.5"
    assert normalize_answer_for_eval("1r2") == "0.5"
    assert normalize_answer_for_eval("10r3").startswith("3.333333")
    assert "r" not in normalize_answer_for_eval("10r3")
    assert normalize_answer_for_eval(None) == ""


if __name__ == "__main__":
    # Run basic smoke test without pytest
    print("=== Prolog execution smoke test ===\n")
    r = execute_solve(CODE_INTEGER)
    print(f"Basic (integer): ok={r.ok}, answer={r.answer!r}, type={type(r.answer).__name__ if r.answer else 'N/A'}")
    if r.ok:
        print(f"  Normalized for eval: {_answer_str(r)!r}")
    else:
        print(f"  Error: {r.error}")

    r2 = execute_solve(CODE_DIVISION_INTEGER)
    print(f"\nDivision (12*50/60): ok={r2.ok}, answer={r2.answer!r}")
    if r2.ok:
        print(f"  Normalized: {_answer_str(r2)!r}")

    r3 = execute_solve(CODE_DECIMAL)
    print(f"\nDecimal (10/3): ok={r3.ok}, answer={r3.answer!r}")
    if r3.ok:
        print(f"  Normalized: {_answer_str(r3)!r}")
    else:
        print(f"  Error: {r3.error}")

    print("\n=== Normalization check ===")
    if r.ok and r.answer is not None:
        is_float = isinstance(r.answer, float)
        print(f"Integer result 14 returned as: {type(r.answer).__name__} = {r.answer!r}")
        if is_float:
            print("  -> Normalize to '14' for eval.")
        else:
            print("  -> Returns int.")
