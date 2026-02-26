from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
import re
import subprocess
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
PROLOG_TMP_DIR = REPO_ROOT / "data" / "tmp" / "prolog_exec"
_PROLOG_VAR_RE = re.compile(r"^[A-Z_][A-Za-z0-9_]*$")
_SWI_RATIONAL_RE = re.compile(r"^([+-]?\d+)r([+-]?\d+)$")


@dataclass
class PrologExecutionResult:
    ok: bool
    query: str
    answer: Any = None
    answers: list[Any] = field(default_factory=list)
    bindings: Optional[dict[str, Any]] = None
    error_type: Optional[str] = None
    error: Optional[str] = None
    elapsed_ms: Optional[float] = None
    normalized_answer: Optional[str] = None
    raw_answer_text: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    returncode: Optional[int] = None


def _normalize_query(query: str) -> str:
    q = query.strip()
    if not q:
        raise ValueError("query must be a non-empty Prolog query")
    if not q.endswith("."):
        q += "."
    return q


def normalize_answer_for_eval(answer: Any) -> str:
    """
    Normalize Prolog answer into a standard float-string representation.

    Examples:
    - 14 -> "14.0"
    - 14.0 -> "14.0"
    - "10r3" -> "3.3333333333333335"
    """
    if answer is None:
        return ""
    s = str(answer).strip()

    # SWI-Prolog may print exact rationals as e.g. "10r3" (especially from clpq).
    # Convert these to float strings automatically.
    m = _SWI_RATIONAL_RE.fullmatch(s)
    if m:
        num = int(m.group(1))
        den = int(m.group(2))
        if den == 0:
            return s
        try:
            with localcontext() as ctx:
                ctx.prec = 28
                dec = Decimal(num) / Decimal(den)
            return str(float(dec))
        except (InvalidOperation, OverflowError, ZeroDivisionError):
            return s

    try:
        return str(float(s))
    except ValueError:
        pass

    return s


def _parse_prolog_result_text(raw: str) -> Any:
    s = raw.strip()
    m = _SWI_RATIONAL_RE.fullmatch(s)
    if m:
        num = int(m.group(1))
        den = int(m.group(2))
        if den != 0:
            try:
                with localcontext() as ctx:
                    ctx.prec = 28
                    dec = Decimal(num) / Decimal(den)
                return float(dec)
            except (InvalidOperation, OverflowError, ZeroDivisionError):
                return s
        return s
    try:
        return float(s)
    except ValueError:
        return s


def _to_str(x: bytes | str | None) -> str | None:
    """Convert subprocess stdout/stderr to str; TimeoutExpired may have bytes."""
    if x is None:
        return None
    return x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x


def _is_syntax_error(*texts: Optional[str]) -> bool:
    for t in texts:
        if not t:
            continue
        low = t.lower()
        if "syntax error" in low or "syntax_error" in low:
            return True
    return False


def _build_swipl_wrapper_goal(query_text: str, answer_var: str) -> str:
    query_no_dot = query_text[:-1].strip() if query_text.endswith(".") else query_text.strip()
    return (
        "catch(("
        f"(once(({query_no_dot})) -> "
        "write('__RESULT__='), "
        f"write_term({answer_var}, [quoted(true), ignore_ops(false)]), nl ; "
        "writeln('__NO_SOLUTION__'))"
        "), E, ("
        "write('__EXCEPTION__='), "
        "write_term(E, [quoted(true), ignore_ops(false)]), nl"
        "))"
    )


def execute_prolog_string(
    code: str,
    *,
    query: str = "solve(Result)",
    answer_var: Optional[str] = "Result",
    max_solutions: int = 1,
    timeout_s: float = 2.0,
    swipl_bin: str = "swipl",
) -> PrologExecutionResult:
    """
    Execute Prolog code from a string via a one-shot `swipl` subprocess.

    This implementation intentionally keeps things simple:
    - writes code to a temp `.pl` file under a dedicated repo temp directory
    - executes a single query (default: `solve(Result)`)
    - parses one result and stores a normalized string form
    """
    started = perf_counter()
    tmp_path: Optional[Path] = None

    try:
        if max_solutions <= 0:
            raise ValueError("max_solutions must be > 0")
        if max_solutions != 1:
            raise ValueError("only max_solutions=1 is supported in the subprocess backend")
        if answer_var is None or not _PROLOG_VAR_RE.fullmatch(answer_var):
            raise ValueError("answer_var must be a valid Prolog variable name (e.g., 'Result')")

        query_text = _normalize_query(query)

        PROLOG_TMP_DIR.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w",
            suffix=".pl",
            prefix="run_",
            dir=PROLOG_TMP_DIR,
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(code.rstrip() + "\n")
            tmp_path = Path(f.name)

        goal = _build_swipl_wrapper_goal(query_text, answer_var)
        cmd = [
            swipl_bin,
            "-q",
            "-f",
            "none",
            "-l",
            str(tmp_path),
            "-g",
            goal,
            "-t",
            "halt",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        stdout_lines = [line.strip() for line in stdout.splitlines() if line.strip()]

        for line in stdout_lines:
            if line.startswith("__RESULT__="):
                raw = line[len("__RESULT__="):]
                parsed = _parse_prolog_result_text(raw)
                return PrologExecutionResult(
                    ok=True,
                    query=query_text,
                    answer=parsed,
                    answers=[parsed],
                    bindings={answer_var: parsed},
                    elapsed_ms=(perf_counter() - started) * 1000,
                    normalized_answer=normalize_answer_for_eval(parsed),
                    raw_answer_text=raw,
                    stdout=stdout,
                    stderr=stderr,
                    returncode=proc.returncode,
                )
            if line == "__NO_SOLUTION__":
                return PrologExecutionResult(
                    ok=False,
                    query=query_text,
                    error_type="no_solution",
                    error="Query returned no solutions.",
                    elapsed_ms=(perf_counter() - started) * 1000,
                    stdout=stdout,
                    stderr=stderr,
                    returncode=proc.returncode,
                )
            if line.startswith("__EXCEPTION__="):
                exc_text = line[len("__EXCEPTION__="):]
                return PrologExecutionResult(
                    ok=False,
                    query=query_text,
                    error_type="syntax_error" if _is_syntax_error(stderr, exc_text) else "execution_error",
                    error=exc_text,
                    elapsed_ms=(perf_counter() - started) * 1000,
                    stdout=stdout,
                    stderr=stderr,
                    returncode=proc.returncode,
                )

        err_msg = "No result marker found in SWI-Prolog output."
        if proc.returncode != 0:
            err_msg = f"{err_msg} swipl exited with code {proc.returncode}."
        return PrologExecutionResult(
            ok=False,
            query=query_text,
            error_type="syntax_error" if _is_syntax_error(stderr, stdout) else "execution_error",
            error=err_msg,
            elapsed_ms=(perf_counter() - started) * 1000,
            stdout=stdout,
            stderr=stderr,
            returncode=proc.returncode,
        )
    except FileNotFoundError as e:
        return PrologExecutionResult(
            ok=False,
            query=query if query.strip() else "solve(Result).",
            error_type="dependency_error",
            error=f"{type(e).__name__}: {e}",
            elapsed_ms=(perf_counter() - started) * 1000,
        )
    except subprocess.TimeoutExpired as e:
        return PrologExecutionResult(
            ok=False,
            query=query if query.strip() else "solve(Result).",
            error_type="timeout",
            error=f"{type(e).__name__}: exceeded timeout_s={timeout_s}",
            elapsed_ms=(perf_counter() - started) * 1000,
            stdout=_to_str(e.stdout) if hasattr(e, "stdout") else None,
            stderr=_to_str(e.stderr) if hasattr(e, "stderr") else None,
        )
    except Exception as e:
        return PrologExecutionResult(
            ok=False,
            query=query if query.strip() else "solve(Result).",
            error_type="execution_error",
            error=f"{type(e).__name__}: {e}",
            elapsed_ms=(perf_counter() - started) * 1000,
        )
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def execute_solve(code: str) -> PrologExecutionResult:
    """
    Convenience wrapper for the dataset contract: solve/1.
    """
    return execute_prolog_string(code, query="solve(Result)", answer_var="Result", max_solutions=1)
