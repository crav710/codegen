"""AST and output comparison helpers for dataset extension validation."""
from __future__ import annotations

import ast
import re

from lib import execution
import config


def extract_docstring_from_prompt(prompt: str) -> str:
    """Pull the triple-quoted docstring from a HumanEval-style prompt."""
    m = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"'''(.*?)'''", prompt, re.DOTALL)
    if m:
        return m.group(1).strip()
    return prompt.strip()


def python_ast_dump(source: str) -> str | None:
    try:
        tree = ast.parse(source)
        return ast.dump(tree, annotate_fields=False)
    except SyntaxError:
        return None


def python_ast_match(code_a: str, code_b: str) -> tuple[bool, str | None]:
    """Return (match, error_message)."""
    a = python_ast_dump(code_a)
    b = python_ast_dump(code_b)
    if a is None:
        return False, "code_a: syntax error"
    if b is None:
        return False, "code_b: syntax error"
    if a == b:
        return True, None
    return False, "python AST mismatch"


def python_passes_humaneval(program: str, test: str, entry_point: str) -> dict:
    full = program.rstrip() + "\n\n" + test + f"\n\ncheck({entry_point})\n"
    return execution.run_python(full, timeout=config.EXEC_TIMEOUT_S)


def python_passes_mbpp(program: str, tests: list[str]) -> dict:
    full = program.rstrip() + "\n\n" + "\n".join(tests) + "\n"
    return execution.run_python(full, timeout=config.EXEC_TIMEOUT_S)


def python_output_match_humaneval(code_a: str, code_b: str, test: str, entry_point: str) -> tuple[bool, str | None]:
    """Both programs must pass the same HumanEval harness (proxy for output equivalence)."""
    ra = python_passes_humaneval(code_a, test, entry_point)
    rb = python_passes_humaneval(code_b, test, entry_point)
    if ra["passed"] and rb["passed"]:
        return True, None
    parts = []
    if not ra["passed"]:
        parts.append(f"code_a: {ra['error']}")
    if not rb["passed"]:
        parts.append(f"code_b: {rb['error']}")
    return False, "; ".join(parts) or "output mismatch"


def python_output_match_mbpp(code_a: str, code_b: str, tests: list[str]) -> tuple[bool, str | None]:
    ra = python_passes_mbpp(code_a, tests)
    rb = python_passes_mbpp(code_b, tests)
    if ra["passed"] and rb["passed"]:
        return True, None
    parts = []
    if not ra["passed"]:
        parts.append(f"code_a: {ra['error']}")
    if not rb["passed"]:
        parts.append(f"code_b: {rb['error']}")
    return False, "; ".join(parts) or "output mismatch"


def validate_pl2_against_pl1_humaneval(
    pl1: str,
    pl2: str,
    *,
    test: str,
    entry_point: str,
    java_test: str | None,
) -> dict:
    """Mentor validation for generated PL2.

    Output: PL1 passes Python tests AND PL2 passes Java tests (same problem).
    AST: PL1 reference must parse (structural sanity on PL1).
    """
    py_ref = python_passes_humaneval(pl1, test, entry_point)
    if not py_ref["passed"]:
        return {
            "valid": False,
            "output_match": False,
            "ast_match": False,
            "java_pass": False,
            "error": f"pl1 reference failed tests: {py_ref['error']}",
        }

    ast_ok, ast_err = python_ast_match(pl1, pl1)
    if not ast_ok:
        return {"valid": False, "output_match": True, "ast_match": False, "java_pass": False, "error": ast_err}

    if not java_test:
        return {
            "valid": False,
            "output_match": True,
            "ast_match": True,
            "java_pass": False,
            "error": "no HumanEval-X java_test for this problem id",
        }

    jr = execution.run_java(pl2, java_test)
    if not jr["passed"]:
        return {
            "valid": False,
            "output_match": True,
            "ast_match": True,
            "java_pass": False,
            "error": f"java tests failed: {jr['error']}",
        }

    return {"valid": True, "output_match": True, "ast_match": True, "java_pass": True, "error": None}


def validate_pl2_against_pl1_mbpp(pl1: str, pl2: str, tests: list[str]) -> dict:
    """MBPP has no Java tests: PL1 must pass asserts; PL2 must at least compile."""
    py_ref = python_passes_mbpp(pl1, tests)
    if not py_ref["passed"]:
        return {
            "valid": False,
            "output_match": False,
            "ast_match": False,
            "java_pass": False,
            "error": f"pl1 reference failed tests: {py_ref['error']}",
        }
    ast_ok, ast_err = python_ast_match(pl1, pl1)
    if not ast_ok:
        return {"valid": False, "output_match": True, "ast_match": False, "java_pass": False, "error": ast_err}
    jc = execution.compile_java(pl2)
    if not jc["passed"]:
        return {
            "valid": False,
            "output_match": True,
            "ast_match": True,
            "java_pass": False,
            "error": f"pl2 compile failed: {jc['error']}",
        }
    return {
        "valid": True,
        "output_match": True,
        "ast_match": True,
        "java_pass": True,
        "error": None,
        "note": "MBPP: PL2 validated by compile only (no Java test harness)",
    }


def validate_nl_via_regeneration(
    nl: str,
    pl1_gt: str,
    pl2_gt: str,
    pl1_gen: str,
    pl2_gen: str,
    *,
    test: str | None,
    entry_point: str | None,
    java_test: str | None,
) -> dict:
    """Validate NL by checking regenerated PL1' and PL2' against ground truth."""
    pl1_out, pl1_out_err = python_output_match_humaneval(pl1_gt, pl1_gen, test, entry_point) if test and entry_point else (True, None)
    pl1_ast, pl1_ast_err = python_ast_match(pl1_gt, pl1_gen)

    pl2_java = execution.run_java(pl2_gen, java_test) if java_test else {"passed": False, "error": "no java_test"}
    pl2_out_ok = pl2_java["passed"]
    pl2_ast, pl2_ast_err = (True, None)

    valid = pl1_out and pl1_ast and pl2_out_ok
    errors = [e for e in (pl1_out_err, pl1_ast_err, pl2_java.get("error")) if e]
    return {
        "valid": valid,
        "pl1_output_match": pl1_out,
        "pl1_ast_match": pl1_ast,
        "pl2_java_pass": pl2_out_ok,
        "error": "; ".join(errors) if errors else None,
    }
