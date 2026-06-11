"""Execution sandboxes. subprocess + timeout in lieu of Docker (Colab constraint).

run_python: feed prompt + generated body + tests to a Python interpreter, score pass/fail.
run_java:   compile generated Java, run JUnit5 standalone JAR, score pass/fail.

"""
from __future__ import annotations
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import config


# ---------------- Python ----------------
def run_python(full_program: str, timeout: int = config.EXEC_TIMEOUT_S) -> dict:
    """`full_program` should already include the test harness (assertions, calls, etc).
    Returns {"passed": bool, "error": str | None}.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "solution.py"
        path.write_text(full_program, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            if result.returncode == 0:
                return {"passed": True, "error": None}
            err = (result.stderr or result.stdout).strip().splitlines()[-1:]
            return {"passed": False, "error": "\n".join(err) or "nonzero exit"}
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": f"timeout >{timeout}s"}
        except Exception as e:
            return {"passed": False, "error": f"runner: {e!r}"}


# ---------------- Java ----------------
_CLASS_NAME_RE = re.compile(r"public\s+class\s+(\w+)")

# Imports the original HumanEval-X scoring expects to be available in BOTH
# Solution.java and Main.java. The reference scoring uses a single combined
# file (prompt + canonical_solution + test), where the prompt's imports flow
# through. Since we split into two files, each file needs its own import
# section. Java tolerates duplicate imports (just a warning), so prepending
# unconditionally is safe.
_JAVA_STD_IMPORTS = (
    "import java.util.*;\n"
    "import java.util.stream.*;\n"
    "import java.util.regex.*;\n"
    "import java.lang.*;\n"
    "import java.math.*;\n"
)


def _extract_classes(java_src: str) -> list[str]:
    return _CLASS_NAME_RE.findall(java_src)


def _inject_imports(src: str) -> str:
    """Prepend standard java.util / java.util.stream imports if not already at top."""
    head = src.lstrip()
    if head.startswith("package "):  # never seen for HumanEval-X but defensive
        nl = head.find("\n")
        return head[: nl + 1] + _JAVA_STD_IMPORTS + head[nl + 1 :]
    return _JAVA_STD_IMPORTS + src


def run_java(java_source: str, test_source: str, timeout: int = config.EXEC_TIMEOUT_S) -> dict:
    """Compile both files, run the test class' main() (HumanEval-X convention).

    java_source: the model's translated code (typically class Solution { ... })
    test_source: the reference test file from HumanEval-X (typically class Main { public static void main ... })
    Returns {"passed": bool, "error": str | None}.
    """
    classes_in_test = _extract_classes(test_source)
    main_class = classes_in_test[0] if classes_in_test else "Main"

    # HumanEval-X test files reference List/Arrays/Collectors etc. without imports;
    # inject standard imports to make the split-file harness compile.
    java_source = _inject_imports(java_source)
    test_source = _inject_imports(test_source)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # split combined java_source into per-class files (compiler tolerates one-file multi-class
        # but write_text once is simplest)
        sol_path = tmp_path / "Solution.java"
        test_path = tmp_path / f"{main_class}.java"
        sol_path.write_text(java_source, encoding="utf-8")
        test_path.write_text(test_source, encoding="utf-8")

        try:
            compile_res = subprocess.run(
                ["javac", "-d", str(tmp_path), str(sol_path), str(test_path)],
                capture_output=True, text=True, timeout=timeout,
            )
            if compile_res.returncode != 0:
                return {"passed": False, "error": "compile: " + (compile_res.stderr or "").strip()[:300]}

            run_res = subprocess.run(
                ["java", "-cp", str(tmp_path), main_class],
                capture_output=True, text=True, timeout=timeout,
            )
            if run_res.returncode == 0:
                return {"passed": True, "error": None}
            tail = (run_res.stderr or run_res.stdout).strip().splitlines()[-3:]
            return {"passed": False, "error": "\n".join(tail) or "nonzero exit"}
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": f"timeout >{timeout}s"}
        except Exception as e:
            return {"passed": False, "error": f"runner: {e!r}"}


def compile_java(java_source: str, timeout: int = config.EXEC_TIMEOUT_S) -> dict:
    """Compile Java source only (no test execution)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sol_path = tmp_path / "Solution.java"
        sol_path.write_text(_inject_imports(java_source), encoding="utf-8")
        try:
            compile_res = subprocess.run(
                ["javac", str(sol_path)],
                capture_output=True, text=True, timeout=timeout,
            )
            if compile_res.returncode == 0:
                return {"passed": True, "error": None}
            return {"passed": False, "error": "compile: " + (compile_res.stderr or "").strip()[:300]}
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": f"timeout >{timeout}s"}
        except Exception as e:
            return {"passed": False, "error": f"runner: {e!r}"}


def smoke_test_java() -> dict:
    """Sanity-check the Java toolchain. Compiles and runs a trivial passing program."""
    sol = "public class Solution { public static int add(int a, int b) { return a + b; } }"
    tst = (
        "public class Main {\n"
        "    public static void main(String[] args) {\n"
        "        if (Solution.add(2, 3) != 5) throw new AssertionError();\n"
        "    }\n"
        "}\n"
    )
    return run_java(sol, tst)
