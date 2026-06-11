"""Mentor task: extend datasets so each row has NL + PL1 + PL2 with validation flags."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

import config
from lib import ast_compare, execution, prompts
from lib.benchmarks import load_hex_lookup
from lib.prompts import generate


def _gen_pl2_from_nl_pl1(tokenizer, model, nl: str, pl1: str, discrepancy: str | None, pl2_attempt: str | None) -> str:
    if discrepancy and pl2_attempt:
        prompt = prompts.build_pl2_feedback_prompt(tokenizer, nl, pl1, pl2_attempt, discrepancy)
    else:
        prompt = prompts.build_pl2_generation_prompt(tokenizer, nl, pl1)
    raw = generate(tokenizer, model, prompt, config.DECODING_GREEDY)
    return prompts.extract_java_body(raw)


def _gen_nl_from_pl1_pl2(tokenizer, model, pl1: str, pl2: str, discrepancy: str | None, nl_attempt: str | None) -> str:
    if discrepancy and nl_attempt:
        prompt = prompts.build_nl_feedback_prompt(tokenizer, nl_attempt, pl1, pl2, discrepancy)
    else:
        prompt = prompts.build_nl_generation_prompt(tokenizer, pl1, pl2)
    raw = generate(tokenizer, model, prompt, config.DECODING_GREEDY)
    return raw.strip()


def _gen_pl1_from_nl(tokenizer, model, nl: str, humaneval_prompt_stub: str | None = None) -> str:
    if humaneval_prompt_stub:
        prompt = prompts.build_completion_prompt(tokenizer, {"prompt": humaneval_prompt_stub})
    else:
        system = "You are an expert Python programmer."
        user = (
            "Write a complete Python solution for this specification.\n"
            "Return the full program in a ```python ... ``` block.\n\n"
            f"{nl}"
        )
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    raw = generate(tokenizer, model, prompt, config.DECODING_GREEDY)
    return prompts.extract_python_body(raw)


def _gen_pl2_from_nl(tokenizer, model, nl: str, java_prompt_stub: str | None = None) -> str:
    if java_prompt_stub:
        prompt = prompts.build_java_completion_from_nl_prompt(tokenizer, java_prompt_stub)
    else:
        system = "You are an expert Java programmer."
        user = (
            "Write a complete Java solution (class Solution) for this specification.\n"
            "Return the full program in a ```java ... ``` block.\n\n"
            f"{nl}"
        )
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    raw = generate(tokenizer, model, prompt, config.DECODING_GREEDY)
    return prompts.extract_java_body(raw)


def extend_humaneval_row(row: dict, tokenizer, model, max_retries: int | None = None) -> dict[str, Any]:
    """Generate missing PL2 from NL+PL1; validate with Python+Java tests; retry up to N times."""
    max_retries = max_retries or config.EXTEND_MAX_RETRIES
    nl, pl1 = row["nl"], row["pl1"]
    pl2 = None
    discrepancy = None
    pl2_attempt = None
    validation = {}

    for attempt in range(1, max_retries + 1):
        pl2 = _gen_pl2_from_nl_pl1(tokenizer, model, nl, pl1, discrepancy, pl2_attempt)
        pl2_attempt = pl2
        validation = ast_compare.validate_pl2_against_pl1_humaneval(
            pl1, pl2,
            test=row["test"],
            entry_point=row["entry_point"],
            java_test=row.get("java_test"),
        )
        if validation["valid"]:
            break
        discrepancy = validation.get("error") or "validation failed"

    return {
        "dataset": "humaneval",
        "id": row["id"],
        "nl": nl,
        "pl1": pl1,
        "pl2": pl2,
        "pl2_valid": bool(validation.get("valid")),
        "pl2_attempts": attempt,
        "validation": validation,
        "hex_key": row.get("hex_key"),
    }


def extend_mbpp_row(row: dict, tokenizer, model, max_retries: int | None = None) -> dict[str, Any]:
    max_retries = max_retries or config.EXTEND_MAX_RETRIES
    nl, pl1 = row["nl"], row["pl1"]
    pl2 = None
    discrepancy = None
    pl2_attempt = None
    validation = {}

    for attempt in range(1, max_retries + 1):
        pl2 = _gen_pl2_from_nl_pl1(tokenizer, model, nl, pl1, discrepancy, pl2_attempt)
        pl2_attempt = pl2
        validation = ast_compare.validate_pl2_against_pl1_mbpp(pl1, pl2, row["tests"])
        if validation["valid"]:
            break
        discrepancy = validation.get("error") or "validation failed"

    return {
        "dataset": "mbpp",
        "id": row["id"],
        "nl": nl,
        "pl1": pl1,
        "pl2": pl2,
        "pl2_valid": bool(validation.get("valid")),
        "pl2_attempts": attempt,
        "validation": validation,
    }


def extend_humaneval_x_row(row: dict, tokenizer, model, max_retries: int | None = None) -> dict[str, Any]:
    """Generate missing NL from PL1+PL2; validate by NL→PL1' and NL→PL2'; retry up to N times."""
    max_retries = max_retries or config.EXTEND_MAX_RETRIES
    pl1, pl2 = row["pl1"], row["pl2"]
    nl = None
    discrepancy = None
    nl_attempt = None
    validation = {}

    for attempt in range(1, max_retries + 1):
        nl = _gen_nl_from_pl1_pl2(tokenizer, model, pl1, pl2, discrepancy, nl_attempt)
        nl_attempt = nl
        pl1_gen = _gen_pl1_from_nl(tokenizer, model, nl)
        pl2_gen = _gen_pl2_from_nl(tokenizer, model, nl, row.get("java_declaration"))

        key = row["id"].split("/")[-1]
        hx = load_hex_lookup().get(key, {})
        test_py = hx.get("test_py")
        entry_point = hx.get("entry_point_py")

        if test_py and entry_point:
            validation = ast_compare.validate_nl_via_regeneration(
                nl, pl1, pl2, pl1_gen, pl2_gen,
                test=test_py,
                entry_point=entry_point,
                java_test=row.get("java_test"),
            )
        else:
            pl1_ast, pl1_ast_err = ast_compare.python_ast_match(pl1, pl1_gen)
            jr = execution.run_java(pl2_gen, row["java_test"])
            validation = {
                "valid": pl1_ast and jr["passed"],
                "pl1_ast_match": pl1_ast,
                "pl2_java_pass": jr["passed"],
                "error": pl1_ast_err or jr.get("error"),
            }

        if validation.get("valid"):
            break
        discrepancy = validation.get("error") or "nl regeneration validation failed"

    return {
        "dataset": "humaneval_x",
        "id": row["id"],
        "nl": nl,
        "pl1": pl1,
        "pl2": pl2,
        "nl_valid": bool(validation.get("valid")),
        "nl_attempts": attempt,
        "validation": validation,
    }


def run_extension(
    tokenizer,
    model,
    *,
    humaneval_rows: list[dict] | None = None,
    mbpp_rows: list[dict] | None = None,
    hex_rows: list[dict] | None = None,
) -> list[dict[str, Any]]:
    from lib.benchmarks import (
        load_humaneval_for_extension,
        load_mbpp_for_extension,
        load_humaneval_x_for_extension,
    )

    humaneval_rows = humaneval_rows if humaneval_rows is not None else load_humaneval_for_extension()
    mbpp_rows = mbpp_rows if mbpp_rows is not None else load_mbpp_for_extension()
    hex_rows = hex_rows if hex_rows is not None else load_humaneval_x_for_extension()

    out: list[dict[str, Any]] = []
    for row in tqdm(humaneval_rows, desc="extend humaneval (NL+PL1→PL2)"):
        out.append(extend_humaneval_row(row, tokenizer, model))
    for row in tqdm(mbpp_rows, desc="extend mbpp (NL+PL1→PL2)"):
        out.append(extend_mbpp_row(row, tokenizer, model))
    for row in tqdm(hex_rows, desc="extend humaneval-x (PL1+PL2→NL)"):
        out.append(extend_humaneval_x_row(row, tokenizer, model))
    return out


def save_extended(rows: list[dict[str, Any]], path: Path | None = None) -> Path:
    config.ensure_dirs()
    path = path or config.EXTENDED_DIR / "unified_dataset.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def summarize_extension(rows: list[dict[str, Any]]) -> str:
    from collections import defaultdict

    stats = defaultdict(lambda: {"total": 0, "valid": 0})
    for r in rows:
        ds = r["dataset"]
        stats[ds]["total"] += 1
        if r.get("pl2_valid") or r.get("nl_valid"):
            stats[ds]["valid"] += 1
    lines = ["dataset  total  valid  rate"]
    for ds, s in sorted(stats.items()):
        rate = s["valid"] / s["total"] if s["total"] else 0
        lines.append(f"{ds:12s} {s['total']:5d} {s['valid']:5d} {rate:5.1%}")
    return "\n".join(lines)
