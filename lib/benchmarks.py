
from __future__ import annotations
import gzip
import json
import random
from datasets import load_dataset
from huggingface_hub import hf_hub_download

import config


def _limit(rows, n):    
    if not n or n >= len(rows):
        return rows
    rng = random.Random(config.SEED)
    indices = sorted(rng.sample(range(len(rows)), n))
    return [rows[i] for i in indices]


# ---------------- HumanEval ----------------
HUMANEVAL_REPO = "openai/openai_humaneval"


def load_humaneval():
    ds = load_dataset(HUMANEVAL_REPO, split="test")
    rows = [{"id": r["task_id"], "prompt": r["prompt"], "test": r["test"],
             "entry_point": r["entry_point"], "gold": r["canonical_solution"]} for r in ds]
    return _limit(rows, config.BENCHMARK_LIMIT)


# ---------------- MBPP ----------------
MBPP_REPO = "google-research-datasets/mbpp"


def load_mbpp():
    ds = load_dataset(MBPP_REPO, "sanitized", split="test")
    rows = []
    for r in ds:
        prompt = (f'"""\n{r["prompt"]}\n"""\n# Tests:\n' + "\n".join(r["test_list"]) + "\n")
        rows.append({"id": f"mbpp/{r['task_id']}", "prompt": prompt,
                     "tests": r["test_list"], "gold": r["code"]})
    return _limit(rows, config.BENCHMARK_LIMIT)


# ---------------- HumanEval-X (Python -> Java translation) ----------------
HUMANEVAL_X_REPO = "THUDM/humaneval-x"


def _hex_split(lang: str) -> list[dict]:
    """Load one HumanEval-X language split.

    `THUDM/humaneval-x` ships as a dataset *script* (`humaneval-x.py`) which newer
    `datasets` versions refuse to execute. We bypass `load_dataset()` and download the
    raw `.jsonl.gz` shard directly via `huggingface_hub`.
    """
    candidates = [
        f"data/{lang}/data/humaneval.jsonl",             # current THUDM layout (uncompressed)
        f"data/{lang}/data/humaneval_{lang}.jsonl.gz",   # original CodeGeeX layout
        f"data/{lang}/humaneval_{lang}.jsonl.gz",
        f"{lang}/data/humaneval_{lang}.jsonl.gz",
        f"data/{lang}/data/humaneval.jsonl.gz",          # in case it gets re-compressed
    ]
    last_err: Exception | None = None
    for filename in candidates:
        try:
            path = hf_hub_download(
                repo_id=HUMANEVAL_X_REPO,
                filename=filename,
                repo_type="dataset",
            )
            opener = gzip.open if filename.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Failed to load HumanEval-X split '{lang}' from {HUMANEVAL_X_REPO}. "
        f"Last error: {last_err!r}"
    )


def load_humaneval_x_py2java():
    py = _hex_split("python")
    ja = _hex_split("java")
    by_id_py = {r["task_id"].split("/")[-1]: r for r in py}
    by_id_ja = {r["task_id"].split("/")[-1]: r for r in ja}
    rows = []
    for k in sorted(by_id_ja.keys(), key=lambda x: int(x) if x.isdigit() else x):
        if k not in by_id_py:
            continue
        rp, rj = by_id_py[k], by_id_ja[k]
        rows.append({
            "id": f"hex/{k}",
            "python_source": rp["prompt"] + rp["canonical_solution"],
            "java_prompt": rj["prompt"],
            "java_test": rj["test"],
            "java_declaration": rj.get("declaration", rj["prompt"]),
            "java_reference": rj["prompt"] + rj["canonical_solution"],
        })
    return _limit(rows, config.BENCHMARK_LIMIT)


def _humaneval_hex_key(task_id: str) -> str:
    return task_id.split("/")[-1]


def load_hex_lookup() -> dict[str, dict]:
    """HumanEval-X rows keyed by problem number (e.g. '0', '1', ...)."""
    py = _hex_split("python")
    ja = _hex_split("java")
    by_id_py = {r["task_id"].split("/")[-1]: r for r in py}
    by_id_ja = {r["task_id"].split("/")[-1]: r for r in ja}
    lookup = {}
    for k in by_id_ja:
        if k not in by_id_py:
            continue
        rp, rj = by_id_py[k], by_id_ja[k]
        lookup[k] = {
            "hex_id": f"hex/{k}",
            "python_source": rp["prompt"] + rp["canonical_solution"],
            "java_prompt": rj["prompt"],
            "java_test": rj["test"],
            "java_reference": rj["prompt"] + rj["canonical_solution"],
            "entry_point_py": rp.get("entry_point"),
            "test_py": rp.get("test"),
        }
    return lookup


def load_humaneval_for_extension():
    """HumanEval rows with NL/PL1 fields and optional HEX java metadata."""
    from lib.ast_compare import extract_docstring_from_prompt

    hex_lookup = load_hex_lookup()
    rows = []
    for r in load_humaneval():
        key = _humaneval_hex_key(r["id"])
        hx = hex_lookup.get(key, {})
        rows.append({
            **r,
            "nl": extract_docstring_from_prompt(r["prompt"]),
            "pl1": r["prompt"] + r["gold"],
            "pl2": None,
            "hex_key": key,
            "java_test": hx.get("java_test"),
            "java_reference": hx.get("java_reference"),
        })
    return _limit(rows, config.EXTEND_LIMIT)


def load_mbpp_for_extension():
    from lib.ast_compare import extract_docstring_from_prompt

    rows = []
    for r in load_mbpp():
        rows.append({
            **r,
            "nl": extract_docstring_from_prompt(r["prompt"]) or r["prompt"],
            "pl1": r["gold"] if r["gold"].lstrip().startswith("def") else r["prompt"] + r["gold"],
            "pl2": None,
        })
    return _limit(rows, config.EXTEND_LIMIT)


def load_humaneval_x_for_extension():
    """HumanEval-X rows with PL1/PL2; NL to be generated."""
    rows = []
    for r in load_humaneval_x_py2java():
        rows.append({
            **r,
            "nl": None,
            "pl1": r["python_source"],
            "pl2": r["java_reference"],
            "java_declaration": r.get("java_declaration"),
        })
    return _limit(rows, config.EXTEND_LIMIT)


