"""Qwen chat-template prompt builders + code-block extractors.

  - HumanEval / MBPP / HumanEval-X all ask the model to return a COMPLETE
    program (or full function / full class) inside a single fenced code block.
  - Scorers feed the extracted block to the execution sandbox AS-IS — they do
    NOT prepend the original benchmark prompt or strip 'def' / 'class' lines.
    See `docs/round1_analysis.md` for why.
"""
from __future__ import annotations
import re

def _apply_chat(tokenizer, system: str, user: str) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ----------------- HumanEval (Python completion) -----------------
def build_completion_prompt(tokenizer, problem: dict) -> str:
    """HumanEval-style code completion.

    The model is expected to return the COMPLETE Python program (imports +
    function definition with body) inside a ```python ... ``` block. The
    scorer treats the extracted block as the entire program — no concatenation
    with the original prompt.
    """
    system = "You are an expert Python programmer."
    user = (
        f"Complete the following Python code. "
        f"Return the COMPLETE program (imports, the full function signature, and the implementation) "
        f"in a single ```python ... ``` code block. "
        f"Do NOT return only the body — repeat the function signature and any required imports.\n\n"
        f"```python\n{problem['prompt']}\n```"
    )
    return _apply_chat(tokenizer, system, user)


# ----------------- HumanEval-X (Python -> Java translation) -----------------
def build_translation_prompt(tokenizer, python_source: str, java_declaration: str) -> str:
    """Python -> Java translation.

    The model is expected to return a COMPLETE Java solution — imports +
    `class Solution { ... }` with the method — inside a ```java ... ``` block.
    The scorer writes the extracted code to `Solution.java` and compiles it
    together with the reference test (a separate `class Main`). NOTE: do NOT
    prepend `java_declaration`; the model's output is the entire Solution.java.
    `java_declaration` is shown only for signature reference.
    """
    system = "You are a code translator. You translate Python to Java preserving behavior."
    user = (
        f"Translate the following Python function to Java. Preserve behavior exactly.\n"
        f"Return the COMPLETE Java solution — imports + `class Solution {{ ... }}` containing "
        f"the translated method — inside a single ```java ... ``` code block. "
        f"Match the method signature shown below.\n\n"
        f"### Python source\n```python\n{python_source}\n```\n\n"
        f"### Required Java method signature\n```java\n{java_declaration}\n```"
    )
    return _apply_chat(tokenizer, system, user)


# ----------------- generic generate -----------------
def generate(tokenizer, model, prompt: str, decoding: dict) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **decoding,
        )
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return text


# ----------------- code-fence-aware extractors -----------------
def _strip_fences(text: str, lang_hints: tuple[str, ...]) -> str:
    """Return the contents of the FIRST ```<lang>?\\n ... ``` block in `text`,
    falling back to the stripped text if no fence is present.

    Tolerates a missing closing fence (returns from the open fence to end of
    text). Does NOT strip language tokens like `def` or `class` from the body —
    that's the caller's job (and our scorers no longer do it).
    """
    s = text.strip()
    open_re = re.compile(r"```(?:" + "|".join(lang_hints) + r")?\s*\n?", re.IGNORECASE)
    m = open_re.search(s)
    if not m:
        return (s + "\n") if s else ""
    body = s[m.end():]
    close = re.search(r"```", body)
    body = body[:close.start()] if close else body
    body = body.rstrip()
    return body + "\n" if body else ""


def extract_python_body(text: str) -> str:
    """Extract Python code from a model response. Returns the contents of the
    first ```python ... ``` (or unlabeled ``` ... ```) block. If no fence is
    found, returns the whole stripped text."""
    return _strip_fences(text, ("python", "py"))


def extract_java_body(text: str) -> str:
    """Extract Java code from a model response. Returns the contents of the
    first ```java ... ``` (or unlabeled ``` ... ```) block. If no fence is
    found, returns the whole stripped text."""
    return _strip_fences(text, ("java",))
