"""Model loaders. fp16 for 1.5B; 4-bit nf4 for 7B (T4-friendly).
"""
from __future__ import annotations
import gc
import urllib.request
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import config


def _bnb_4bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_qwen(size: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """size in {'1.5b', '7b'}. Returns (tokenizer, model) on GPU."""
    model_id = config.MODELS[size]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if size == "7b":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=_bnb_4bit_config(),
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    return tokenizer, model


def unload(*_unused) -> None:  
    gc.collect()
    torch.cuda.empty_cache()


def setup_java_runtime() -> None:
    """Idempotent: download JUnit5 console-launcher JAR if missing."""
    config.ensure_dirs()
    jar = Path(config.JUNIT_JAR)
    if jar.exists():
        return
    print(f"Downloading {config.JUNIT_JAR_URL} ...")
    urllib.request.urlretrieve(config.JUNIT_JAR_URL, jar)
    assert jar.exists(), "JUnit JAR download failed"
    print(f"Saved to {jar} ({jar.stat().st_size / 1e6:.1f} MB)")
