"""
Unit tests for the Universal Speculative Router & Heuristic Auto-Selector.
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
from typing import List

from mlx_ternary.speculative.router import (
    select_speculative_mode,
    generate_speculative,
    native_generate,
    get_device_memory_gb,
)


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 99
        self.vocab = {
            "hello": 1,
            "world": 2,
            "def": 3,
            "foo": 4,
            "calculate": 5,
            "end": 99,
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        words = text.lower().replace(":", "").replace("(", "").replace(")", "").split()
        return [self.vocab.get(w, 10) for w in words]

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.inv_vocab.get(t, f"<tok_{t}>") for t in token_ids]
        return " ".join(tokens) + " "


class MockLayer(nn.Module):
    def __init__(self, is_linear: bool = False):
        super().__init__()
        self.is_linear = is_linear

    def __call__(self, x, mask=None, cache=None):
        if cache is not None:
            cache.offset += x.shape[1]
        return x + 0.1


class MockInnerModel(nn.Module):
    def __init__(self, num_layers: int = 8):
        super().__init__()
        self.fa_idx = 1
        self.ssm_idx = 0
        self.layers = [MockLayer(is_linear=(i % 2 == 0)) for i in range(num_layers)]

    def embed_tokens(self, x):
        B, S = x.shape
        return mx.ones((B, S, 16))

    def norm(self, h):
        return h

    def __call__(self, x, mask=None, cache=None):
        h = self.embed_tokens(x)
        for layer, c in zip(self.layers, cache if cache else [None] * len(self.layers)):
            h = layer(h, mask=mask, cache=c)
        return self.norm(h)


class MockCache:
    def __init__(self):
        self.offset = 0

    def trim(self, n):
        self.offset -= n


class MockModel(nn.Module):
    def __init__(self, num_layers: int = 8, vocab_size: int = 100):
        super().__init__()
        self.model = MockInnerModel(num_layers=num_layers)
        self.vocab_size = vocab_size

    def make_cache(self):
        return [MockCache() for _ in range(len(self.model.layers))]

    def lm_head(self, h):
        B, S = h.shape[:2]
        # Return deterministic logits: index 2 highest, so argmax gives token 2
        logits = mx.zeros((B, S, self.vocab_size))
        return logits


# -----------------------------------------------------------------------------
# 1. Mode Selection Heuristic Unit Tests
# -----------------------------------------------------------------------------

def test_select_speculative_mode_explicit():
    assert select_speculative_mode("any prompt", mode="pld") == "pld"
    assert select_speculative_mode("any prompt", mode="self") == "self"
    assert select_speculative_mode("any prompt", mode="qwen") == "qwen"
    assert select_speculative_mode("any prompt", mode="none") == "none"

    with pytest.raises(ValueError):
        select_speculative_mode("any prompt", mode="invalid_mode")


def test_select_speculative_mode_code():
    code_prompts = [
        "```python\ndef fibonacci(n):\n    return n\n```",
        "Write a Python class Foo with an __init__ method.",
        "import numpy as np\nfrom math import sqrt",
        '{"name": "Alice", "age": 30, "skills": ["python", "mlx"]}',
        "Produce the JSON schema for this API response.",
    ]
    for prompt in code_prompts:
        assert select_speculative_mode(prompt, mode="auto", device_mem_gb=32.0) == "pld"


def test_select_speculative_mode_memory_constrained():
    # When device memory < 24 GB, it should prioritize zero-RAM PLD
    prompt = "Tell me an epic bedtime story about a friendly dragon."
    assert select_speculative_mode(prompt, mode="auto", device_mem_gb=16.0) == "pld"
    assert select_speculative_mode(prompt, mode="auto", device_mem_gb=23.9) == "pld"


def test_select_speculative_mode_reasoning():
    reasoning_prompts = [
        "Calculate the derivative of f(x) = x^3 - 4x + 1.",
        "Solve this math puzzle step by step please.",
        "Evaluate the following GSM8K math problem.",
        "Prove that the square root of 2 is irrational.",
    ]
    for prompt in reasoning_prompts:
        assert select_speculative_mode(prompt, mode="auto", device_mem_gb=32.0) == "self"


def test_select_speculative_mode_fallback():
    prompt = "Describe the view from the top of Mount Everest on a sunny morning."
    assert select_speculative_mode(prompt, mode="auto", device_mem_gb=32.0) == "none"


def test_get_device_memory_gb():
    mem = get_device_memory_gb()
    # On Apple Silicon running macOS, this should return a positive float (e.g. 24.0 or 32.0)
    assert mem is not None
    assert mem > 0.0


# -----------------------------------------------------------------------------
# 2. Execution & Generation Unit Tests
# -----------------------------------------------------------------------------

def test_native_generate():
    model = MockModel(num_layers=4)
    prompt_tokens = [1, 2, 3]

    tokens_gen = []
    for tok, is_acc, stats in native_generate(model, prompt_tokens, max_tokens=5):
        tokens_gen.append(tok)
        assert is_acc is False
        assert stats["mode"] == "none"

    assert len(tokens_gen) == 5
    assert stats["total_tokens"] == 5


def test_generate_speculative_none_mode():
    model = MockModel(num_layers=4)
    tokenizer = MockTokenizer()
    prompt = "hello world"

    chunks = list(
        generate_speculative(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            mode="none",
            max_tokens=4,
        )
    )
    assert len(chunks) == 4
    assert all(isinstance(c, str) for c in chunks)


def test_generate_speculative_auto_code_dispatch():
    model = MockModel(num_layers=4)
    tokenizer = MockTokenizer()
    prompt = "def foo(): return 42"

    chunks = list(
        generate_speculative(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            mode="auto",
            max_tokens=4,
        )
    )
    assert len(chunks) == 4


def test_generate_speculative_auto_self_dispatch():
    model = MockModel(num_layers=4)
    tokenizer = MockTokenizer()
    prompt = "calculate the sum step by step"

    chunks = list(
        generate_speculative(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            mode="self",
            max_tokens=4,
            draft_layers=2,
        )
    )
    assert len(chunks) == 4


# -----------------------------------------------------------------------------
# 3. CLI Argument Parsing Unit Tests
# -----------------------------------------------------------------------------

def test_cli_generate_argument_parsing():
    import sys
    from mlx_ternary.cli import main
    import argparse

    # Test parser creation and arguments
    from mlx_ternary import cli
    parser = argparse.ArgumentParser(prog="mlx_ternary")
    subparsers = parser.add_subparsers(dest="command")

    gen_p = subparsers.add_parser("generate")
    gen_p.add_argument("repo_id")
    gen_p.add_argument("prompt_pos", nargs="?", default=None)
    gen_p.add_argument("--prompt", "-p", default=None)
    gen_p.add_argument("--speculative", default="auto", choices=["auto", "pld", "self", "qwen", "none"])
    gen_p.add_argument("--draft-layers", type=int, default=16)
    gen_p.add_argument("--draft-model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    gen_p.add_argument("--max-tokens", type=int, default=256)
    gen_p.add_argument("--verbose", "-v", action="store_true")

    # Positional prompt
    args = parser.parse_args(["generate", "my-org/my-model", "def test(): pass"])
    assert args.command == "generate"
    assert args.repo_id == "my-org/my-model"
    assert args.prompt_pos == "def test(): pass"
    assert args.speculative == "auto"
    assert args.draft_layers == 16
    assert args.max_tokens == 256
    assert args.verbose is False

    # Flagged prompt with options
    args2 = parser.parse_args([
        "generate",
        "my-model",
        "--prompt", "calculate 2+2",
        "--speculative", "self",
        "--draft-layers", "8",
        "--max-tokens", "100",
        "--verbose",
    ])
    assert args2.prompt == "calculate 2+2"
    assert args2.speculative == "self"
    assert args2.draft_layers == 8
    assert args2.max_tokens == 100
    assert args2.verbose is True


def test_cli_bench_argument_parsing():
    import argparse
    from mlx_ternary.cli import get_hardware_info

    hw = get_hardware_info()
    assert isinstance(hw, dict)
    assert "chip" in hw
    assert "memory_gb" in hw

    parser = argparse.ArgumentParser(prog="mlx_ternary")
    subparsers = parser.add_subparsers(dest="command")
    bench_p = subparsers.add_parser("bench")
    bench_p.add_argument("--duration-sec", type=float, default=2.0)
    bench_p.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args(["bench", "--duration-sec", "0.5", "-v"])
    assert args.command == "bench"
    assert args.duration_sec == 0.5
    assert args.verbose is True

