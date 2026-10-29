"""
Universal Speculative Router & Heuristic Auto-Selector for mlx-ternary.

Unifies all speculative decoding strategies under a single model-aware runtime interface:
1. "pld": Prompt Lookup Decoding (Zero-RAM, n-gram matching, 1.65x speedup on code/structured data).
2. "self": Self-Speculative Layer-Pruned Drafting (Intermediate layer exit, 0 MB extra weight VRAM).
3. "qwen": Micro-Draft LLM Engine (External small model proposal with cross-vocabulary mapping).
4. "none": Direct autoregressive decode (M=1) utilizing our verified shape-aware hybrid GEMV kernel.
5. "auto": Intelligent heuristic selector determining the optimal speculative engine based on
   prompt syntax, memory constraints, and task characteristics.
"""

from typing import Generator, Optional, Any, Set, List, Tuple, Dict
import os
import sys
import subprocess
import mlx.core as mx

from .pld import pld_generate, generate_pld_speculative
from .self_speculative import self_speculative_generate, generate_self_speculative
from .draft_model import (
    draft_speculative_generate,
    generate_draft_speculative,
    load_draft_model,
    VocabularyMapper,
)

CODE_PATTERNS = [
    "```",
    "def ",
    "class ",
    "import ",
    '{"',
    "{'",
    "{\n",
    "function ",
    "const ",
    "let ",
    "var ",
    "select * from",
    "#!/bin",
    "->",
    "=>",
]

CODE_INSENSITIVE_PATTERNS = [
    "json",
    "<html",
    "<script",
    "schema",
    "select * from",
]

REASONING_PATTERNS = [
    "calculate",
    "solve",
    "step by step",
    "reasoning",
    "think step",
    "gsm8k",
    "math",
    "prove",
    "how many",
    "evaluate the following",
]


def get_device_memory_gb() -> Optional[float]:
    """Query Apple Silicon unified memory capacity in gigabytes."""
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
            bytes_mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            return float(bytes_mem) / (1024.0 ** 3)
    except Exception:
        pass

    try:
        res = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(int(res.stdout.strip())) / (1024.0 ** 3)
    except Exception:
        return None


def select_speculative_mode(
    prompt: str,
    mode: str = "auto",
    device_mem_gb: Optional[float] = None,
) -> str:
    """
    Select the optimal speculative decoding mode based on prompt characteristics,
    hardware memory constraints, or user override.

    Args:
        prompt: Raw input text prompt.
        mode: User-selected mode ("auto", "pld", "self", "qwen", "none").
        device_mem_gb: Optional total unified memory in GB for testing heuristics.

    Returns:
        One of {"pld", "self", "qwen", "none"}.
    """
    valid_modes = {"auto", "pld", "self", "qwen", "none"}
    mode_lower = mode.lower().strip()
    if mode_lower not in valid_modes:
        raise ValueError(f"Unknown speculative mode '{mode}'. Valid modes: {sorted(list(valid_modes))}")

    if mode_lower != "auto":
        return mode_lower

    prompt_lower = prompt.lower()

    # Heuristic 1: Code / structured JSON syntax -> PLD (zero VRAM overhead, 100% acceptance on repeats)
    if (
        any(p in prompt for p in CODE_PATTERNS)
        or any(p in prompt_lower for p in CODE_INSENSITIVE_PATTERNS)
        or prompt.strip().startswith("from ")
        or "\nfrom " in prompt
    ):
        return "pld"

    # Heuristic 2: Constrained unified memory (< 24 GB) -> PLD (0 MB extra VRAM)
    if device_mem_gb is None:
        device_mem_gb = get_device_memory_gb()
    if device_mem_gb is not None and device_mem_gb < 24.0:
        return "pld"

    # Heuristic 3: Multi-step reasoning / math -> Self-speculative layer-pruned draft
    if any(p in prompt_lower for p in REASONING_PATTERNS):
        return "self"

    # Heuristic 4: Fallback -> Native shape-aware GEMV decode
    return "none"


def native_generate(
    model: Any,
    prompt_tokens: List[int],
    max_tokens: int = 256,
    stop_tokens: Optional[Set[int]] = None,
    cache: Optional[List[Any]] = None,
) -> Generator[Tuple[int, bool, Dict[str, Any]], None, None]:
    """
    Standard M=1 autoregressive decode loop utilizing verified shape-aware hybrid GEMV.

    Yields:
        (token_id, is_draft_accepted, stats_dict)
    """
    if stop_tokens is None:
        stop_tokens = set()

    if cache is None and hasattr(model, "make_cache"):
        cache = model.make_cache()

    stats = {
        "mode": "none",
        "speculative_steps": 0,
        "fallback_steps": 0,
        "draft_tokens_proposed": 0,
        "draft_tokens_accepted": 0,
        "acceptance_rate": 0.0,
        "total_tokens": 0,
    }

    # 1. Prompt Prefill Phase
    x_prompt = mx.array([prompt_tokens], dtype=mx.int32)
    if cache is not None:
        hidden = model.model(x_prompt, cache=cache)
    else:
        hidden = model.model(x_prompt)
    logits = model.lm_head(hidden[:, -1:, :])[:, -1, :]
    curr_token = int(mx.argmax(logits, axis=-1).item())

    stats["total_tokens"] += 1
    yield curr_token, False, stats
    if curr_token in stop_tokens or max_tokens <= 1:
        return

    # 2. Autoregressive Decode Loop (M = 1)
    for _ in range(max_tokens - 1):
        x = mx.array([[curr_token]], dtype=mx.int32)
        if cache is not None:
            hidden = model.model(x, cache=cache)
        else:
            hidden = model.model(x)
        logits = model.lm_head(hidden[:, -1:, :])[:, -1, :]
        curr_token = int(mx.argmax(logits, axis=-1).item())

        stats["total_tokens"] += 1
        yield curr_token, False, stats
        if curr_token in stop_tokens:
            break


def generate_speculative(
    model: Any,
    tokenizer: Any,
    prompt: str,
    mode: str = "auto",
    max_tokens: int = 256,
    draft_layers: int = 16,
    draft_model_path: Optional[str] = None,
    verbose: bool = False,
) -> Generator[str, None, None]:
    """
    Unified high-level speculative generation interface.

    Dispatches seamlessly across PLD, Self-Speculative, Micro-Draft, or Native Decode
    based on intelligent heuristics or user specification.

    Args:
        model: Loaded ternary language model.
        tokenizer: Model tokenizer with encode() and decode() methods.
        prompt: Raw text input prompt.
        mode: Speculative mode ('auto', 'pld', 'self', 'qwen', 'none').
        max_tokens: Maximum number of tokens to generate.
        draft_layers: Number of draft layers for 'self' mode (default: 16).
        draft_model_path: Checkpoint identifier for 'qwen' mode.
        verbose: If True, log telemetry and routing decisions to stderr.

    Yields:
        Streaming decoded string chunks (str).
    """
    selected_mode = select_speculative_mode(prompt, mode=mode)
    if verbose:
        print(f"[mlx_ternary router] Speculative routing policy: requested='{mode}', active='{selected_mode}'", file=sys.stderr)

    # 1. Encode prompt
    if hasattr(tokenizer, "encode"):
        enc = tokenizer.encode(prompt)
        prompt_tokens = enc.tolist() if hasattr(enc, "tolist") else list(enc)
    else:
        enc = tokenizer(prompt)
        if isinstance(enc, dict) and "input_ids" in enc:
            enc = enc["input_ids"]
        prompt_tokens = enc.tolist() if hasattr(enc, "tolist") else list(enc)

    # 2. Resolve stop tokens
    stop_tokens: Set[int] = set()
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        stop_tokens.add(int(tokenizer.eos_token_id))
    if hasattr(model, "config") and hasattr(model.config, "eos_token_id"):
        eos = model.config.eos_token_id
        if isinstance(eos, (list, tuple, set)):
            stop_tokens.update(int(t) for t in eos if t is not None)
        elif isinstance(eos, int):
            stop_tokens.add(eos)

    # 3. Instantiate Generator
    generator = None
    if selected_mode == "pld":
        generator = pld_generate(
            model=model,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            stop_tokens=stop_tokens,
        )
    elif selected_mode == "self":
        generator = self_speculative_generate(
            model=model,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            num_draft_layers=draft_layers,
            stop_tokens=stop_tokens,
        )
    elif selected_mode == "qwen":
        draft_path = draft_model_path or "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        draft_model, draft_tokenizer = load_draft_model(draft_path)
        vocab_mapper = VocabularyMapper(tokenizer, draft_tokenizer)
        draft_prompt_tokens = draft_tokenizer.encode(prompt)
        if hasattr(draft_prompt_tokens, "tolist"):
            draft_prompt_tokens = draft_prompt_tokens.tolist()
        generator = draft_speculative_generate(
            target_model=model,
            draft_model=draft_model,
            target_prompt_tokens=prompt_tokens,
            draft_prompt_tokens=list(draft_prompt_tokens),
            vocab_mapper=vocab_mapper,
            max_tokens=max_tokens,
            stop_tokens=stop_tokens,
        )
    else:  # "none"
        generator = native_generate(
            model=model,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            stop_tokens=stop_tokens,
        )

    # 4. Stream tokens
    last_stats = {}
    total_yielded = 0
    for tok, is_accepted, stats in generator:
        chunk = tokenizer.decode([tok]) if hasattr(tokenizer, "decode") else str(tok)
        last_stats = stats
        total_yielded += 1
        yield chunk

    if verbose:
        print(
            f"[mlx_ternary router] Finished generation ({total_yielded} tokens). Active mode: {selected_mode.upper()} | Telemetry: {last_stats}",
            file=sys.stderr,
        )
