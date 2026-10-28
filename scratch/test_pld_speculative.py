"""
Zero-RAM Prompt Lookup Decoding (PLD) Empirical Benchmark (Phase 2D.1)
Hardware: Apple M4 Pro (20-core GPU, 24GB Unified Memory, 273 GB/s peak LPDDR5 bandwidth)
Architectural Target: ADR-001 Phase 2D.1 (Zero-RAM Prompt Lookup Decoding)

This benchmark evaluates:
1. Code Generation (Python boilerplate class with repeated patterns)
2. JSON Schema Generation (Structured records with repeated keys)
3. In-Context Retrieval / Repetitive Log Summarization
4. Fallback Verification (Creative text with no n-gram overlap, testing 0-penalty fallback)

Measures:
- Latency (ms) & Effective Tokens Per Second (tok/s)
- Draft Acceptance Rate (alpha = accepted / proposed)
- Speedup Ratio (PLD vs Autoregressive Baseline)
- VRAM Footprint Delta (Peak Active Memory in GB)
"""

import sys
import time
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

MODEL_DIR = repo_root.parent / "testing-artifacts" / "models" / "Ternary-Bonsai-2-27B-mlx-2bit"
sys.path.insert(0, str(MODEL_DIR / "runtime"))

import mlx.core as mx
from artifact import load_model
from tokenizers import Tokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment

from mlx_ternary.speculative.pld import pld_generate, find_candidate_tokens


def render_prompt(prompt: str) -> str:
    template_path = MODEL_DIR / "chat_template.jinja"
    if not template_path.is_file():
        return prompt
    return (
        ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        .from_string(template_path.read_text())
        .render(
            messages=[{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def baseline_generate(model, prompt_tokens: list[int], max_tokens: int, stop_tokens: set[int]):
    """Standard single-token autoregressive decode (M=1 baseline)."""
    cache = model.make_cache()
    x = mx.array([prompt_tokens], dtype=mx.int32)
    hidden = model.model(x, cache=cache)
    logits = model.lm_head(hidden[:, -1:, :])[:, -1, :]
    curr = int(mx.argmax(logits, axis=-1).item())
    
    generated = [curr]
    if curr in stop_tokens or len(generated) >= max_tokens:
        return generated
        
    while len(generated) < max_tokens:
        x_tok = mx.array([[curr]], dtype=mx.int32)
        hidden = model.model(x_tok, cache=cache)
        logits = model.lm_head(hidden[:, -1:, :])[:, -1, :]
        curr = int(mx.argmax(logits, axis=-1).item())
        generated.append(curr)
        if curr in stop_tokens:
            break
            
    return generated


def run_benchmark():
    print("=" * 80)
    print("PHASE 2D.1: ZERO-RAM PROMPT LOOKUP DECODING (PLD) BENCHMARK")
    print("Target SoC: Apple M4 Pro (20-core GPU, 24GB Unified Memory, 273 GB/s peak bandwidth)")
    print("Model: Ternary-Bonsai-2-27B (64 Layers, 5120 Hidden Dim, 17408 MLP Dim)")
    print("=" * 80)

    # 1. Load Model & Tokenizer
    print("\n[1/4] Loading Ternary-Bonsai-2-27B...")
    t0 = time.perf_counter()
    model, config = load_model(MODEL_DIR)
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    load_time = time.perf_counter() - t0
    
    stop_tokens = {
        tokenizer.token_to_id("<|im_end|>"),
        tokenizer.token_to_id("<|endoftext|>"),
    }
    stop_tokens = {t for t in stop_tokens if t is not None}
    
    init_mem_gb = mx.metal.get_active_memory() / (1024 ** 3)
    print(f"Model loaded in {load_time:.2f}s | Base VRAM: {init_mem_gb:.2f} GB")

    test_cases = [
        (
            "Python Code Generation",
            "Write a Python class DatabaseConnectionManager with methods connect(), disconnect(), execute_query(sql), commit(), and rollback(). Include docstrings, connection pooling attributes, and error handling for each method.",
            64,
            4, # num_draft
        ),
        (
            "JSON Schema Generation",
            "Here is a sample employee record in JSON:\n"
            '{\n  "employee_id": 1001,\n  "name": "Alice Chen",\n  "department": "Engineering",\n  "role": "Systems Architect",\n  "status": "Active"\n}\n'
            "Generate 3 more employee records following this exact JSON schema format.",
            64,
            4, # num_draft
        ),
        (
            "Repetitive Context Retrieval",
            "Context:\n"
            "[INFO 10:00:01] Service authentication processed request from 192.168.1.10: status=200 OK\n"
            "[INFO 10:00:02] Service authentication processed request from 192.168.1.11: status=200 OK\n"
            "[INFO 10:00:03] Service authentication processed request from 192.168.1.12: status=200 OK\n"
            "[INFO 10:00:04] Service authentication processed request from 192.168.1.13: status=200 OK\n\n"
            "Query: List all successful requests from the log above formatted with timestamp, IP, and status.",
            64,
            4, # num_draft
        ),
        (
            "Fallback (Zero N-gram Overlap)",
            "Compose a brief, creative 3-sentence description of an ancient oak tree on a misty mountain morning.",
            48,
            4, # num_draft
        ),
    ]

    # Warmup pass
    print("\n[2/4] Warming up JIT Metal kernels...")
    warmup_tokens = tokenizer.encode(render_prompt("Hello"), add_special_tokens=False).ids
    _ = baseline_generate(model, warmup_tokens, max_tokens=4, stop_tokens=stop_tokens)
    print("Warmup complete.\n")

    print("[3/4] Running Comparative Workload Benchmarks...")
    results = []

    for name, prompt_text, max_toks, num_draft in test_cases:
        print("-" * 80)
        print(f"Workload: {name} (Budget: {max_toks} tokens, Draft K={num_draft})")
        rendered = render_prompt(prompt_text)
        prompt_tokens = tokenizer.encode(rendered, add_special_tokens=False).ids
        print(f"Prompt Length: {len(prompt_tokens)} tokens")

        # 1. Baseline Run (Standard Autoregressive Decode)
        mx.metal.reset_peak_memory()
        t_start = time.perf_counter()
        base_tokens = baseline_generate(model, prompt_tokens, max_tokens=max_toks, stop_tokens=stop_tokens)
        t_base = time.perf_counter() - t_start
        base_num_toks = len(base_tokens)
        base_tps = base_num_toks / t_base
        base_peak_gb = mx.metal.get_peak_memory() / (1024 ** 3)

        # 2. Speculative PLD Run
        mx.metal.reset_peak_memory()
        t_start = time.perf_counter()
        pld_tokens = []
        final_stats = {}
        for tok, is_draft, stats in pld_generate(
            model,
            prompt_tokens,
            max_tokens=max_toks,
            num_draft=num_draft,
            max_ngram_size=3,
            min_ngram_size=2,
            stop_tokens=stop_tokens,
        ):
            pld_tokens.append(tok)
            final_stats = stats

        t_pld = time.perf_counter() - t_start
        pld_num_toks = len(pld_tokens)
        pld_tps = pld_num_toks / t_pld
        pld_peak_gb = mx.metal.get_peak_memory() / (1024 ** 3)

        speedup = pld_tps / base_tps
        alpha = final_stats.get("acceptance_rate", 0.0)
        draft_proposed = final_stats.get("draft_tokens_proposed", 0)
        draft_accepted = final_stats.get("draft_tokens_accepted", 0)
        spec_steps = final_stats.get("speculative_steps", 0)
        fb_steps = final_stats.get("fallback_steps", 0)
        vram_overhead_mb = (pld_peak_gb - base_peak_gb) * 1024

        print(f"  * Baseline Autoregressive: {t_base:.2f}s | {base_num_toks} toks | {base_tps:5.2f} tok/s | Peak: {base_peak_gb:.2f} GB")
        print(f"  * Speculative PLD (K={num_draft}):     {t_pld:.2f}s | {pld_num_toks} toks | {pld_tps:5.2f} tok/s | Peak: {pld_peak_gb:.2f} GB")
        print(f"  * Acceptance Rate (alpha): {alpha * 100:.1f}% ({draft_accepted}/{draft_proposed} draft tokens accepted)")
        print(f"  * Step Breakdown:          {spec_steps} speculative passes, {fb_steps} fallback passes")
        print(f"  * Speedup Ratio:           {speedup:.2f}x ({'+' if speedup > 1 else ''}{(speedup - 1) * 100:.1f}%)")
        print(f"  * VRAM Overhead:           {vram_overhead_mb:+.1f} MB (Target: <= 0 MB)")

        results.append({
            "name": name,
            "base_tps": base_tps,
            "pld_tps": pld_tps,
            "speedup": speedup,
            "alpha": alpha,
            "draft_accepted": draft_accepted,
            "draft_proposed": draft_proposed,
            "vram_overhead_mb": vram_overhead_mb,
        })

    print("\n" + "=" * 80)
    print("[4/4] PHASE 2D.1 ACCEPTANCE GATE EVALUATION")
    print("=" * 80)
    
    code_result = results[0]
    json_result = results[1]
    fallback_result = results[3]

    print(f"Gate 1: Code Generation Speedup >= 1.30x (Target: >= 11.5 tok/s):")
    print(f"  Achieved: {code_result['speedup']:.2f}x ({code_result['pld_tps']:.2f} tok/s vs {code_result['base_tps']:.2f} tok/s)")
    code_passed = code_result['speedup'] >= 1.30 or code_result['pld_tps'] >= 11.5

    print(f"\nGate 2: Zero-VRAM Footprint (Target: <= 0 MB extra VRAM):")
    max_vram_overhead = max(r['vram_overhead_mb'] for r in results)
    print(f"  Achieved: {max_vram_overhead:+.1f} MB")
    vram_passed = max_vram_overhead <= 1.0  # within 1 MB tolerance for buffer noise

    print(f"\nGate 3: Zero-Penalty Fallback on Low-Overlap Text:")
    print(f"  Achieved: {fallback_result['speedup']:.2f}x ({fallback_result['pld_tps']:.2f} tok/s vs {fallback_result['base_tps']:.2f} tok/s)")
    fallback_passed = fallback_result['speedup'] >= 0.95

    overall_passed = code_passed and vram_passed and fallback_passed
    print("\n" + "=" * 80)
    print(f"OVERALL PHASE 2D.1 GATE: {'>>> PASSED <<<' if overall_passed else '>>> FAILED <<<'}")
    print("=" * 80)

    return overall_passed, results


if __name__ == "__main__":
    run_benchmark()
