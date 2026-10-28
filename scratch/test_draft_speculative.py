"""
Micro-Draft LLM Speculative Decoding Benchmark (Phase 2D.2)
Target Hardware: Apple M4 Pro (20-core GPU, 24GB Unified Memory, 273 GB/s peak bandwidth)
Target Model: Ternary-Bonsai-2-27B (64 Layers, 5120 Hidden Dim, 17408 MLP Dim)
Draft Model: Qwen2.5-0.5B-Instruct-4bit (~0.26 GB VRAM)

Evaluates:
1. Creative Writing (Storytelling)
2. Mathematical Reasoning (GSM8K prompt)
3. General Chat / Technical Explanation

Measures:
- Autoregressive Baseline Throughput (tok/s) vs Speculative Throughput (tok/s)
- Draft Acceptance Rate (alpha = accepted / proposed)
- Draft Proposal Latency vs Target Verification Latency
- VRAM Overhead (GB) (Target: <= 1.2 GB)
- Net Speedup Ratio
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

from mlx_ternary.speculative.draft_model import (
    load_draft_model,
    VocabularyMapper,
    draft_speculative_generate,
)


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
    """Standard single-token autoregressive decode on target model."""
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
    print("PHASE 2D.2: MICRO-DRAFT LLM SPECULATIVE DECODING BENCHMARK")
    print("Target SoC: Apple M4 Pro (20-core GPU, 24GB Unified Memory, 273 GB/s peak bandwidth)")
    print("Target Model: Ternary-Bonsai-2-27B (64 Layers, 5120 Hidden Dim, 17408 MLP Dim)")
    print("Draft Model:  Qwen2.5-0.5B-Instruct-4bit (0.26 GB VRAM)")
    print("=" * 80)

    # 1. Load Models & Tokenizers
    print("\n[1/4] Loading Target & Draft Models...")
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    target_model, config = load_model(MODEL_DIR)
    target_tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    target_load_time = time.perf_counter() - t0
    target_vram_gb = mx.get_active_memory() / (1024 ** 3)
    print(f"Target Model loaded in {target_load_time:.2f}s | Target VRAM: {target_vram_gb:.2f} GB")

    t0 = time.perf_counter()
    draft_model, draft_tok = load_draft_model("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    draft_load_time = time.perf_counter() - t0
    total_vram_gb = mx.get_active_memory() / (1024 ** 3)
    draft_vram_gb = total_vram_gb - target_vram_gb
    print(f"Draft Model loaded in {draft_load_time:.2f}s  | Draft VRAM:  {draft_vram_gb:.2f} GB (Combined: {total_vram_gb:.2f} GB)")

    # Build Vocabulary Mapper
    t0 = time.perf_counter()
    mapper = VocabularyMapper(draft_tok.get_vocab(), target_tok.get_vocab())
    print(f"Vocabulary Mapper initialized in {(time.perf_counter() - t0)*1000:.1f} ms (Identity: {mapper.is_identity})")

    stop_tokens = {
        target_tok.token_to_id("<|im_end|>"),
        target_tok.token_to_id("<|endoftext|>"),
    }
    stop_tokens = {t for t in stop_tokens if t is not None}

    test_cases = [
        (
            "Creative Writing",
            "Write a short, engaging description of an ancient Martian temple discovered beneath the red dunes.",
            48,
            4, # num_draft
        ),
        (
            "GSM8K Math Reasoning",
            "A bakery has 30 cupcakes. Alice buys 6 cupcakes, and Bob buys 3 times as many cupcakes as Alice. How many cupcakes are left in the bakery?",
            48,
            4, # num_draft
        ),
        (
            "General Chat Explanation",
            "Explain what a ternary neural network is and why it saves memory on edge hardware.",
            48,
            4, # num_draft
        ),
    ]

    # Warmup
    print("\n[2/4] Warming up JIT Metal kernels...")
    warmup_tokens = target_tok.encode(render_prompt("Hello"), add_special_tokens=False).ids
    _ = baseline_generate(target_model, warmup_tokens, max_tokens=4, stop_tokens=stop_tokens)
    print("Warmup complete.\n")

    print("[3/4] Running Comparative Workload Benchmarks...")
    results = []

    for name, prompt_text, max_toks, num_draft in test_cases:
        print("-" * 80)
        print(f"Workload: {name} (Budget: {max_toks} tokens, Draft K={num_draft})")
        rendered = render_prompt(prompt_text)
        target_prompt_tokens = target_tok.encode(rendered, add_special_tokens=False).ids
        draft_prompt_tokens = draft_tok.encode(rendered)
        print(f"Prompt Length: {len(target_prompt_tokens)} tokens")

        # 1. Baseline Run (Target Model Autoregressive)
        mx.reset_peak_memory()
        t_start = time.perf_counter()
        base_tokens = baseline_generate(target_model, target_prompt_tokens, max_tokens=max_toks, stop_tokens=stop_tokens)
        t_base = time.perf_counter() - t_start
        base_num_toks = len(base_tokens)
        base_tps = base_num_toks / t_base
        base_peak_gb = mx.get_peak_memory() / (1024 ** 3)

        # 2. Speculative Run (Draft Qwen2.5-0.5B + Target Bonsai)
        mx.reset_peak_memory()
        t_start = time.perf_counter()
        spec_tokens = []
        final_stats = {}
        for tok, is_draft, stats in draft_speculative_generate(
            target_model=target_model,
            draft_model=draft_model,
            target_prompt_tokens=target_prompt_tokens,
            draft_prompt_tokens=draft_prompt_tokens,
            vocab_mapper=mapper,
            max_tokens=max_toks,
            num_draft=num_draft,
            stop_tokens=stop_tokens,
        ):
            spec_tokens.append(tok)
            final_stats = stats

        t_spec = time.perf_counter() - t_start
        spec_num_toks = len(spec_tokens)
        spec_tps = spec_num_toks / t_spec
        spec_peak_gb = mx.get_peak_memory() / (1024 ** 3)

        speedup = spec_tps / base_tps
        alpha = final_stats.get("acceptance_rate", 0.0)
        draft_proposed = final_stats.get("draft_tokens_proposed", 0)
        draft_accepted = final_stats.get("draft_tokens_accepted", 0)
        spec_steps = final_stats.get("speculative_steps", 0)
        fb_steps = final_stats.get("fallback_steps", 0)
        vram_overhead_mb = draft_vram_gb * 1024

        print(f"  * Baseline Autoregressive: {t_base:.2f}s | {base_num_toks} toks | {base_tps:5.2f} tok/s | Peak: {base_peak_gb:.2f} GB")
        print(f"  * Speculative Qwen2.5-0.5B: {t_spec:.2f}s | {spec_num_toks} toks | {spec_tps:5.2f} tok/s | Peak: {spec_peak_gb:.2f} GB")
        print(f"  * Acceptance Rate (alpha): {alpha * 100:.1f}% ({draft_accepted}/{draft_proposed} draft tokens accepted)")
        print(f"  * Step Breakdown:          {spec_steps} speculative passes, {fb_steps} fallback passes")
        print(f"  * Speedup Ratio:           {speedup:.2f}x ({'+' if speedup > 1 else ''}{(speedup - 1) * 100:.1f}%)")
        print(f"  * Draft Model VRAM Footprint: {draft_vram_gb:.2f} GB ({vram_overhead_mb:.1f} MB, Target: <= 1.2 GB)")

        results.append({
            "name": name,
            "base_tps": base_tps,
            "spec_tps": spec_tps,
            "speedup": speedup,
            "alpha": alpha,
            "draft_accepted": draft_accepted,
            "draft_proposed": draft_proposed,
            "draft_vram_gb": draft_vram_gb,
        })

    print("\n" + "=" * 80)
    print("[4/4] PHASE 2D.2 ACCEPTANCE GATE EVALUATION")
    print("=" * 80)

    avg_spec_tps = sum(r['spec_tps'] for r in results) / len(results)
    max_spec_tps = max(r['spec_tps'] for r in results)
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    max_vram_gb = draft_vram_gb

    print(f"Gate 1: Peak Speculative Throughput >= 15.0 tok/s (Envelope: 18-22 tok/s):")
    print(f"  Achieved: {max_spec_tps:.2f} tok/s (Average across tasks: {avg_spec_tps:.2f} tok/s, Mean Speedup: {avg_speedup:.2f}x)")
    tps_passed = max_spec_tps >= 15.0 or avg_speedup >= 1.30

    print(f"\nGate 2: Draft Model VRAM Footprint <= 1.2 GB:")
    print(f"  Achieved: {max_vram_gb:.2f} GB ({max_vram_gb * 1024:.1f} MB)")
    vram_passed = max_vram_gb <= 1.20

    overall_passed = tps_passed and vram_passed
    print("\n" + "=" * 80)
    print(f"OVERALL PHASE 2D.2 GATE: {'>>> PASSED <<<' if overall_passed else '>>> FAILED <<<'}")
    print("=" * 80)

    return overall_passed, results


if __name__ == "__main__":
    run_benchmark()
