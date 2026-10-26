#!/usr/bin/env python3
"""State-of-the-art End-to-End Benchmark for Ternary-Bonsai-2-27B on Apple Silicon.

Measures:
1. Prompt Prefill Time & Throughput (TTFT, prefill tok/s)
2. Autoregressive Generation Throughput (tok/s)
3. Peak Memory Allocation (GB)
4. Full Token Latency Distribution
"""

import sys
import time
import json
from pathlib import Path
import mlx.core as mx

# Setup path to model and runtime
WORKSPACE = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = WORKSPACE / "testing-artifacts" / "models" / "Ternary-Bonsai-2-27B-mlx-2bit"
sys.path.insert(0, str(MODEL_DIR / "runtime"))
sys.path.insert(0, str(WORKSPACE / "mlx-ternary" / "src"))

import mlx_ternary
from artifact import load_model
from tokenizers import Tokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment

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

def run_benchmark(prompt: str, max_tokens: int = 64):
    print("=" * 70)
    print(f"BENCHMARK PROMPT:\n\"{prompt}\"")
    print("=" * 70)
    
    # 1. Load model & tokenizer
    t0 = time.perf_counter()
    print(f"Loading Ternary-Bonsai-2-27B from {MODEL_DIR}...")
    model, config = load_model(MODEL_DIR)
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    load_time = time.perf_counter() - t0
    
    peak_mem_gb = mx.metal.get_peak_memory() / (1024 ** 3)
    curr_mem_gb = mx.metal.get_active_memory() / (1024 ** 3)
    print(f"Model loaded in {load_time:.2f}s | Active Memory: {curr_mem_gb:.2f} GB | Peak: {peak_mem_gb:.2f} GB\n")

    # 2. Tokenize prompt
    rendered = render_prompt(prompt)
    prompt_tokens = tokenizer.encode(rendered, add_special_tokens=False).ids
    prompt_len = len(prompt_tokens)
    print(f"Input Prompt Length: {prompt_len} tokens")

    # 3. Warmup
    print("Warming up JIT Metal kernels...")
    cache = model.make_cache()
    x_warmup = mx.array([prompt_tokens[:8]], dtype=mx.int32)
    logits_w = model.lm_head(model.model(x_warmup, cache=cache)[:, -1:, :])
    mx.eval(logits_w)
    del cache
    
    # 4. Measure Prefill (TTFT)
    mx.metal.reset_peak_memory()
    cache = model.make_cache()
    x_prompt = mx.array([prompt_tokens], dtype=mx.int32)
    
    t_start_prefill = time.perf_counter()
    # Evaluate prefill pass
    hidden = model.model(x_prompt, cache=cache)
    logits = model.lm_head(hidden[:, -1:, :])
    mx.eval(logits)
    t_prefill = time.perf_counter() - t_start_prefill
    
    prefill_tps = prompt_len / t_prefill
    ttft_ms = t_prefill * 1000
    print(f"\n[Prefill Phase]")
    print(f"  * Time To First Token (TTFT): {ttft_ms:.2f} ms")
    print(f"  * Prefill Throughput:         {prefill_tps:.2f} tokens/s")

    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    
    gen_config = json.loads((MODEL_DIR / "generation_config.json").read_text())
    sample = make_sampler(
        temp=gen_config.get("temperature", 0.7),
        top_p=gen_config.get("top_p", 0.95),
        top_k=gen_config.get("top_k", 20),
    )
    processors = make_logits_processors(repetition_penalty=1.1)

    # 5. Measure Autoregressive Generation
    logits_last = logits[:, -1, :]
    for processor in processors:
        logits_last = processor(mx.array([]), logits_last)
    next_token = sample(logits_last)[:, None]
    mx.eval(next_token)
    
    generated_tokens = [int(next_token.item())]
    eos_ids = {tokenizer.token_to_id("<|im_end|>"), tokenizer.token_to_id("<|endoftext|>")}
    
    token_times = []
    print(f"\n[Generation Phase (Target: {max_tokens} tokens)]")
    
    for step in range(max_tokens - 1):
        t_tok_start = time.perf_counter()
        logits = model.lm_head(model.model(next_token, cache=cache)[:, -1:, :])[:, -1, :]
        for processor in processors:
            logits = processor(mx.array(generated_tokens), logits)
        next_token = sample(logits)[:, None]
        mx.eval(next_token)
        t_tok = time.perf_counter() - t_tok_start
        token_times.append(t_tok)
        
        tok_id = int(next_token.item())
        generated_tokens.append(tok_id)
        if tok_id in eos_ids:
            break

            
    total_gen_time = sum(token_times)
    num_generated = len(token_times)
    gen_tps = num_generated / total_gen_time if total_gen_time > 0 else 0
    avg_tok_ms = (total_gen_time / num_generated) * 1000 if num_generated > 0 else 0
    
    peak_mem_gb = mx.metal.get_peak_memory() / (1024 ** 3)
    active_mem_gb = mx.metal.get_active_memory() / (1024 ** 3)
    
    output_text = tokenizer.decode(generated_tokens)
    
    print(f"  * Tokens Generated:        {num_generated + 1}")
    print(f"  * Generation Time:         {total_gen_time:.2f} s")
    print(f"  * Generation Throughput:   {gen_tps:.2f} tokens/s")
    print(f"  * Average Token Latency:   {avg_tok_ms:.2f} ms/token")
    print(f"  * Peak VRAM Usage:         {peak_mem_gb:.2f} GB (Active: {active_mem_gb:.2f} GB)")
    print("\n[Generated Output Sample]:")
    print("-" * 50)
    print(output_text[:300] + ("..." if len(output_text) > 300 else ""))
    print("-" * 50)
    
    return {
        "prompt_tokens": prompt_len,
        "generated_tokens": num_generated + 1,
        "ttft_ms": ttft_ms,
        "prefill_tps": prefill_tps,
        "generation_tps": gen_tps,
        "avg_token_ms": avg_tok_ms,
        "peak_vram_gb": peak_mem_gb,
    }

def main():
    test_prompt = "Explain why Rayleigh scattering causes the sky to be blue and sunsets to be red."
    results = run_benchmark(test_prompt, max_tokens=32)
    
    # Save benchmark results as official artifact
    report_path = WORKSPACE / "testing-artifacts" / "benchmarks" / "latest_e2e_results.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved benchmark results to {report_path}")

if __name__ == "__main__":
    main()
