#!/usr/bin/env python3
"""Multi-Turn Agentic Flow & KV Cache Growth Benchmark for Bonsai 2 (27B).

Simulates an agentic execution loop across expanding context lengths:
- Turn 1: 512 tokens (System prompt + Initial Task)
- Turn 2: 1,024 tokens (+First Tool Output)
- Turn 3: 2,048 tokens (+Second Tool Output / Log Inspection)
- Turn 4: 4,096 tokens (+Codebase Context / Multi-Turn History)

Evaluates:
- FP16 vs 8-bit Quantized vs 4-bit Quantized KV Cache
- KV Cache Memory Growth (MB)
- TTFT (Prefill Latency & Throughput)
- Autoregressive Generation Throughput (tokens/second)
- Degradation / Regression Analysis across context milestones
"""

import sys
import time
import json
from pathlib import Path
import mlx.core as mx

WORKSPACE = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = WORKSPACE / "testing-artifacts" / "models" / "Ternary-Bonsai-2-27B-mlx-2bit"
sys.path.insert(0, str(MODEL_DIR / "runtime"))
sys.path.insert(0, str(WORKSPACE / "mlx-ternary" / "src"))

from artifact import load_model
from mlx_lm.models.cache import QuantizedKVCache, KVCache, ArraysCache
from tokenizers import Tokenizer

def make_cache(model, kv_bits=None, group_size=64):
    c = []
    layers = getattr(model, "model", model).layers
    for l in layers:
        if getattr(l, "is_linear", False):
            c.append(ArraysCache(size=2))
        elif kv_bits in (4, 8):
            c.append(QuantizedKVCache(bits=kv_bits, group_size=group_size))
        else:
            c.append(KVCache())
    return c

def get_cache_size_bytes(cache):
    total = 0
    for entry in cache:
        if hasattr(entry, "empty") and entry.empty():
            continue
        if hasattr(entry, "nbytes"):
            try:
                total += entry.nbytes
            except Exception:
                pass
        elif isinstance(entry, KVCache):
            for tensor in [entry.keys, entry.values]:
                if tensor is not None:
                    total += tensor.nbytes
    return total

def benchmark_turn(model, cache, input_ids, gen_tokens=8):
    x = mx.array([input_ids], dtype=mx.int32)
    
    # 1. Prefill step
    t0 = time.perf_counter()
    hidden = model.model(x, cache=cache)
    logits = model.lm_head(hidden[:, -1:, :])
    mx.eval(logits)
    t_prefill = time.perf_counter() - t0
    prefill_tps = len(input_ids) / t_prefill
    
    # 2. Decode step
    next_tok = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
    mx.eval(next_tok)
    
    gen_times = []
    for _ in range(gen_tokens):
        t1 = time.perf_counter()
        hidden = model.model(next_tok, cache=cache)
        logits = model.lm_head(hidden[:, -1:, :])
        next_tok = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
        mx.eval(next_tok)
        gen_times.append(time.perf_counter() - t1)
        
    avg_gen_time = sum(gen_times) / len(gen_times)
    gen_tps = 1.0 / avg_gen_time
    
    cache_bytes = get_cache_size_bytes(cache)
    active_gb = mx.get_active_memory() / (1024 ** 3)
    peak_gb = mx.get_peak_memory() / (1024 ** 3)
    
    return {
        "prefill_ms": t_prefill * 1000,
        "prefill_tps": prefill_tps,
        "gen_tps": gen_tps,
        "gen_ms_tok": avg_gen_time * 1000,
        "cache_mb": cache_bytes / (1024 ** 2),
        "active_vram_gb": active_gb,
        "peak_vram_gb": peak_gb,
    }

def run_agentic_flow_benchmark():
    print("=" * 75)
    print("AGENTIC FLOW & MULTI-TURN CONTEXT GROWTH BENCHMARK")
    print("Model: Ternary-Bonsai-2-27B | Target SoC: Apple M4 Pro (24 GB)")
    print("=" * 75)
    
    print("\nLoading model weights...")
    t0 = time.perf_counter()
    model, _ = load_model(MODEL_DIR)
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s\n")
    
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    
    # Simulated agentic context milestones (tokens)
    milestones = [
        ("Turn 1 (Task Init)", 256),
        ("Turn 2 (+Tool 1 Logs)", 512),
        ("Turn 3 (+Tool 2 Stacktrace)", 1024),
        ("Turn 4 (+Full Context)", 2048),
    ]
    
    cache_modes = [
        ("FP16 (Standard)", None),
        ("8-bit Quantized", 8),
        ("4-bit Quantized", 4),
    ]
    
    summary_results = {}
    
    for mode_name, kv_bits in cache_modes:
        print(f"\n>>> Running Evaluation for KV Cache Mode: {mode_name} <<<")
        print(f"{'Milestone':<28} | {'Context':<8} | {'Cache MB':<10} | {'TTFT (ms)':<10} | {'Prefill TPS':<12} | {'Decode TPS':<10}")
        print("-" * 92)
        
        mode_key = f"kv_{kv_bits if kv_bits else 16}"
        summary_results[mode_key] = []
        
        # In a real conversation, cache is preserved between turns
        cache = make_cache(model, kv_bits=kv_bits, group_size=64)
        
        prev_len = 0
        for name, target_len in milestones:
            # Generate dummy tokens representing the new turn input
            tokens_in_turn = target_len - prev_len
            dummy_turn_tokens = [100 + (i % 5000) for i in range(tokens_in_turn)]
            
            res = benchmark_turn(model, cache, dummy_turn_tokens, gen_tokens=12)
            prev_len = target_len + 12 # Account for generated tokens
            
            print(f"{name:<28} | {target_len:<8} | {res['cache_mb']:<10.1f} | {res['prefill_ms']:<10.1f} | {res['prefill_tps']:<12.1f} | {res['gen_tps']:<10.2f}")
            
            res["milestone"] = name
            res["context_length"] = target_len
            summary_results[mode_key].append(res)
            
        del cache
        mx.clear_cache()

    # Save artifact
    output_path = WORKSPACE / "testing-artifacts" / "benchmarks" / "agentic_flow_results.json"
    output_path.write_text(json.dumps(summary_results, indent=2))
    print("\n" + "=" * 75)
    print(f"Benchmark results successfully saved to: {output_path}")
    print("=" * 75)

if __name__ == "__main__":
    run_agentic_flow_benchmark()
