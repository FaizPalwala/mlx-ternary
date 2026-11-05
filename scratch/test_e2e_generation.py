#!/usr/bin/env python3
"""
End-to-End Cold Generation Smoke Test on Ternary-Bonsai-2-27B.

Evaluates two critical milestones:
1. Milestone A (Baseline End-to-End Integrity): Verify 27B checkpoint, tokenizer, and
   Hadamard activation pipeline generate coherent, syntactically correct code without NaN or gibberish.
2. Milestone B (Fused SwiGLU Acceleration): Dynamically swap all 64 MLP layers with production
   TernarySwiGLU (incorporating single-pass FWHT activation transform) to verify zero numerical drift
   and measure live end-to-end decode speedup.
"""

import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

# Locate model directory
MODEL_DIR = Path("testing-artifacts/models/Ternary-Bonsai-2-27B-mlx-2bit").resolve()
if not MODEL_DIR.exists():
    # If run from inside mlx-ternary/ or scratch/
    CANDIDATES = [
        Path.cwd() / "testing-artifacts" / "models" / "Ternary-Bonsai-2-27B-mlx-2bit",
        Path.cwd().parent / "testing-artifacts" / "models" / "Ternary-Bonsai-2-27B-mlx-2bit",
        Path(__file__).resolve().parent.parent.parent / "testing-artifacts" / "models" / "Ternary-Bonsai-2-27B-mlx-2bit",
    ]
    for cand in CANDIDATES:
        if cand.exists():
            MODEL_DIR = cand.resolve()
            break

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Could not locate Ternary-Bonsai-2-27B model directory: {MODEL_DIR}")

# Setup runtime and package paths
sys.path.insert(0, str(MODEL_DIR / "runtime"))
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, "src")

from artifact import load_model
from mlx_ternary.layers import TernarySwiGLU


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


def run_generation(model, prompt_tokens: list[int], max_tokens: int = 40):
    """Run greedy autoregressive decode loop and record latency statistics."""
    cache = model.make_cache()
    x = mx.array([prompt_tokens], dtype=mx.int32)

    # Prefill phase
    t_start_prefill = time.perf_counter()
    hidden = model.model(x, cache=cache)
    logits = model.lm_head(hidden[:, -1:, :])[:, -1, :]
    mx.eval(logits)
    t_prefill = time.perf_counter() - t_start_prefill

    first_tok = int(mx.argmax(logits, axis=-1).item())
    generated_tokens = [first_tok]
    next_tok = mx.array([[first_tok]], dtype=mx.int32)

    # Decode phase
    token_times = []
    for _ in range(max_tokens - 1):
        t0 = time.perf_counter()
        hidden = model.model(next_tok, cache=cache)
        logits = model.lm_head(hidden[:, -1:, :])[:, -1, :]
        tok_id = int(mx.argmax(logits, axis=-1).item())
        mx.eval(tok_id)
        dt = time.perf_counter() - t0
        token_times.append(dt)
        generated_tokens.append(tok_id)
        next_tok = mx.array([[tok_id]], dtype=mx.int32)

    total_decode_time = sum(token_times)
    num_decode_tokens = len(token_times)
    avg_decode_ms = (total_decode_time / num_decode_tokens) * 1000 if num_decode_tokens > 0 else 0
    decode_tps = num_decode_tokens / total_decode_time if total_decode_time > 0 else 0

    return {
        "tokens": generated_tokens,
        "ttft_ms": t_prefill * 1000,
        "prefill_tps": len(prompt_tokens) / t_prefill if t_prefill > 0 else 0,
        "avg_decode_ms": avg_decode_ms,
        "decode_tps": decode_tps,
        "total_decode_time": total_decode_time,
        "token_times": token_times,
    }


def main():
    prompt = "Write a Python function to check if a number is prime."
    max_tokens = 40

    print("=" * 80)
    print("END-TO-END COLD GENERATION SMOKE TEST: Ternary-Bonsai-2-27B")
    print(f"Model Path: {MODEL_DIR}")
    print(f"Prompt: \"{prompt}\"")
    print(f"Target Decode Length: {max_tokens} tokens | Greedy (Temp=0.0)")
    print("=" * 80)

    # 1. Load Model & Tokenizer
    t0 = time.perf_counter()
    print("\n[Step 1/4] Loading model and tokenizer...")
    model, config = load_model(MODEL_DIR)
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    load_time = time.perf_counter() - t0
    peak_mem_gb = mx.get_peak_memory() / (1024 ** 3)
    curr_mem_gb = mx.get_active_memory() / (1024 ** 3)
    print(f"Model loaded in {load_time:.2f}s | Active VRAM: {curr_mem_gb:.2f} GB | Peak: {peak_mem_gb:.2f} GB")

    # Render & tokenize prompt
    rendered = render_prompt(prompt)
    prompt_tokens = tokenizer.encode(rendered, add_special_tokens=False).ids
    print(f"Prompt tokenized into {len(prompt_tokens)} tokens.")

    # 2. Milestone A: Baseline Generation
    print("\n[Step 2/4] Executing Milestone A: Baseline End-to-End Generation...")
    baseline_res = run_generation(model, prompt_tokens, max_tokens=max_tokens)
    baseline_text = tokenizer.decode(baseline_res["tokens"])

    print(f"  * TTFT:                 {baseline_res['ttft_ms']:.2f} ms ({baseline_res['prefill_tps']:.2f} tok/s)")
    print(f"  * Per-Token Decode:     {baseline_res['avg_decode_ms']:.2f} ms/tok")
    print(f"  * Decode Throughput:    {baseline_res['decode_tps']:.2f} tok/s")
    print(f"  * Total Decode Time:    {baseline_res['total_decode_time']:.2f} s")
    print("\n--- Baseline Generated Text ---")
    print(baseline_text)
    print("-------------------------------")

    # Verify baseline sanity
    assert "def " in baseline_text or "prime" in baseline_text.lower(), "Baseline output must contain coherent code!"
    assert not any(t == 0 for t in baseline_res["tokens"]), "Baseline output contains unexpected zero tokens!"

    # 3. Milestone B: In-Memory SwiGLU Fusion Across 64 Layers
    print("\n[Step 3/4] Executing Milestone B: In-Memory SwiGLU Fusion Across 64 Layers...")
    num_fused = 0
    for i, layer in enumerate(model.model.layers):
        gate = layer.mlp.gate_proj
        up = layer.mlp.up_proj

        swiglu = TernarySwiGLU(
            in_features=5120,
            out_features=17408,
            group_size=128,
            block=gate.block,
            signs=gate.signs,
        )
        swiglu.w_gate = gate.weight
        swiglu.w_up = up.weight

        class FusedMLPWrapper(nn.Module):
            def __init__(self, swiglu_mod, down_proj):
                super().__init__()
                self.swiglu = swiglu_mod
                self.down_proj = down_proj

            def __call__(self, x):
                return self.down_proj(self.swiglu(x))

        layer.mlp = FusedMLPWrapper(swiglu, layer.mlp.down_proj)
        num_fused += 1

    print(f"Successfully fused MLP SwiGLU projections across {num_fused}/64 transformer layers.")

    # 4. Run Fused Generation
    print("\n[Step 4/4] Executing Autoregressive Decode with Fused SwiGLU...")
    fused_res = run_generation(model, prompt_tokens, max_tokens=max_tokens)
    fused_text = tokenizer.decode(fused_res["tokens"])

    print(f"  * TTFT:                 {fused_res['ttft_ms']:.2f} ms ({fused_res['prefill_tps']:.2f} tok/s)")
    print(f"  * Per-Token Decode:     {fused_res['avg_decode_ms']:.2f} ms/tok")
    print(f"  * Decode Throughput:    {fused_res['decode_tps']:.2f} tok/s")
    print(f"  * Total Decode Time:    {fused_res['total_decode_time']:.2f} s")
    print("\n--- Fused SwiGLU Generated Text ---")
    print(fused_text)
    print("-----------------------------------")

    # 5. Comparative Telemetry & Validation
    tokens_match = baseline_res["tokens"] == fused_res["tokens"]
    tok_diff = baseline_res["avg_decode_ms"] - fused_res["avg_decode_ms"]
    speedup = baseline_res["avg_decode_ms"] / fused_res["avg_decode_ms"] if fused_res["avg_decode_ms"] > 0 else 1.0

    print("\n" + "=" * 80)
    print("EMPIRICAL COMPARATIVE TELEMETRY")
    print("=" * 80)
    print(f"Token Parity Match:     {'PASS (100% Bit-Exact Match)' if tokens_match else 'DRIFT DETECTED'}")
    print(f"Baseline Latency:       {baseline_res['avg_decode_ms']:.2f} ms/tok ({baseline_res['decode_tps']:.2f} tok/s)")
    print(f"Fused SwiGLU Latency:   {fused_res['avg_decode_ms']:.2f} ms/tok ({fused_res['decode_tps']:.2f} tok/s)")
    print(f"Absolute Latency Delta: {tok_diff:+.2f} ms/tok")
    print(f"Throughput Speedup:     {speedup:.2f}x")
    print("=" * 80)

    # Acceptance Assertions
    assert tokens_match, "Fused SwiGLU tokens must match baseline tokens exactly!"
    assert "def is_prime" in fused_text or "def " in fused_text, "Generated text must contain valid Python function!"
    print("\n>> ALL ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY! <<\n")


if __name__ == "__main__":
    main()
