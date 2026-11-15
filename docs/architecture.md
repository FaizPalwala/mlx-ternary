# Architecture: `mlx-ternary` Runtime

`mlx-ternary` is a specialized Apple Silicon execution runtime engineered for 1.58-bit ternary language models (such as Prism ML's [Ternary-Bonsai-2-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit)).

This document outlines the core runtime architecture, memory geometry, Metal Shading Language (MSL) kernel designs, and speculative decoding isolation in `v0.6.0`.

---

## 1. System Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                        User Application / CLI                          │
│        mlx-ternary run / serve / generate (Python API: load_model)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              Execution Router & Fused Graph Compiler                   │
│   • Auto-fuses SwiGLU (Gate + Up + Elementwise Multiply)               │
│   • Auto-fuses QKV projections into unified hardware dispatch          │
│   • Dynamic sequence-length dispatch (GEMV decode vs AMX prefill)      │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
      Prefill ($M > 1$)              Autoregressive Decode ($M = 1$)
                    ▼                                ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────┐
│       AMX JIT Fast-Path              │  │      Tier 3 Fused GEMV       │
│ • Dequantizes to contiguous half     │  │ • 1D SRAM activation caching │
│ • Dispatches hardware matrix engine  │  │ • 32-lane SIMD warp reduce   │
│ • Sustains >110 tok/s prefill        │  │ • 128-byte DRAM coalescing   │
└──────────────────────────────────────┘  └──────────────────────────────┘
```

---

## 2. Weight Geometry & Memory Layout

Ternary models restrict weights to $\{-1, 0, +1\}$, yielding a theoretical information entropy of $\log_2(3) \approx 1.58\text{ bits per parameter}$.

### A. Horizon 32-Byte Block Interleaving
Traditional affine quantizers store scales, biases, and packed bits in separated memory tensors. On unified memory architectures, this forces the memory controller to issue multiple non-contiguous memory requests per tile, destroying cache locality.

`mlx-ternary` packs weights into contiguous **32-byte cacheline blocks**:
- **Bytes 0–1:** FP16 Block Scale (`half`).
- **Bytes 2–31:** 30 bytes storing Base-3 packed weight trits (5 trits per byte: $3^5 = 243 \le 256$).
- **Alignment:** Every block load matches Apple Silicon's 128-byte L2 cacheline granularity, ensuring 100% burst bus utilization without stride stalls.

### B. Fast Walsh-Hadamard Transform (FWHT)
To suppress large activation outliers common in low-bit LLMs, the model applies an orthogonal Walsh-Hadamard rotation across hidden dimensions:
$$Y = \frac{1}{\sqrt{d}} H_d \cdot X$$
`mlx-ternary` implements an in-place radix-2 fast butterfly transform in Metal, rotating activations in shared registers with zero intermediate tensor allocations.

---

## 3. Kernel Dispatch & Execution Tiers

The runtime uses shape-aware dispatch tuned for the physical memory subsystem of Apple Silicon:

### A. Wide Projections ($N > 1024$: MLP Gate/Up, MLP Down, Attention O)
- **Problem:** Matrix dimensions are large ($17408 \times 5120$), making activation reloading from DRAM the primary bottleneck.
- **Solution:** 1D Contiguous Column Tiling with on-chip SRAM caching.
  - The threadgroup loads incoming FP16 activations into `threadgroup half shared_act[128]` once.
  - Weights are streamed sequentially in 128-byte coalesced bursts.
  - Sustained memory throughput exceeds **$95.1\text{ GB/s}$** on an Apple M4 Pro.

### B. Narrow Projections ($N \le 1024$: Attention KV)
- **Problem:** Narrow matrix dimensions starve the GPU's execution units if mapped to standard threadgroup grids.
- **Solution:** 32-lane cooperative SIMD reduction (`simd_sum`).
  - Threadgroups execute purely within a hardware warp.
  - Cross-thread reduction uses hardware shuffle registers (`simd_shuffle_down`) with **zero threadgroup barriers**.
  - Eliminates pipeline stalls and guarantees full saturation across all 20 GPU cores on M4 Pro.

### C. Fused Operator Graph
Rather than executing layers as isolated matrix-vector multiplications, `mlx-ternary` fuses adjacent linear projections into single-launch kernels:
- **TernarySwiGLU:** Fuses Gate projection, Up projection, and the $\text{SiLU}(x) \cdot y$ activation into a single kernel dispatch. This halves global memory roundtrips for the MLP block.
- **TernaryFusedQKV:** Computes Query, Key, and Value projections in one contiguous memory sweep.

### D. Prefill Acceleration ($M > 1$)
During prompt evaluation ($M > 1$), memory bandwidth is no longer the sole bottleneck—compute throughput dominates. 
- For prefill, `mlx-ternary` dispatches a JIT dequantization kernel directly into Apple Silicon's AMX (Apple Matrix Coprocessor) block GEMM.
- Delivers **$112 - 116\text{ tok/s}$ prefill throughput**, outperforming official baseline implementations by $1.15\times$.

---

## 4. Speculative Decoding Subsystem

`mlx-ternary` contains a multi-engine speculative decoding harness quarantined behind the `--speculative-experimental` flag.

1. **Prompt Lookup Decoding (PLD):**
   - Extracts n-gram candidates ($K=3..5$) directly from prompt and past KV history using an $O(1)$ dictionary index (`FastNGramIndex`).
   - Requires **0 MB extra VRAM**.
   - Verified via batched GEMM with early-break rejection, eliminating double-forward cache recomputations.
2. **Self-Speculative Layer Pruning (RFC-002):**
   - Executes the first 16 layers of the target model as a lightweight draft proposal.
   - Synchronizes representations using an intermediate `EarlyExitAdapter` ($5120 \to 1024 \to 5120$) to prevent representation drift before projecting through `lm_head`.
   - Requires zero secondary model checkpoints.

---

## 5. Zero-Build Design

`mlx-ternary` does not use static C++ compilation, CMake toolchains, or nanobind wrappers. Shaders are compiled Just-In-Time directly via `mlx.core.fast.metal_kernel`. 

This guarantees:
- Zero ABI mismatches across Python versions (3.10 – 3.12).
- Zero dependency on heavy Xcode installations (only standard macOS runtime required).
- Instant installation via `pip install mlx-ternary`.
