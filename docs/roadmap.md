# Project Roadmap: `mlx-ternary`

This document outlines the engineering roadmap, released features, and planned milestones for `mlx-ternary`.

---

## Current Release: v0.6.0 (Production Core)

*Status: Shipped (November 2026)*

The focus of `v0.6.0` is establishing a rock-solid, production-grade baseline for 1.58-bit inference on Apple Silicon with zero memory leaks and 100% numerical parity.

- [x] **Tier 3 Fused SOTA Kernels:**
  - `TernarySwiGLU`: Fused Gate + Up + SiLU elementwise multiplication.
  - `TernaryFusedQKV`: Contiguous single-sweep attention projection.
  - 32-lane cooperative SIMD reduction with 0 threadgroup barriers for narrow Attention KV layers ($N \le 1024$).
  - 1D SRAM activation caching for wide MLP layers ($N > 1024$), sustaining $95.1\text{ GB/s}$ memory bandwidth.
- [x] **AMX JIT Prefill Fast-Path:**
  - JIT AMX matrix-vector acceleration sustaining **$112 - 116\text{ tok/s}$** on prompt evaluation ($1.15\times$ faster than official baseline).
- [x] **Physical Memory Hardening:**
  - Flat **$6.37\text{ GB}$ Unified Memory footprint** on 27.2B models ($0.78\text{ GB}$ lighter than official baseline, $0.00\text{ GB}$ memory leaks).
- [x] **Bit-Exact Parity:**
  - 100% bit-exact output match against official Bonsai reference implementations.
- [x] **Quarantined Speculative Research Track:**
  - Speculative decoding isolated behind `--speculative-experimental [pld|self|auto]` with terminal warning banner.
  - Implemented $O(1)$ n-gram lookup (`FastNGramIndex`) and early-break sequential GEMV verification.
  - Trained RFC-002 `EarlyExitAdapter` for 16-layer self-speculative draft proposal.
- [x] **Developer Experience & CLI:**
  - Dual entrypoints (`mlx-ternary` and `mlx_ternary`).
  - Zero-build JIT installation (`pip install mlx-ternary`).
  - Silicon hardware probe CLI (`mlx-ternary bench`).

---

## Upcoming Milestone: v0.7.0 (Performance Expansion)

*Target: Q1 2027*

The focus of `v0.7.0` is pushing memory bus efficiency from $70 - 95\text{ GB/s}$ toward the physical bandwidth ceiling ($>140\text{ GB/s}$) via Pure-ALU decompression and register pressure reduction.

- [ ] **Pure-ALU Bit-Shifting & Masking:**
  - Replace constant-memory LUT lookups (`base3_lut`) with pure ALU shift-and-mask arithmetic.
  - Eliminate constant cache bottlenecks and register spills during high-occupancy threadgroup dispatch.
- [ ] **Sub-Byte Bitplane Packing:**
  - Alternative 2-bit sign/magnitude bitplane layout for sub-warp cooperative execution.
  - Hardware popcount-assisted accumulation for binary and ternary sub-matrices.
- [ ] **Distilled Semantic Exit-Heads:**
  - Pre-trained, distilled exit adapters for layers 8, 16, and 24 to increase speculative acceptance rates on open-ended prose.
- [ ] **Speculative Auto-Tuning:**
  - Dynamic run-time heuristic that monitors token acceptance rate $\alpha$ and automatically engages or disables speculation per sequence without user intervention.

---

## Future Horizon: v0.8.0 (Enterprise Serving & Multi-Model)

*Target: Mid 2027*

Scaling `mlx-ternary` for multi-tenant, multi-agent local production environments:

- [ ] **Continuous Batching & Paged KV Cache:**
  - Paged memory management for attention KV tensors to prevent unified memory fragmentation across long-running multi-agent sessions.
  - Continuous token batching for the OpenAI-compatible HTTP server (`mlx-ternary serve`).
- [ ] **Multi-Model In-Memory Serving:**
  - Exploit the tiny $6.37\text{ GB}$ footprint to host multiple 27B specialized ternary checkpoints simultaneously in a single 24 GB / 36 GB Mac.
- [ ] **4-Bit KV Cache Quantization:**
  - Integration of quantized KV cache to support 64k+ context windows under $10\text{ GB}$ total RAM.
