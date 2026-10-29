# MLX-TERNARY DEVELOPER INVARIANTS

This repository maintains strict engineering invariants modeled after Apple MLX and PyTorch core runtimes. Any contribution, PR, or automated modification must adhere to the following invariants without exception.

---

## 1. Zero-Build Pure-Python Architecture
- The package is distributed as a pure-Python wheel (`py3-none-any.whl`).
- **NEVER** introduce C++, CMake, nanobind, Cython, or native build toolchains.
- All high-performance kernels must be authored in Metal Shading Language (MSL) string templates and compiled dynamically at runtime using `mlx.core.fast.metal_kernel`.

---

## 2. Mathematical Precision & Correctness
- Any modification to kernel dispatch, packing, or numeric accumulation must be validated against reference unpacked matmul in `tests/test_ops.py`.
- Relative error threshold for FP16 accumulation must remain strictly within:
  $$\frac{|Y_{\text{kernel}} - Y_{\text{ref}}|}{|Y_{\text{ref}}| + 10^{-3}} \le 0.03$$
- Numerical tests must be deterministic and fully reproducible across runs (`mx.random.seed(42)`).

---

## 3. Memory Hierarchy & Shape-Aware Kernel Routing
- **Wide Layers ($N > 1024$, e.g. MLP Gate/Up, MLP Down, Attention Q/O):**
  Must stage activation tiles into on-chip SRAM (`threadgroup half shared_act[128]`), streaming 4KB memory pages in coalesced 128-byte DRAM bursts to amortize memory bandwidth.
- **Narrow Layers ($N \le 1024$, e.g. Attention KV):**
  Must route to 32-lane cooperative SIMD warp reduction (`simd_sum`) with 0 threadgroup barriers, eliminating barrier synchronization penalties and ensuring 100% saturation across all GPU execution cores.

---

## 4. Lazy Loading & Minimal Import Footprint
- The core package (`mlx_ternary`, `mlx_ternary.ops`, `mlx_ternary.layers`, `mlx_ternary.quantize`) must import only `mlx.core` and standard library modules.
- **NEVER** import heavy dependencies (`mlx_lm`, `transformers`, `huggingface_hub`) at root module or top-level import time.
- Heavy dependencies must only be imported lazily inside specialized CLI handlers or server execution paths (`cli.py`, `server.py`).

---

## 5. Empirical Honesty & Silicon Telemetry
- Never cite hypothetical silicon, unverified chips, or unmeasured claims in documentation or code.
- Ground all telemetry directly in empirical hardware measurements on physical Apple Silicon (execution time in $\mu\text{s}$, sustained bandwidth in $\text{GB/s}$, and end-to-end tokens/second).
- Every micro-optimization must be proven in benchmark telemetry before promotion to production.
