# System Architecture: `mlx-ternary`

`mlx-ternary` is a specialized inference runtime engineered specifically for 1.58-bit ternary language models on Apple Silicon.

---

## 1. Core Technical Problem

Ternary language models quantize neural network weights to the discrete set $\{-1, 0, +1\}$, achieving an information density of $\approx 1.58\text{ bits per parameter}$. 

To preserve reasoning capabilities, state-of-the-art models (such as Prism ML's Bonsai series) apply a **Fast Walsh-Hadamard Transform (FWHT)** to activations and weights, dispersing outlier feature magnitudes uniformly across channels.

Standard inference engines (like unpatched `llama.cpp` or default `mlx-lm`) cannot execute these checkpoints because:
1. They lack the native Hadamard activation rotation steps.
2. Standard 2-bit affine quantizers misinterpret Base-3 integer trits.
3. Decompressing ternary weights with scalar math destroys memory bandwidth.

---

## 2. Architectural Pillars (ADR 002 Pivot)

### A. Horizon 32-Byte Cacheline Interleaving
Standard quantizers store scales and biases in separate tensor allocations, requiring two distinct DRAM memory requests per matrix tile. 

`mlx-ternary` uses **Horizon Block Interleaving**:
- Each group of 128 (or 150) weights is packed into a contiguous **32-byte memory block**.
- The first **2 bytes** store the FP16 block scale (`float16_t`).
- The remaining **30 bytes** store the Base-3 packed weight trits (5 trits per byte: $3^5 = 243 \le 256$).
- Every weight request perfectly aligns with Apple Silicon's 128-byte L2/DRAM cachelines, achieving zero-stride sequential memory loads.

### B. Hardware Vector ALU (`dot(half4, half4)`)
In standard decode kernels, looking up 5 trits per byte and multiplying activations scalars produces 130 scalar memory lookups per group.

In `mlx-ternary`:
- Base-3 combinations are pre-mapped into constant GPU memory: `base3_lut_low[243]` (`constant half4`) and `base3_lut_high[243]` (`constant half`).
- Activations are loaded as vector quadruplets (`half4`).
- Multiplications execute using Apple Silicon's native **1-cycle vector dot product instruction**:
  $$\text{block\_acc} += \text{dot}(x_{low}, w_{low}) + x_{high} \cdot w_{high}$$

### C. Automated Dual-Dispatch: GEMV vs. AMX GEMM
The runtime automatically switches execution strategies based on prompt sequence length $M$:
- **Autoregressive Single-Token Decode ($M \le 16$):** Dispatches the vectorized coalesced GEMV kernel with **Split-K multi-slice reduction** for narrow projections (`Attention KV`).
- **Prefill & Speculative Verification ($M > 16$):** Dispatches the AMX block GEMM kernel, utilizing Apple Silicon's `simdgroup_matrix<half, 8, 8>` and `simdgroup_multiply_accumulate` matrix hardware.

### D. Pure Python Orchestration
The entire engine compiles its Metal shaders Just-In-Time (JIT) using `mlx.core.fast.metal_kernel`. There are zero static C++ shared libraries, eliminating compilation issues across varying macOS versions.
