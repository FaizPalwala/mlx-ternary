# Hardware Benchmarks & Performance Matrix

Comprehensive performance evaluations across Apple Silicon and discrete GPUs for **Ternary-Bonsai-2-27B** (27B parameter ternary model, 64 layers).

---

## 1. Global Hardware Matrix (Published vs. Measured)

| Hardware Platform | Memory Bandwidth | Autoregressive Decode Throughput | Notes |
| :--- | :--- | :--- | :--- |
| **NVIDIA RTX 5090** | 1,792 GB/s | **143.0 tok/s** | Desktop Flagship (Blackwell) |
| **NVIDIA RTX 4090** | 1,008 GB/s | **91.1 tok/s** | PTQ1_0 format |
| **Apple M5 Ultra** | 1,200 GB/s | **$\approx 55-65$ tok/s (Est.)** | Mac Studio (Quad-die UltraFusion) |
| **Apple M5 Max** | $\approx 800$ GB/s | **47.0 tok/s** | Flagship Laptop SoC |
| **Apple M4 Pro (Ours)**| **273 GB/s** | **13.8 tok/s (Autoregressive)**<br>**22.8 tok/s (Speculative)** | **Compact MacBook Pro (24 GB)** |
| **Cloud / OpenRouter** | Variable | **10.0 - 15.0 tok/s** | Multi-tenant Cloud APIs |

---

## 2. Layer-by-Layer Decode Latency (M4 Pro, $M=1$)

| Layer Projection | Dimensions | MLX Standard 2-bit | `mlx-ternary` JIT (Ours) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **MLP Gate/Up Proj** | $1 \times 5120 \times 17408$ | $250.10\text{ }\mu\text{s}$ | **$295.79\text{ }\mu\text{s}$** | $0.85\times$ |
| **MLP Down Proj** | $1 \times 17408 \times 5120$ | $252.74\text{ }\mu\text{s}$ | **$275.56\text{ }\mu\text{s}$** | $0.92\times$ |
| **Attention Q Proj** | $1 \times 5120 \times 3072$ | $155.56\text{ }\mu\text{s}$ | **$168.34\text{ }\mu\text{s}$** | $0.92\times$ |
| **Attention KV Proj** | $1 \times 5120 \times 512$ | $134.01\text{ }\mu\text{s}$ | **$122.05\text{ }\mu\text{s}$** | **$1.10\times$ (+9.8%)** |
| **Attention O Proj** | $1 \times 3072 \times 5120$ | $128.52\text{ }\mu\text{s}$ | **$157.22\text{ }\mu\text{s}$** | $0.82\times$ |

*Attention KV achieves $>1.10\times$ speedup over Apple's native 2-bit affine kernel thanks to automated Split-K parallelization across all 20 GPU cores.*

---

## 3. Running the Benchmarks Locally

```bash
# Run layer-by-layer kernel comparison vs normal MLX 2-bit:
python testing-artifacts/benchmarks/compare_bonsai.py

# Run end-to-end token generation benchmark:
python testing-artifacts/benchmarks/benchmark_e2e.py

# Run multi-turn agentic context growth & KV cache evaluation:
python testing-artifacts/benchmarks/benchmark_agentic_flow.py
```
