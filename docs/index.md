# MLX Ternary

**Native, high-performance Apple Silicon runtime for 1.58-bit ternary language models.**

`mlx-ternary` enables extreme-compression large language models (such as Prism ML's Ternary-Bonsai series) to run natively at full speed on Apple Silicon (M1 through M5/M6) by utilizing pure Metal JIT kernels, cacheline-interleaved memory packing, and speculative verification.

---

## Key Features

- **Hardware Vector Acceleration:** Native Metal vector arithmetic (`dot(half4, half4)`) and constant memory Base-3 decode tables eliminate scalar dequantization bottlenecks.
- **Split-K Dynamic Reduction:** Automatically parallelizes narrow transformer layers (such as Key/Value projections) across all GPU execution cores, preventing GPU core starvation on M4/M5 architectures.
- **AMX Coprocessor GEMM:** Dedicated `simdgroup_matrix<half, 8, 8>` block GEMM for high-throughput prefill and speculative verification sweeps.
- **Configurable KV Cache Quantization:** First-class support for FP16, 8-bit (`QuantizedKVCache`), and 4-bit KV caching, shrinking attention memory footprints by up to $4\times$ to sustain 32k–64k agentic contexts on 24 GB MacBooks.
- **Zero-Build Installation:** 100% pure Python orchestration via `mlx.core.fast.metal_kernel`. Zero static C++ dependencies, zero CMake compilation, zero nanobind headaches.
- **OpenAI-Compatible Server:** Drop-in compatibility with **LM Studio**, **Ollama** frontends, and **OpenWebUI**.

---

## Quickstart

### Installation

```bash
pip install mlx-ternary
```

Or install from source in editable mode:

```bash
git clone https://github.com/base3labs/mlx-ternary.git
cd mlx-ternary
pip install -e .
```

### Launching the Server

Serve a ternary model locally with an OpenAI-compatible API:

```bash
# Serve with standard FP16 KV cache on port 8080
mlx_ternary serve prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --port 8080

# Serve with 4-bit KV cache quantization for long agentic contexts
mlx_ternary serve prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --kv-bits 4 --port 8080

# Serve with Speculative Decoding enabled (using a 1.5B draft model)
mlx_ternary serve prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --speculate --port 8080
```
