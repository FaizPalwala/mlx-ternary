# mlx-ternary

[![PyPI Version](https://img.shields.io/pypi/v/mlx-ternary.svg)](https://pypi.org/project/mlx-ternary/)
[![Release](https://img.shields.io/badge/release-v0.6.0-blue.svg)](https://github.com/FaizPalwala/mlx-ternary/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M-series-black.svg)]()
[![Metal](https://img.shields.io/badge/Metal-3.1%2B-orange.svg)]()

**High-performance Apple Silicon runtime for 1.58-bit ternary language models.**

`mlx-ternary` enables extreme-compression ternary LLMs (such as Prism ML's [Ternary-Bonsai-2-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit)) to run natively at full speed on Apple Silicon with custom Metal Shading Language (MSL) JIT kernels, hardware vector operations, shape-aware hybrid GEMV dispatch, and multi-engine speculative decoding.

---

> ### 💡 Understanding the Silicon Envelope
> **27.2 Billion parameters running locally on an everyday laptop.**  
> In standard FP16, a 27B model demands **54.4 GB VRAM** just to load weights—requiring multi-GPU workstations or high-end Mac Studios.
> 
> With `mlx-ternary` on a base **Apple M4 Pro (24 GB Unified Memory, 273 GB/s peak bandwidth, AC Mains)**:
> - **Total Working Footprint:** **6.37 GB flat** ($0.78\text{ GB}$ / $11\%$ lighter than official Bonsai, leaving $17.6\text{ GB}$ completely free for 32k KV context and OS buffers with zero memory leaks).
> - **Prefill Latency:** **$116.3\text{ tok/s}$** ($1.15\times$ faster than official Bonsai).
> - **Autoregressive Decode:** **$15.2 - 16.5\text{ tok/s}$** ($60.6 - 65.8\text{ ms/tok}$).
> - **Numerical Precision:** **100% Bit-Exact** matching official baseline.

---

## Performance Matrix

Benchmarking **Ternary-Bonsai-2-27B** (27.2B parameters, 64 Transformer layers, 5120 hidden dimension, 17408 MLP dimension) on Apple M4 Pro (24 GB Unified Memory, AC Mains):

### End-to-End Workload Telemetry

| Workload Profile | Engine | Prefill Throughput | Decode Throughput | Active VRAM | Parity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Workload A** (Coding / JSON) | Official Bonsai<br>**mlx-ternary v0.6.0** | $101.36\text{ tok/s}$ ($9.87\text{ ms}$)<br>**$116.34\text{ tok/s}$ ($8.59\text{ ms}$)** | **$16.51\text{ tok/s}$** ($60.57\text{ ms}$)<br>**$16.51\text{ tok/s}$** ($60.57\text{ ms}$) | $7.15\text{ GB}$<br>**$6.37\text{ GB}$** (-0.78 GB) | Reference<br>**100% Bit-Exact** |
| **Workload B** (Agentic Dialog) | Official Bonsai<br>**mlx-ternary v0.6.0** | $99.63\text{ tok/s}$ ($10.04\text{ ms}$)<br>**$111.56\text{ tok/s}$ ($8.96\text{ ms}$)** | **$15.20\text{ tok/s}$** ($65.81\text{ ms}$)<br>**$15.20\text{ tok/s}$** ($65.81\text{ ms}$) | $7.15\text{ GB}$<br>**$6.37\text{ GB}$** (-0.78 GB) | Reference<br>**100% Bit-Exact** |
| **Workload C** (GSM8K Math) | Official Bonsai<br>**mlx-ternary v0.6.0** | $101.99\text{ tok/s}$ ($9.80\text{ ms}$)<br>**$95.18\text{ tok/s}$ ($10.51\text{ ms}$)** | **$15.46\text{ tok/s}$** ($64.67\text{ ms}$)<br>**$15.46\text{ tok/s}$** ($64.67\text{ ms}$) | $7.15\text{ GB}$<br>**$6.37\text{ GB}$** (-0.78 GB) | Reference<br>**100% Bit-Exact** |

### Kernel Micro-Benchmarks (M4 Pro Hardware Testbed)
- **MLP Gate/Up ($5120 \times 17408$):** **$234.3\text{ }\mu\text{s}$** ($95.1\text{ GB/s}$ sustained, 1D SRAM activation caching).
- **MLP Down ($17408 \times 5120$):** **$321.7\text{ }\mu\text{s}$** ($69.3\text{ GB/s}$ sustained, high-$K$ SRAM caching).
- **Attention KV ($5120 \times 512$):** **$111.7\text{ }\mu\text{s}$** ($5.9\text{ GB/s}$, 32-lane cooperative SIMD warp reduction, 0 barriers).
- **Prefill JIT Fast-Path ($M > 1$):** Dispatches directly into Apple AMX block GEMM, delivering $>110\text{ tok/s}$.

---

## Technical Documentation & Build Logs

- 📐 **Architecture Deep Dive:** [`docs/architecture.md`](docs/architecture.md) — SIMD warp reductions, SRAM activation caching, and memory interleaving.
- 🗺️ **Project Roadmap:** [`docs/roadmap.md`](docs/roadmap.md) — Release progression from v0.6.0 through v0.7.0 (Pure-ALU LUT Inversion) and v0.8.0.
- 🔬 **Technical Case Study:** [`faizpalwala.tech/projects/mlx-ternary`](https://faizpalwala.tech/projects/mlx-ternary)
- 📝 **Engineering Build Logs on Noggin:**
  - 🏷️ [All MLX Systems Essays](https://faizpalwala.tech/noggin/tag/mlx-ternary)
  - 📖 [*Escape from ABI Hell*](https://faizpalwala.tech/noggin/escape-from-abi-hell) — Why compiling C++ extensions is a trap and how runtime Metal JIT eliminates toolchain friction.
  - 🚚 [*Out for Delivery*](https://faizpalwala.tech/noggin/out-for-delivery) — Memory coalescing, GPU cache lines as delivery trucks, and 32-lane cooperative warps.
  - 📐 [*Honey, I Shrunk the Weights*](https://faizpalwala.tech/noggin/honey-i-shrunk-the-weights) — The 27B parameter silicon geometry and memory walls on consumer Apple Silicon.

---

## Installation

```bash
pip install mlx-ternary
```

Or install from source:

```bash
git clone https://github.com/FaizPalwala/mlx-ternary.git
cd mlx-ternary
pip install -e .
```

---

## Command-Line Interface (CLI)

`mlx-ternary` registers both `mlx-ternary` and `mlx_ternary` console entrypoints:

### 1. Streaming Text Generation (`generate`)
Run direct streaming inference with the production Tier 3 Fused engine:
```bash
mlx-ternary generate prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --prompt "def quicksort(arr):"
```

To test experimental speculative decoding on structured code or repetitive prompt schemas:
```bash
# Prompt Lookup Decoding (PLD)
mlx-ternary generate prism-ml/Ternary-Bonsai-2-27B-mlx-2bit -p "def fibonacci(n):" --speculative-experimental pld

# 16-Layer Self-Speculative Draft Proposal
mlx-ternary generate prism-ml/Ternary-Bonsai-2-27B-mlx-2bit -p "Explain quantum superposition" --speculative-experimental self
```

### 2. OpenAI-Compatible Server (`serve`)
Serve your local model directly to **LM Studio**, **OpenWebUI**, or any OpenAI-compatible client:
```bash
# Standard server (Port 8080)
mlx-ternary serve prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --port 8080

# Long-Context Mode (4-bit KV Cache)
mlx-ternary serve prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --kv-bits 4 --port 8080
```

### 3. Silicon Performance Probe (`bench`)
Run an empirical hardware audit of your Apple Silicon GPU:
```bash
mlx-ternary bench
```

### 4. Weight Transcoding & Packing (`pack`)
Pre-pack weights offline into 32-byte hardware-vectorized interleaved format:
```bash
mlx-ternary pack prism-ml/Ternary-Bonsai-2-27B-mlx-2bit -o /path/to/packed_model
```

---

## Python API

```python
import mlx.core as mx
from mlx_ternary import load_model, TernaryLinear, ternary_matmul

# 1. Load Model with Automatic Operator Fusion (Tier 3 Fused SOTA)
model, tokenizer = load_model("prism-ml/Ternary-Bonsai-2-27B-mlx-2bit", fuse=True)

# 2. Low-Level Core Matmul Kernel Dispatch
# Dispatches automatically between 32-lane SIMD warp reduction and 1D SRAM activation caching
layer = TernaryLinear(in_features=5120, out_features=17408, group_size=128)
x = mx.random.normal((1, 5120)).astype(mx.float16)
out = layer(x)
```

---

## Testing & Verification

Run the full unit test suite:

```bash
pytest tests/ -v
# 58 passed in 5.34s
```

---

## Support the Project

Benchmarking, training adapter checkpoints, and developing custom Metal kernels on an Apple Silicon laptop takes real time and compute. As a new grad building open-source systems on a student budget, evaluation compute and hardware testbeds add up quickly.

If `mlx-ternary` saved you from purchasing an expensive workstation or paying for continuous cloud GPU instances, consider supporting the work. Any and all contributions go directly toward buying hardware testbeds, evaluation compute, and just paying bills that support open-source development:

* 💖 **GitHub Sponsors:** [github.com/sponsors/faizpalwala](https://github.com/sponsors/faizpalwala)
* ☕ **Buy Me a Coffee:** [buymeacoffee.com/faizpalwala](https://buymeacoffee.com/faizpalwala)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
