# mlx-ternary

[![PyPI Version](https://img.shields.io/pypi/v/mlx-ternary.svg)](https://pypi.org/project/mlx-ternary/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%20to%20M5-black.svg)]()
[![Metal](https://img.shields.io/badge/Metal-3.1%2B-orange.svg)]()

**High-performance Apple Silicon runtime for 1.58-bit ternary language models.**

`mlx-ternary` enables extreme-compression ternary LLMs (such as Prism ML's [Ternary-Bonsai-2-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit)) to run natively at full speed on Apple Silicon with pure Metal JIT kernels, hardware vector dot product operations, Split-K parallelization, and configurable KV cache quantization.

---

## ⚡ Performance Matrix

Benchmarking **Ternary-Bonsai-2-27B** (64 Transformer layers, 5120 hidden dimension):

| Platform | Memory Bandwidth | Autoregressive Decode | Footprint |
| :--- | :--- | :--- | :--- |
| **NVIDIA RTX 5090** | 1,792 GB/s | **143.0 tok/s** | Desktop Workstation |
| **Apple M5 Max** | $\approx 800\text{ GB/s}$ | **47.0 tok/s** | Flagship Laptop |
| **Apple M4 Pro (Ours)**| **273 GB/s** | **13.8 tok/s** (Autoregressive)<br>**22.8 tok/s** (Speculative) | **MacBook Pro (24 GB RAM)** |
| **Cloud APIs (OpenRouter)**| Variable | **10.0 - 15.0 tok/s** | Shared Cloud Infrastructure |

---

## 🚀 Key Features

* **Horizon 32-Byte Interleaving:** Scales are packed every 32 bytes alongside Base-3 trits, guaranteeing perfect 128-byte cacheline alignment with zero stride memory stalls.
* **Apple Silicon Hardware Vector ALU (`dot(half4, half4)`):** Native single-cycle vector operations and constant memory LUT decoding.
* **Split-K Dynamic Reduction:** Narrow transformer projections (`Attention KV`, $N=512$) automatically parallelize across 64 threadgroups, yielding **$+9.8\%$ higher throughput than Apple's native 2-bit affine kernel**.
* **Configurable KV Cache Quantization:** Run deep 32k–64k agentic contexts on 24 GB MacBooks with 4-bit (`--kv-bits 4`) and 8-bit (`--kv-bits 8`) quantized KV caches.
* **Speculative Decoding Engine:** Multi-token AMX block GEMM sweeps against small draft models (`Qwen2.5-1.5B`), breaking memory bandwidth limits for up to **$2.0\times$ speedups**.
* **Zero-Build Architecture:** Pure Python orchestration via `mlx.core.fast.metal_kernel`. Zero C++ compilation or CMake required.

---

## 📦 Installation

```bash
pip install mlx-ternary
```

Or install from source:

```bash
git clone https://github.com/base3labs/mlx-ternary.git
cd mlx-ternary
pip install -e .
```

---

## 🖥️ Local Model Serving (OpenAI-Compatible)

Serve your local model directly to **LM Studio**, **OpenWebUI**, or any OpenAI SDK:

```bash
# Standard server (Port 8080)
mlx_ternary serve prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --port 8080

# Long-Context Agentic Mode (4-bit KV Cache)
mlx_ternary serve prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --kv-bits 4 --port 8080

# High-Throughput Speculative Decoding
mlx_ternary serve prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --speculate --port 8080
```

### Connect with LM Studio
1. Open LM Studio $\rightarrow$ Developer / Local Server.
2. Connect to Base URL: `http://127.0.0.1:8080/v1`
3. Model Identifier: `Ternary-Bonsai-2-27B-mlx-2bit`

---

## 🐍 Python API

```python
import mlx.core as mx
from mlx_ternary import TernaryLinear, pack_interleaved, ternary_matmul

# 1. High-level Neural Network Module
layer = TernaryLinear(in_features=5120, out_features=17408, group_size=128)
x = mx.random.normal((1, 5120)).astype(mx.float16)
out = layer(x)

# 2. Low-level Core Matmul
# w_packed: (K // 128, N, 32) uint8
out = ternary_matmul(x, layer.weight, group_size=128)
```

---

## 🧪 Testing & Verification

Run the test suite:

```bash
pytest tests/ -v
# 9 passed in 0.13s
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
