# Contributing to mlx-ternary

Thank you for contributing to `mlx-ternary`! We welcome contributions that improve kernel performance, expand model support, or harden hardware compatibility across Apple Silicon.

---

## 🏛️ Architectural Guardrails

Before submitting a pull request, ensure your code complies with our three core architectural principles:

### 1. Pure-Python Zero-Build Architecture
- **No C++ Binaries or CMake:** We do NOT compile C++ extensions or use nanobind/pybind11. All Metal Shading Language (MSL) code is compiled at runtime via `mlx.core.fast.metal_kernel`.
- **Pure Wheel Distribution:** The PyPI wheel must remain 100% platform-agnostic pure Python (`py3-none-any.whl`). Do not introduce platform-specific native binary dependencies.

### 2. Mathematical Rigor & Invariance
- **Parity Threshold:** Any custom kernel or optimization must be verified against an FP32/FP16 unpacked reference matrix multiplication.
- Relative error must satisfy:
  $$\frac{|Y_{\text{kernel}} - Y_{\text{ref}}|}{|Y_{\text{ref}}| + 10^{-3}} < 0.03$$
- Always test on both $M=1$ (autoregressive decode token) and $M > 1$ (prefill/prompt processing).

### 3. Memory & Cacheline Alignment
- Packed weight buffers must respect 32-byte group alignment (scales + packed trits) and 128-byte cacheline boundaries.
- Threadgroup configurations must explicitly support all M-series GPU generations (M1 through M5).

---

## 🧪 Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/base3labs/mlx-ternary.git
   cd mlx-ternary
   ```

2. **Create a virtual environment & install in editable mode:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Run the test suite:**
   ```bash
   pytest tests/ -v
   ```

4. **Run kernel micro-benchmarks:**
   ```bash
   python benchmarks/compare_bonsai.py
   ```

---

## 📝 Submitting a PR
- Ensure all 11 existing unit tests pass cleanly.
- If introducing a new kernel or projection, add corresponding unit tests in `tests/test_ops.py`.
- Include hardware benchmark numbers (SoC model, token decode latency, speedup ratio) in your PR description.
