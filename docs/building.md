# Installation & Building from Source

`mlx-ternary` is a **pure Python package with runtime JIT Metal shader compilation**. 

Unlike older C++ extension architectures, **no C++ compilers, CMake, or nanobind toolchains are required**.

---

## 1. Prerequisites

- **macOS:** macOS Sonoma (14.0+) or Sequoia (15.0+)
- **Hardware:** Apple Silicon (M1, M2, M3, M4, M5 or later)
- **Python:** Python 3.9 through 3.12

---

## 2. Standard Installation

```bash
pip install mlx-ternary
```

---

## 3. Editable Installation (from Source)

```bash
git clone https://github.com/base3labs/mlx-ternary.git
cd mlx-ternary
pip install -e .
```

---

## 4. Running Tests

To verify that all Metal kernels compile and execute properly on your Apple Silicon GPU:

```bash
pip install pytest
pytest tests/ -v
```

All unit tests should pass in under 1 second.
