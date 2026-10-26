# MLX-TERNARY PUBLIC REPOSITORY DIRECTIVE

You are working in **`mlx-ternary`**, the official open-source Apple Silicon runtime for 1.58-bit ternary language models.

## Core Architectural Guardrails:
1. **Zero-Build Runtime:**
   - All custom kernels MUST be implemented in pure Metal Shading Language (MSL) and compiled dynamically at runtime using `mlx.core.fast.metal_kernel`.
   - NEVER introduce C++, CMake, nanobind, or native binary compilation steps. The package must remain a pure-Python wheel (`py3-none-any`).
2. **Mathematical Precision:**
   - Any modifications to kernel dispatch or weight packing must be validated against reference unpacked matmul in `tests/test_ops.py`.
   - Relative error threshold must remain strictly under 0.03 for FP16 accumulation.
3. **Public Repository Isolation:**
   - NEVER commit API keys, `.env` files, local database files (`*.db`), or personal PII to this repository.
   - Do not include internal startup/business strategy or private roadmaps. Keep issues, PRs, and comments strictly technical.
4. **Testing Protocol:**
   - Run `pytest tests/ -v` before every commit. All tests must pass.
