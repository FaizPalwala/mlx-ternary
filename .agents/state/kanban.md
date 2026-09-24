# Kanban Board

## 📋 Backlog
- [ ] Implement end-to-end small model (e.g. Llama/Mistral) using TernaryLinear.
- [ ] Finalize testing strategy and CI/CD

## 🛠️ In Progress
- [ ] Investigate BitNet b1.58 kernel optimizations (Tiling, SIMD groups, avoiding AMX bottlenecks).

## ✅ Done
- [x] Profile performance and write benchmark scripts (Achieved baseline metrics).
- [x] Write pytest suite for verifying outputs against standard matmul (Equivalence matched!)
- [x] Resolve Nanobind cross-module RTTI bridging issue (Pinned v2.10.2).
- [x] Enforce memory contiguity for Metal kernels.
- [x] Implement actual Metal kernel logic for ternary matmul in ternary.metal and dispatch it in TernaryOp.cpp
- [x] Verify Nanobind Python/C++ boundary behavior at runtime (Successfully bridged!)
- [x] Successfully compiled C++ extension within MLX framework constraints (Primitive API)
- [x] Scaffold project directory and CMakeLists.txt for the C++ extension.
- [x] Implement: `mlx_ternary.load()` to parse ternary safetensors and separate scales.
- [x] Initialized repository structure and build system.
- [x] Initialized Agentic Persona (Mac).
