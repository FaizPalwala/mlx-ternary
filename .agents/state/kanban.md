# Kanban Board

## 📋 Backlog
- [ ] Finalize testing strategy and CI/CD

## 🛠️ In Progress
- [ ] Profile performance and write benchmark scripts
- [ ] Investigate BitNet b1.58 kernel optimizations (fusing scaling, avoiding branches)

## ✅ Done
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
