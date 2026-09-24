# Kanban Board

## 📋 Backlog
- [ ] Implement end-to-end small model (e.g. Llama/Mistral) using TernaryLinear.
- [ ] Finalize testing strategy and CI/CD

## 🛠️ In Progress
- [ ] Profile memory bandwidth utilization and write Threadgroup Block-Tiling to reach >200 GB/s on the M4 Pro.
- [ ] Expose `quantize()` API in Python to convert HuggingFace models into our custom format.

## ✅ Done
- [x] **Milestone 2**: Implemented 1.6-bit Base-3 packing in Python.
- [x] **Milestone 2**: Added Constant-Memory Decompression LUT to Metal kernel to eliminate ALU modulo math.
- [x] **Milestone 2**: Resolved Metal UB (Undefined Behavior) during `uint4` memory aliasing.
- [x] **Milestone 1**: Rewrote `ternary.metal` kernel to stream weights using `uint4` (128-bit) vectorized loads.
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
