# Project Architecture: `mlx-ternary`

## Mission
Build a native, high-performance Apple Silicon runtime (C++/Metal kernels + Python bindings) specifically engineered for 1.58-bit ternary language models (like PrismML's Bonsai series). 

## Core Problem
Ternary models use a **Hadamard-rotated weight basis** to spread outlier weights uniformly before quantization. Standard `mlx-lm` or Ollama loaders misinterpret these weights, outputting garbage. We are building the native infrastructure to parse these weights and execute them natively on Metal.

## Core Stack
- **Languages:** C++, Metal (Shaders), Python.
- **Frameworks:** Apple MLX, mlx-lm API compatibility.

## System Boundaries
- **What it IS:** A lightweight loader and custom inference kernel for Hadamard-rotated ternary matrices. It relies on a decoupled, pure C++ core logic layer that compiles custom Metal shaders.
- **What it IS NOT:** A training library. A general-purpose Post-Training Quantization (PTQ) pipeline. 

## Architectural Layers
1. **C++ & Metal Core:** The foundational layer containing custom MLX ops (`mlx::core::Op`) and addition-only `.metal` kernels.
2. **C-API:** Exposes the C++ core via Foreign Function Interface (FFI).
3. **Language Bindings:**
   - **Python:** For ML research, model conversion, and drop-in `mlx-lm` compatibility.
   - **Swift:** For native iOS/macOS production deployment.
   - **Rust/Go (Planned):** For backend agents and local LLM wrappers.

## API Contract (Python Example)
```python
import mlx_ternary
model, tokenizer = mlx_ternary.load("PrismML/Bonsai-2-27B")
response = mlx_ternary.generate(model, tokenizer, prompt="...")
```
