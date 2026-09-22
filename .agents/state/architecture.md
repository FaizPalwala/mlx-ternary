# Project Architecture: `mlx-ternary`

## Mission
Build a native, high-performance Apple Silicon runtime (C++/Metal kernels + Python bindings) specifically engineered for 1.58-bit ternary language models (like PrismML's Bonsai series). 

## Core Problem
Ternary models use a **Hadamard-rotated weight basis** to spread outlier weights uniformly before quantization. Standard `mlx-lm` or Ollama loaders misinterpret these weights, outputting garbage. We are building the native infrastructure to parse these weights and execute them natively on Metal.

## Core Stack
- **Languages:** C++, Metal (Shaders), Python.
- **Frameworks:** Apple MLX, mlx-lm API compatibility.

## System Boundaries
- **What it IS:** A lightweight loader and custom inference kernel for Hadamard-rotated ternary matrices. It executes pure-addition matrix operations to bypass FP16 multipliers entirely.
- **What it IS NOT:** A training library. A general-purpose Post-Training Quantization (PTQ) pipeline. 

## API Contract
The Python interface must be a plug-and-play drop-in replacement for `mlx-lm`:
```python
import mlx_ternary
model, tokenizer = mlx_ternary.load("PrismML/Bonsai-2-27B")
response = mlx_ternary.generate(model, tokenizer, prompt="...")
```
