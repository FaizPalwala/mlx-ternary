# mlx-ternary

Native, high-performance Apple Silicon runtime (C++/Metal kernels + Python bindings) specifically engineered for 1.58-bit ternary language models.

## Speculative Decoding (DFlash / Multi-Token Verification)

Ternary inference is naturally bound by memory bandwidth. Generating one token at a time leaves the GPU ALUs mostly idle while waiting for weights to arrive from RAM. 

To maximize the M4 Pro's AMX coprocessor throughput, `mlx_ternary` natively supports **Speculative Decoding**.

When enabled, a tiny 1.5B parameter "draft model" quickly guesses the next $N$ tokens. Your massive 27B Ternary model then sweeps over its weights exactly once, verifying all $N$ tokens in parallel. This trades cheap ALU cycles to save precious memory bandwidth.

**Performance Impact:** Mathematically lossless output, typically yielding 1.5x to 2.5x speedups in token generation.

**VRAM Warning:** This requires loading *both* the 27B model and the 1.5B draft model into Unified Memory simultaneously. If your machine is near its VRAM limit, do not use this feature, as swapping to SSD will destroy performance.

### Usage

You can enable it with the `--speculate` flag. By default, it uses `Qwen/Qwen2.5-1.5B-Instruct` as the draft model:

```bash
mlx_ternary serve prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --speculate
```

To specify a custom draft model:

```bash
mlx_ternary serve prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --draft Qwen/Qwen2.5-0.5B-Instruct
```
