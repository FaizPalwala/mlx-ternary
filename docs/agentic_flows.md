# Agentic Flows & KV Cache Scaling

Agentic workflows (multi-turn tool calling, terminal execution loops, code generation, and iterative self-correction) generate tens of thousands of tokens over the course of a single session.

While `mlx-ternary` compresses a 27B parameter model down to $\approx 8.2\text{ GB}$ of VRAM, **the unquantized KV cache is the primary limiting factor for long-horizon agent loops on Apple Silicon.**

---

## 1. The KV Cache Memory Budget on 24 GB Unified Memory

On an Apple Silicon machine with 24 GB Unified Memory (e.g. MacBook Pro M4 Pro):

- **macOS System & Display Buffer:** $\approx 3.5\text{ GB}$
- **Ternary-Bonsai-2-27B Weights:** $\approx 8.2\text{ GB}$
- **Remaining Headroom for KV Cache:** $\approx \mathbf{12.3\text{ GB}}$

---

## 2. Memory Consumption by Precision

A 27B model with 64 layers, 16 Key/Value heads, and head dimension 128 requires:

$$\text{Bytes per token} = 64 \times 16 \times 128 \times 2 \times \text{bytes\_per\_elem}$$

| Context Length | Standard FP16 (524 KB/tok) | 8-bit Quantized (262 KB/tok) | 4-bit Quantized (131 KB/tok) |
| :--- | :--- | :--- | :--- |
| **4,000 tokens** | $2.1\text{ GB}$ | $1.0\text{ GB}$ | **$0.5\text{ GB}$** |
| **8,000 tokens** | $4.2\text{ GB}$ | $2.1\text{ GB}$ | **$1.0\text{ GB}$** |
| **16,000 tokens** | $8.4\text{ GB}$ | $4.2\text{ GB}$ | **$2.1\text{ GB}$** |
| **32,000 tokens** | **$16.8\text{ GB}$ (OOM / SSD swap)** | $8.4\text{ GB}$ | **$4.2\text{ GB}$ (Comfortable!)** |
| **64,000 tokens** | ❌ OOM | **$16.8\text{ GB}$ (OOM)** | **$8.4\text{ GB}$ (Fits in 24 GB!)** |

---

## 3. Best Practices for Local Agents

1. **Enable 4-bit KV Cache for Long Sessions:**
   ```bash
   mlx_ternary serve model_path --kv-bits 4
   ```
2. **Context Compaction:** In agent loops where subagents output massive terminal logs or compiler errors, truncate repetitive middle lines before injecting into the prompt context.
3. **Structured Tool Calling:** Because low-bit models have higher entropy on rigid syntax tokens, use schema-constrained sampling to avoid malformed JSON parsing failures.
