# Agentic Flows & KV Cache Scaling

Agentic workflows (multi-turn tool calling, terminal execution loops, code generation, and iterative self-correction) generate tens of thousands of tokens over the course of a single session.

While `mlx-ternary` compresses a 27B parameter model down to **6.37 GB flat** VRAM, **the unquantized KV cache is the primary limiting factor for long-horizon agent loops on Apple Silicon.**

---

## 1. The KV Cache Memory Budget on 24 GB Unified Memory

On an Apple Silicon machine with 24 GB Unified Memory (e.g. MacBook Pro M4 Pro):

- **macOS System & Display Buffer:** ~3.5 GB
- **Ternary-Bonsai-2-27B Weights:** 6.37 GB flat
- **Remaining Headroom for KV Cache & Context:** ~14.1 GB

---

## 2. Memory Consumption by Precision

A 27B model with 64 layers, 16 Key/Value heads, and head dimension 128 requires:

```text
Bytes per token = Layers × KV_Heads × Head_Dim × 2 (K + V) × Bytes_per_Element
```

$$
\text{Elements per token} = 64 \times 16 \times 128 \times 2 = 262{,}144\text{ elements/token}
$$

- **Standard FP16** (2 bytes/element): $262{,}144 \times 2 = 524{,}288\text{ bytes} \approx 524\text{ KB/token}$
- **8-bit Quantized** (1 byte/element): $262{,}144 \times 1 = 262{,}144\text{ bytes} \approx 262\text{ KB/token}$
- **4-bit Quantized** (0.5 bytes/element): $262{,}144 \times 0.5 = 131{,}072\text{ bytes} \approx 131\text{ KB/token}$

| Context Length | Standard FP16 (524 KB/tok) | 8-bit Quantized (262 KB/tok) | 4-bit Quantized (131 KB/tok) |
| :--- | :--- | :--- | :--- |
| **4,000 tokens** | 2.1 GB | 1.0 GB | **0.5 GB** |
| **8,000 tokens** | 4.2 GB | 2.1 GB | **1.0 GB** |
| **16,000 tokens** | 8.4 GB | 4.2 GB | **2.1 GB** |
| **32,000 tokens** | **16.8 GB (OOM / swap)** | 8.4 GB | **4.2 GB (Comfortable)** |
| **64,000 tokens** | ❌ OOM | **16.8 GB (OOM / swap)** | **8.4 GB (Fits in 24 GB)** |

---

## 3. Best Practices for Local Agents

1. **Enable 4-bit KV Cache for Long Sessions:**
   ```bash
   mlx_ternary serve model_path --kv-bits 4
   ```
2. **Context Compaction:** In agent loops where subagents output massive terminal logs or compiler errors, truncate repetitive middle lines before injecting into the prompt context.
3. **Structured Tool Calling:** Because low-bit models have higher entropy on rigid syntax tokens, use schema-constrained sampling to avoid malformed JSON parsing failures.
