# Python API Reference

`mlx_ternary` provides high-level neural network modules and low-level Metal operations for 1.58-bit ternary models.

---

## 1. High-Level Neural Network Layers

### `mlx_ternary.TernaryLinear`

A drop-in replacement for `mlx.nn.Linear` that stores weights in 32-byte cacheline interleaved format and evaluates via JIT Metal kernels.

```python
from mlx_ternary import TernaryLinear
import mlx.core as mx

# Initialize a ternary linear layer
layer = TernaryLinear(in_features=5120, out_features=17408, group_size=128)

# Input activation
x = mx.random.normal((1, 5120)).astype(mx.float16)

# Forward pass
output = layer(x)
print(output.shape) # (1, 17408)
```

---

## 2. Low-Level Core Operations

### `mlx_ternary.ternary_matmul(x, w_packed, group_size=None)`

Executes a high-performance ternary matrix multiplication with automatic dual-dispatch:
- If $M \le 16$, executes the coalesced vector GEMV kernel with Split-K reduction.
- If $M > 16$, executes the AMX coprocessor GEMM kernel.

```python
import mlx_ternary
import mlx.core as mx

# x: (M, K) float16
# w_packed: (K_blocks, N, 32) uint8
out = mlx_ternary.ternary_matmul(x, w_packed, group_size=128)
```

---

## 3. Quantization & Weight Packing

### `mlx_ternary.pack_interleaved(w_raw, group_size=128)`

Packs an unquantized FP16 weight matrix $(K, N)$ into the Horizon 32-byte interleaved cacheline format.

```python
from mlx_ternary import pack_interleaved
import mlx.core as mx

w_fp16 = mx.random.normal((5120, 17408)).astype(mx.float16)
w_packed, pad_k = pack_interleaved(w_fp16, group_size=128)

# w_packed shape: (K // group_size, N, 32) uint8
```
