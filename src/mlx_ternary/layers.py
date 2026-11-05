import mlx.core as mx
import mlx.nn as nn
from .ops import ternary_matmul, ternary_swiglu, ternary_fused_kv, ternary_fused_qkv

def fwht(x, block, signs=None, inverse=False):
    """
    Fast Walsh-Hadamard Transform implementation for Bonsai.
    """
    if block <= 0:
        return x
    
    # We fallback to standard MLX ops here, or a custom Metal kernel in the future
    # A standard FWHT multiplies activations against a Walsh-Hadamard matrix
    # This is a placeholder for the actual FWHT algorithm for PrismML
    import math
    if signs is not None:
        if inverse:
            x = x * signs
    
    # Simple unoptimized FWHT for demonstration
    # (In production, this needs a custom Metal kernel or iterative butterfly network)
    n = block
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x_j = x[..., j]
                x_jh = x[..., j + h]
                x[..., j] = x_j + x_jh
                x[..., j + h] = x_j - x_jh
        h *= 2
        
    x = x / math.sqrt(n)
    
    if signs is not None and not inverse:
        x = x * signs
        
    return x

from .quantize import pack_interleaved

class TernaryLinear(nn.Module):
    """
    Model-agnostic Neural Network layer for 1.6-bit Ternary Matrix Multiplication.
    Acts as a drop-in replacement for standard `nn.Linear` in MLX models,
    or initialized directly from pre-packed interleaved weights.
    """
    def __init__(
        self,
        input_dims_or_weight,
        output_dims: int = None,
        bias: bool = False,
        group_size: int = 128,
        pad_k: int = 0
    ):
        super().__init__()
        self.group_size = group_size
        self.pad_k = pad_k
        self.bias = None

        if isinstance(input_dims_or_weight, mx.array):
            self.weight_packed = input_dims_or_weight
            self.input_dims = None
            self.output_dims = self.weight_packed.shape[1]
        else:
            self.input_dims = input_dims_or_weight
            self.output_dims = output_dims
            self.pad_k = (group_size - (self.input_dims % group_size)) % group_size
            padded_k = self.input_dims + self.pad_k
            blocks = padded_k // group_size
            self.weight_packed = mx.zeros((blocks, output_dims, 32), dtype=mx.uint8)
            if bias:
                self.bias = mx.zeros((output_dims,), dtype=mx.float16)

    def quantize_from_raw(self, w_raw: mx.array):
        """
        Quantize FP16 weights into 32-byte cache-line interleaved ternary blocks.
        Accepts standard PyTorch/MLX weight shape (output_dims, input_dims) or transposed (input_dims, output_dims).
        """
        if self.output_dims is not None and self.input_dims is not None:
            if w_raw.shape == (self.output_dims, self.input_dims):
                w_raw = w_raw.T
        self.weight_packed, self.pad_k = pack_interleaved(w_raw, group_size=self.group_size)

    def __call__(self, x: mx.array) -> mx.array:
        if self.pad_k > 0:
            *batch, k = x.shape
            x = mx.concatenate([x, mx.zeros((*batch, self.pad_k), dtype=x.dtype)], axis=-1)
        out = ternary_matmul(x, self.weight_packed, group_size=self.group_size)
        if self.bias is not None:
            out = out + self.bias
        return out

class BonsaiTernaryLinear(TernaryLinear):
    """
    PrismML Bonsai style Linear layer with Fast Walsh-Hadamard Transform (FWHT) preprocessing.
    """
    def __init__(
        self,
        weight_packed,
        output_dims: int = None,
        bias: bool = False,
        group_size: int = 128,
        pad_k: int = 0,
        block: int = 0,
        signs: mx.array = None
    ):
        super().__init__(weight_packed, output_dims=output_dims, bias=bias, group_size=group_size, pad_k=pad_k)
        self.block = block
        self.signs = signs

    def __call__(self, x: mx.array) -> mx.array:
        if self.block:
            x = fwht(x, self.block, self.signs)
        return super().__call__(x)


class TernarySwiGLU(nn.Module):
    """
    Fused Ternary SwiGLU projection layer: SiLU(x @ w_gate) * (x @ w_up).

    Executes single-pass memory traversal across interleaved ternary weights
    with in-register activation evaluation.
    """
    def __init__(
        self,
        input_dims_or_gate,
        output_dims_or_up=None,
        output_dims: int = None,
        group_size: int = 128,
        pad_k: int = 0
    ):
        super().__init__()
        self.group_size = group_size
        self.pad_k = pad_k

        if isinstance(input_dims_or_gate, mx.array) and isinstance(output_dims_or_up, mx.array):
            self.gate_packed = input_dims_or_gate
            self.up_packed = output_dims_or_up
            self.input_dims = None
            self.output_dims = self.gate_packed.shape[1]
        else:
            self.input_dims = input_dims_or_gate
            self.output_dims = output_dims if output_dims is not None else output_dims_or_up
            self.pad_k = (group_size - (self.input_dims % group_size)) % group_size
            padded_k = self.input_dims + self.pad_k
            blocks = padded_k // group_size
            self.gate_packed = mx.zeros((blocks, self.output_dims, 32), dtype=mx.uint8)
            self.up_packed = mx.zeros((blocks, self.output_dims, 32), dtype=mx.uint8)

    def quantize_from_raw(self, w_gate_raw: mx.array, w_up_raw: mx.array):
        """
        Quantize FP16 Gate and Up weights into 32-byte cache-line interleaved ternary blocks.
        Accepts standard PyTorch/MLX weight shape (output_dims, input_dims) or transposed (input_dims, output_dims).
        """
        if self.output_dims is not None and self.input_dims is not None:
            if w_gate_raw.shape == (self.output_dims, self.input_dims):
                w_gate_raw = w_gate_raw.T
            if w_up_raw.shape == (self.output_dims, self.input_dims):
                w_up_raw = w_up_raw.T
        self.gate_packed, self.pad_k = pack_interleaved(w_gate_raw, group_size=self.group_size)
        self.up_packed, _ = pack_interleaved(w_up_raw, group_size=self.group_size)

    def __call__(self, x: mx.array) -> mx.array:
        if self.pad_k > 0:
            *batch, k = x.shape
            x = mx.concatenate([x, mx.zeros((*batch, self.pad_k), dtype=x.dtype)], axis=-1)
        return ternary_swiglu(x, self.gate_packed, self.up_packed, group_size=self.group_size)


class TernaryFusedKV(nn.Module):
    """
    Fused Key and Value projection layer: x @ w_k and x @ w_v.

    Reuses loaded activation vector chunks in registers across dual-issue ALU pipelines,
    evaluating both projections concurrently in a single kernel dispatch.
    """
    def __init__(
        self,
        input_dims_or_k,
        kv_dims_or_v=None,
        kv_dims: int = None,
        group_size: int = 128,
        pad_k: int = 0
    ):
        super().__init__()
        self.group_size = group_size
        self.pad_k = pad_k

        if isinstance(input_dims_or_k, mx.array) and isinstance(kv_dims_or_v, mx.array):
            self.k_packed = input_dims_or_k
            self.v_packed = kv_dims_or_v
            self.input_dims = None
            self.kv_dims = self.k_packed.shape[1]
        else:
            self.input_dims = input_dims_or_k
            self.kv_dims = kv_dims if kv_dims is not None else kv_dims_or_v
            self.pad_k = (group_size - (self.input_dims % group_size)) % group_size
            padded_k = self.input_dims + self.pad_k
            blocks = padded_k // group_size
            self.k_packed = mx.zeros((blocks, self.kv_dims, 32), dtype=mx.uint8)
            self.v_packed = mx.zeros((blocks, self.kv_dims, 32), dtype=mx.uint8)

    @classmethod
    def from_raw(cls, w_k: mx.array, w_v: mx.array, group_size: int = 128) -> "TernaryFusedKV":
        """
        Construct TernaryFusedKV directly from unquantized FP16 weights.
        Accepts weights shaped as (K, N) or PyTorch standard (N, K).
        """
        if w_k.ndim == 2 and w_v.ndim == 2:
            if w_k.shape[0] == w_v.shape[0]:
                pass  # Contraction axis is already axis 0
            elif w_k.shape[1] == w_v.shape[1]:
                w_k = w_k.T
                w_v = w_v.T
            else:
                raise ValueError(
                    f"Incompatible weight shapes for KV fusion: w_k={w_k.shape}, w_v={w_v.shape}"
                )
        k_packed, pad_k = pack_interleaved(w_k, group_size=group_size)
        v_packed, _ = pack_interleaved(w_v, group_size=group_size)
        layer = cls(k_packed, v_packed, group_size=group_size, pad_k=pad_k)
        layer.input_dims = w_k.shape[0]
        layer.kv_dims = w_k.shape[1]
        return layer

    def quantize_from_raw(self, w_k_raw: mx.array, w_v_raw: mx.array):
        """
        Quantize FP16 K and V weights into 32-byte cache-line interleaved ternary blocks.
        """
        if self.kv_dims is not None and self.input_dims is not None:
            if w_k_raw.shape == (self.kv_dims, self.input_dims):
                w_k_raw = w_k_raw.T
            if w_v_raw.shape == (self.kv_dims, self.input_dims):
                w_v_raw = w_v_raw.T
        self.k_packed, self.pad_k = pack_interleaved(w_k_raw, group_size=self.group_size)
        self.v_packed, _ = pack_interleaved(w_v_raw, group_size=self.group_size)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        if self.pad_k > 0:
            *batch, k = x.shape
            x = mx.concatenate([x, mx.zeros((*batch, self.pad_k), dtype=x.dtype)], axis=-1)
        return ternary_fused_kv(x, self.k_packed, self.v_packed, group_size=self.group_size)


class TernaryQKV(nn.Module):
    """
    Unified Query, Key, and Value projection layer.

    Evaluates concatenated [Q, K, V] projections in a single 0-barrier wide GEMV dispatch
    and slices the output into query, key, and value activation tensors.
    """
    def __init__(
        self,
        input_dims_or_qkv,
        q_dim: int = None,
        k_dim: int = None,
        v_dim: int = None,
        split: bool = True,
        group_size: int = 128,
        pad_k: int = 0
    ):
        super().__init__()
        self.group_size = group_size
        self.pad_k = pad_k
        self.split = split
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim

        if isinstance(input_dims_or_qkv, mx.array):
            self.qkv_packed = input_dims_or_qkv
            self.input_dims = None
            total_n = self.qkv_packed.shape[1]
            self.output_dims = total_n
        else:
            self.input_dims = input_dims_or_qkv
            assert q_dim is not None and k_dim is not None and v_dim is not None
            self.output_dims = q_dim + k_dim + v_dim
            self.pad_k = (group_size - (self.input_dims % group_size)) % group_size
            padded_k = self.input_dims + self.pad_k
            blocks = padded_k // group_size
            self.qkv_packed = mx.zeros((blocks, self.output_dims, 32), dtype=mx.uint8)

    @classmethod
    def from_raw(
        cls,
        w_q: mx.array,
        w_k: mx.array,
        w_v: mx.array,
        split: bool = True,
        group_size: int = 128
    ) -> "TernaryQKV":
        """
        Construct TernaryQKV directly from unquantized FP16 weights.
        Accepts weights shaped as (K, N) or PyTorch standard (N, K).
        """
        if w_q.ndim == 2 and w_k.ndim == 2 and w_v.ndim == 2:
            if w_q.shape[0] == w_k.shape[0] == w_v.shape[0]:
                pass  # Contraction axis is already axis 0
            elif w_q.shape[1] == w_k.shape[1] == w_v.shape[1]:
                w_q = w_q.T
                w_k = w_k.T
                w_v = w_v.T
            else:
                raise ValueError(
                    f"Incompatible weight shapes for QKV fusion: w_q={w_q.shape}, w_k={w_k.shape}, w_v={w_v.shape}"
                )

        q_dim = w_q.shape[1]
        k_dim = w_k.shape[1]
        v_dim = w_v.shape[1]

        w_qkv = mx.concatenate([w_q, w_k, w_v], axis=1)
        qkv_packed, pad_k = pack_interleaved(w_qkv, group_size=group_size)
        layer = cls(
            qkv_packed,
            q_dim=q_dim,
            k_dim=k_dim,
            v_dim=v_dim,
            split=split,
            group_size=group_size,
            pad_k=pad_k
        )
        layer.input_dims = w_q.shape[0]
        return layer

    def quantize_from_raw(self, w_q_raw: mx.array, w_k_raw: mx.array, w_v_raw: mx.array):
        """
        Quantize FP16 Q, K, and V weights into 32-byte cache-line interleaved ternary blocks.
        """
        if self.q_dim is not None and self.input_dims is not None:
            if w_q_raw.shape == (self.q_dim, self.input_dims):
                w_q_raw = w_q_raw.T
            if w_k_raw.shape == (self.k_dim, self.input_dims):
                w_k_raw = w_k_raw.T
            if w_v_raw.shape == (self.v_dim, self.input_dims):
                w_v_raw = w_v_raw.T
        w_qkv = mx.concatenate([w_q_raw, w_k_raw, w_v_raw], axis=1)
        self.qkv_packed, self.pad_k = pack_interleaved(w_qkv, group_size=self.group_size)

    def __call__(self, x: mx.array):
        if self.pad_k > 0:
            *batch, k = x.shape
            x = mx.concatenate([x, mx.zeros((*batch, self.pad_k), dtype=x.dtype)], axis=-1)
        out = ternary_fused_qkv(x, self.qkv_packed, group_size=self.group_size)
        if self.split and self.q_dim is not None and self.k_dim is not None:
            q = out[..., :self.q_dim]
            k = out[..., self.q_dim:self.q_dim + self.k_dim]
            v = out[..., self.q_dim + self.k_dim:]
            return q, k, v
        return out


