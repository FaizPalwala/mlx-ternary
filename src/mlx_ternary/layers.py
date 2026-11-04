import mlx.core as mx
import mlx.nn as nn
from .ops import ternary_matmul, ternary_swiglu

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


