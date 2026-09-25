import mlx.core as mx
import mlx.nn as nn
from .ops import ternary_matmul
from .quantize import pack_interleaved

class TernaryLinear(nn.Module):
    """
    A model-agnostic Neural Network layer for 1.6-bit Ternary Matrix Multiplication.
    This acts as a drop-in replacement for standard `nn.Linear` in MLX models.
    """
    def __init__(self, input_dims: int, output_dims: int, bias: bool = False):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        
        # Hardware alignment requires K to be divisible by the cache-line group size (150)
        self.pad_k = (150 - (input_dims % 150)) % 150
        padded_input_dims = input_dims + self.pad_k
        blocks = padded_input_dims // 150
        
        # Allocate the 32-byte interlaced blocks: shape (K_blocks, N, 32)
        # Note: We do not initialize with random weights because ternary weights 
        # must be explicitly loaded via state_dict or quantized via `.quantize_from_fp16()`
        self.w_interleaved = mx.zeros((blocks, output_dims, 32), dtype=mx.uint8)
        
        if bias:
            self.bias = mx.zeros((output_dims,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.
        x: shape (..., input_dims)
        """
        out = ternary_matmul(x, self.w_interleaved)
        if self.bias is not None:
            out = out + self.bias
        return out

    def quantize_from_raw(self, w_raw: mx.array):
        """
        Utility to inject unquantized weights into this layer.
        Expects standard MLX/PyTorch weight shape: (output_dims, input_dims).
        """
        if w_raw.shape == (self.output_dims, self.input_dims):
            w_raw = w_raw.T
            
        assert w_raw.shape == (self.input_dims, self.output_dims), \
            f"Expected shape {(self.input_dims, self.output_dims)}, got {w_raw.shape}"
            
        self.w_interleaved, self.pad_k = pack_interleaved(w_raw, group_size=150)
