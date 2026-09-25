import mlx.core as mx
from . import mlx_ternary_ext

def ternary_matmul(x: mx.array, w_interleaved: mx.array, group_size: int = 150) -> mx.array:
    """
    Computes a dot product of activations `x` with Ternary weights `w_interleaved`.
    
    w_interleaved should have shape (K / group_size, N, 32).
    """
    return mlx_ternary_ext.ternary_matmul(x, w_interleaved, group_size)
