import mlx.core as mx
import mlx_ternary_ext

def ternary_matmul(x: mx.array, w_interleaved: mx.array) -> mx.array:
    """
    Ternary Matrix Multiplication.
    x: (..., M, K)
    w_interleaved: (K/150, N, 32) uint8 interleaved block format
    Returns: (..., M, N)
    """
    return mlx_ternary_ext.ternary_matmul(x, w_interleaved)
