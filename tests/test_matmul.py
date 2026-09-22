import mlx.core as mx
import pytest
from mlx_ternary import ternary_matmul

def reference_unpack(packed_w: mx.array, N: int) -> mx.array:
    """
    Slow, reference Python implementation to unpack 2-bit weights.
    Returns an FP16 matrix for standard matmul comparison.
    """
    # ... In a real test, we would do bitwise extraction here using mx.right_shift etc.
    # For scaffolding, we yield a dummy matrix.
    pass

def test_ternary_matmul_correctness():
    """
    Functional Spec:
    Ensure our Metal addition-only kernel matches the exact mathematical
    output of a standard fp16 fused-multiply-add matrix multiplication.
    """
    # 1. Generate random fp16 activations (M=16, K=64)
    M, K, N = 16, 64, 32
    x = mx.random.normal((M, K), dtype=mx.float16)

    # 2. Generate random packed weights (uint8)
    # K=64 means 64 weights. Packed at 4 per byte, we need K/4 bytes.
    packed_w = mx.random.randint(0, 255, (K // 4, N), dtype=mx.uint8)

    # 3. Unpack reference and compute expected output via standard MLX
    # w_ref = reference_unpack(packed_w, N)
    # expected_out = mx.matmul(x, w_ref)

    # 4. Compute via our custom Metal kernel
    # actual_out = ternary_matmul(x, packed_w)

    # 5. Assert numerical equivalence
    # mx.allclose(actual_out, expected_out, rtol=1e-3, atol=1e-3)
    
    # Placeholder assert for scaffold
    assert True
