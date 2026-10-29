import mlx.core as mx
try:
    import pytest
except ImportError:
    pytest = None

import mlx_ternary
from mlx_ternary.quantize import pack_interleaved

def check_matmul_equivalence(M, K, N, group_size):
    mx.random.seed(42)
    x = mx.random.normal((M, K)).astype(mx.float16)
    w_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    w_interleaved, _ = pack_interleaved(w_raw, group_size=group_size)
    
    K_blocks = K // group_size
    w_grouped = w_raw.reshape(K_blocks, group_size, N)
    scales = mx.abs(w_grouped).max(axis=1).astype(mx.float16)
    safe_scales = mx.where(scales == 0, 1.0, scales)
    w_norm = w_grouped / safe_scales[:, None, :]
    w_ternary = mx.where(w_norm > 0.33, 1, mx.where(w_norm < -0.33, -1, 0))
    w_reconstructed = (w_ternary * scales[:, None, :]).reshape(K, N)
    
    expected = mx.matmul(x, w_reconstructed)
    actual = mlx_ternary.ternary_matmul(x, w_interleaved, group_size=group_size)
    mx.eval(expected, actual)
    
    assert actual.shape == expected.shape
    rel_diff = (mx.abs(actual - expected) / (mx.abs(expected) + 1e-3)).mean().item()
    diff = mx.abs(actual - expected).max().item()
    print(f"M={M:2d}, G={group_size:3d} | Max diff: {diff:.4f}, Mean rel diff: {rel_diff:.4f}")
    assert rel_diff < 0.03, f"Mean relative diff exceeded threshold: {rel_diff}"

def test_gemv_group128():
    check_matmul_equivalence(M=1, K=256, N=128, group_size=128)

def test_gemv_group150():
    check_matmul_equivalence(M=1, K=300, N=128, group_size=150)

def test_gemm_group128():
    check_matmul_equivalence(M=32, K=256, N=64, group_size=128)

def test_gemm_group150():
    check_matmul_equivalence(M=32, K=300, N=64, group_size=150)

def test_batched_input():
    x = mx.random.normal((2, 3, 256)).astype(mx.float16)
    w_raw = mx.random.randint(0, 3, (256, 64)).astype(mx.float16) - 1.0
    w_interleaved, _ = pack_interleaved(w_raw, group_size=128)
    out = mlx_ternary.ternary_matmul(x, w_interleaved, group_size=128)
    assert out.shape == (2, 3, 64)

if __name__ == "__main__":
    test_gemv_group128()
    test_gemv_group150()
    test_gemm_group128()
    test_gemm_group150()
    test_batched_input()
