import mlx.core as mx
import pytest
import mlx_ternary
from mlx_ternary.quantize import pack_interleaved

def test_ternary_matmul_equivalence():
    M, K, N = 32, 300, 64
    x = mx.random.normal((M, K)).astype(mx.float16)
    
    # Generate ternary weights
    w_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    
    # Pack them using our new interleaved packer
    w_interleaved, pad_k = pack_interleaved(w_raw, group_size=150)
    
    # Compute the expected FP16 result manually via Group-wise decoding
    # w_interleaved has shape (K/150, N, 32). The actual math uses group-wise FP16 scales.
    # To get equivalence, we just use standard matrix multiplication with the quantized weights!
    # Wait, the quantized weights have scales applied per block.
    # We can reconstruct the unquantized weights:
    K_blocks = K // 150
    w_grouped = w_raw.reshape(K_blocks, 150, N)
    scales = mx.abs(w_grouped).max(axis=1).astype(mx.float16)
    safe_scales = mx.where(scales == 0, 1.0, scales)
    w_norm = w_grouped / safe_scales[:, None, :]
    w_ternary = mx.where(w_norm > 0.33, 1, mx.where(w_norm < -0.33, -1, 0))
    w_reconstructed = (w_ternary * scales[:, None, :]).reshape(K, N)
    
    expected = mx.matmul(x, w_reconstructed)
    
    # Compute our custom kernel result
    actual = mlx_ternary.ternary_matmul(x, w_interleaved)
    
    mx.eval(expected, actual)
    
    assert actual.shape == expected.shape
    
    # We allow slight differences due to FP16 accumulation order (AMX/GPU vs CPU)
    diff = mx.abs(actual - expected).max().item()
    print(f"Max diff: {diff}")
    assert diff < 1e-2

if __name__ == "__main__":
    test_ternary_matmul_equivalence()
