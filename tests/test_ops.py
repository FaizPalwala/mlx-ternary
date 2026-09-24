import mlx.core as mx
import mlx_ternary
import pytest

def pack_ternary_weights(w: mx.array) -> mx.array:
    """
    Simulates the 2-bit packing of ternary weights {-1, 0, 1} -> {2, 0, 1}
    Four weights are packed into a single uint8.
    """
    # Map weights: 1 -> 1, -1 -> 2, 0 -> 0
    w_mapped = mx.where(w == 1, 1, mx.where(w == -1, 2, 0)).astype(mx.uint8)
    
    K, N = w.shape
    assert K % 4 == 0, "K must be divisible by 4 for packing"
    
    # Reshape to (K//4, 4, N)
    w_reshaped = w_mapped.reshape(K // 4, 4, N)
    
    # Pack 4 values into 1 byte (little endian bit order based on our kernel unpack logic)
    # unpack_ternary logic: (packed >> (index * 2)) & 0x3
    w_packed = (
        (w_reshaped[:, 0, :] << 0) |
        (w_reshaped[:, 1, :] << 2) |
        (w_reshaped[:, 2, :] << 4) |
        (w_reshaped[:, 3, :] << 6)
    )
    return w_packed

def test_ternary_matmul_equivalence():
    M, K, N = 32, 128, 64
    
    # 1. Create FP16 Activations
    x = mx.random.normal((M, K)).astype(mx.float16)
    
    # 2. Create Ternary Weights (-1, 0, 1)
    # Generate random ints in [0, 3) and map to {-1, 0, 1}
    w_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    
    # 3. Create FP16 Scales
    scales = mx.random.uniform(0.5, 1.5, (N,)).astype(mx.float16)
    
    # 4. Standard FP16 Matmul (Ground Truth)
    # y = (x @ w_raw) * scales
    expected_out = mx.matmul(x, w_raw) * scales
    
    # 5. Pack Weights and run our custom Metal Kernel
    w_packed = pack_ternary_weights(w_raw)
    actual_out = mlx_ternary.ternary_matmul(x, w_packed, scales)
    
    # Evaluate to force GPU execution
    mx.eval(expected_out, actual_out)
    
    # 6. Verify numeric equivalence
    assert actual_out.shape == expected_out.shape, "Shape mismatch"
    
    # Calculate Max Absolute Error
    diff = mx.abs(actual_out - expected_out)
    max_err = mx.max(diff).item()
    
    # FP16 accumulation can have slight rounding differences compared to standard matmul
    assert max_err < 1e-2, f"Outputs diverge! Max absolute error: {max_err}"

def test_ternary_matmul_edge_cases():
    # Test batch dimensions
    B, M, K, N = 2, 16, 64, 32
    
    x = mx.random.normal((B, M, K)).astype(mx.float16)
    w_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    scales = mx.ones((N,), dtype=mx.float16)
    
    expected_out = mx.matmul(x, w_raw) * scales
    w_packed = pack_ternary_weights(w_raw)
    actual_out = mlx_ternary.ternary_matmul(x, w_packed, scales)
    
    mx.eval(expected_out, actual_out)
    
    max_err = mx.max(mx.abs(actual_out - expected_out)).item()
    assert max_err < 1e-2, f"Batched matmul failed. Max error: {max_err}"
