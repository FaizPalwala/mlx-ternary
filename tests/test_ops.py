import mlx.core as mx
import mlx_ternary
import pytest

def pack_ternary_weights(w: mx.array) -> mx.array:
    """
    Packs 5 ternary weights {-1, 0, 1} into a single byte using Base-3 encoding.
    This achieves 1.6 bits per weight (3^5 = 243 <= 256).
    """
    K, N = w.shape
    pad_K = (5 - (K % 5)) % 5
    
    if pad_K > 0:
        w = mx.concatenate([w, mx.zeros((pad_K, N))], axis=0)
    
    K_padded = w.shape[0]
    
    # Map weights: 0 -> 0, 1 -> 1, -1 -> 2
    w_mapped = mx.where(w == 1, 1, mx.where(w == -1, 2, 0)).astype(mx.uint16)
    
    # Reshape to groups of 5
    w_reshaped = w_mapped.reshape(K_padded // 5, 5, N)
    
    # Base-3 packing: w0 + w1*3 + w2*9 + w3*27 + w4*81
    w_packed = (
        w_reshaped[:, 0, :] * 1 +
        w_reshaped[:, 1, :] * 3 +
        w_reshaped[:, 2, :] * 9 +
        w_reshaped[:, 3, :] * 27 +
        w_reshaped[:, 4, :] * 81
    ).astype(mx.uint8)
    
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
