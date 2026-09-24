import mlx.core as mx
import numpy as np

def pack_interleaved(w_raw, group_size=150):
    K, N = w_raw.shape
    assert K % group_size == 0, f"K must be divisible by {group_size}"
    
    # 1. Calculate FP16 scales per block (max abs value)
    w_grouped = w_raw.reshape(K // group_size, group_size, N)
    scales = mx.abs(w_grouped).max(axis=1).astype(mx.float16)
    
    # Avoid div by zero
    safe_scales = mx.where(scales == 0, 1.0, scales)
    
    # 2. Quantize to {-1, 0, 1}
    # For ternary, we typically scale by 1.0 / scale
    w_norm = w_grouped / safe_scales[:, None, :]
    
    w_ternary = mx.where(w_norm > 0.33, 1, mx.where(w_norm < -0.33, -1, 0))
    
    # 3. Pack 5 weights into 1 byte (Base-3)
    w_shifted = w_ternary + 1  # Map {-1, 0, 1} -> {0, 1, 2}
    w_shifted = w_shifted.astype(mx.uint32)
    
    blocks, _, _ = w_shifted.shape
    w_packed = mx.zeros((blocks, group_size // 5, N), dtype=mx.uint8)
    
    for i in range(5):
        w_packed = w_packed + (w_shifted[:, i::5, :] * (3 ** i)).astype(mx.uint8)
        
    # 4. Interleave Scales (2 bytes) and Weights (30 bytes)
    scales_uint8 = mx.array(np.frombuffer(np.array(scales), dtype=np.uint8)).reshape(blocks, N, 2)
    
    # We want final shape to be (blocks, N, 32)
    w_packed_transposed = mx.transpose(w_packed, (0, 2, 1)) # (blocks, N, 30)
    
    interleaved = mx.concatenate([scales_uint8, w_packed_transposed], axis=2) # (blocks, N, 32)
    return interleaved, w_ternary, scales

def test_interleaved():
    K, N = 1500, 256
    w_raw = mx.random.uniform(-1.5, 1.5, (K, N))
    
    interleaved, w_ternary, scales = pack_interleaved(w_raw)
    
    print("Interleaved shape:", interleaved.shape)
    print("Expected footprint: 1.706 bits/weight")
    
    actual_bits = (interleaved.size * 8) / (K * N)
    print(f"Actual footprint: {actual_bits:.3f} bits/weight")
    assert abs(actual_bits - 1.706) < 0.01

if __name__ == "__main__":
    test_interleaved()
