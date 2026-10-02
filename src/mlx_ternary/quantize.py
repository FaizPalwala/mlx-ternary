import mlx.core as mx

def pack_interleaved(w_raw, group_size=150):
    assert group_size in [128, 150], "Hardware Metal kernel currently requires group_size=128 or 150"
    K, N = w_raw.shape
    
    pad_k = (group_size - (K % group_size)) % group_size
    if pad_k > 0:
        w_raw = mx.concatenate([w_raw, mx.zeros((pad_k, N), dtype=w_raw.dtype)], axis=0)
        K = w_raw.shape[0]

    w_grouped = w_raw.reshape(K // group_size, group_size, N)
    scales = mx.abs(w_grouped).max(axis=1).astype(mx.float16)
    
    safe_scales = mx.where(scales == 0, 1.0, scales)
    
    w_norm = w_grouped / safe_scales[:, None, :]
    w_ternary = mx.where(w_norm > 0.33, 1, mx.where(w_norm < -0.33, -1, 0))
    
    w_shifted = w_ternary + 1  # Map {-1, 0, 1} -> {0, 1, 2}
    w_shifted = w_shifted.astype(mx.uint32)
    
    # Pad to multiple of 5
    pad_5 = (5 - (group_size % 5)) % 5
    if pad_5 > 0:
        # Pad with 1 (which represents 0 in Base-3, so it adds nothing)
        pad_tensor = mx.ones((K // group_size, pad_5, N), dtype=mx.uint32)
        w_shifted = mx.concatenate([w_shifted, pad_tensor], axis=1)
        
    num_bytes = w_shifted.shape[1] // 5
    blocks = K // group_size
    
    w_packed = mx.zeros((blocks, num_bytes, N), dtype=mx.uint8)
    
    for i in range(5):
        w_packed = w_packed + (w_shifted[:, i::5, :] * (3 ** i)).astype(mx.uint8)
        
    scales_uint8 = scales.view(mx.uint8).reshape(blocks, N, 2)
    
    w_packed_transposed = mx.transpose(w_packed, (0, 2, 1)) # (blocks, N, num_bytes)
    
    # We must hit exactly 32 bytes per block for optimal cache-lines
    # scales = 2 bytes. We need 30 bytes of payload.
    pad_bytes = 30 - num_bytes
    if pad_bytes > 0:
        padding = mx.zeros((blocks, N, pad_bytes), dtype=mx.uint8)
        interleaved = mx.concatenate([scales_uint8, w_packed_transposed, padding], axis=2)
    else:
        interleaved = mx.concatenate([scales_uint8, w_packed_transposed], axis=2)
        
    return interleaved, pad_k
