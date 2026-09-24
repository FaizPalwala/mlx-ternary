import mlx.core as mx
import numpy as np
from pathlib import Path
import json

def pack_interleaved(w_raw, group_size=150):
    """
    Packs a raw FP16 weight matrix into the 1.6-bit Base-3 Interleaved format.
    Group Size must be exactly 150.
    Returns a uint8 tensor of shape (K // 150, N, 32).
    Each 32-byte block contains a 2-byte FP16 scale and 30-bytes (150 weights) of Base-3.
    """
    assert group_size == 150, "Hardware Metal kernel currently requires group_size=150"
    K, N = w_raw.shape
    
    pad_k = (group_size - (K % group_size)) % group_size
    if pad_k > 0:
        w_raw = mx.concatenate([w_raw, mx.zeros((pad_k, N), dtype=w_raw.dtype)], axis=0)
        K = w_raw.shape[0]

    # 1. Calculate FP16 scales per block (max abs value)
    w_grouped = w_raw.reshape(K // group_size, group_size, N)
    scales = mx.abs(w_grouped).max(axis=1).astype(mx.float16)
    
    # Avoid div by zero
    safe_scales = mx.where(scales == 0, 1.0, scales)
    
    # 2. Quantize to {-1, 0, 1}
    w_norm = w_grouped / safe_scales[:, None, :]
    w_ternary = mx.where(w_norm > 0.33, 1, mx.where(w_norm < -0.33, -1, 0))
    
    # 3. Pack 5 weights into 1 byte (Base-3)
    w_shifted = w_ternary + 1  # Map {-1, 0, 1} -> {0, 1, 2}
    w_shifted = w_shifted.astype(mx.uint32)
    
    blocks = K // group_size
    w_packed = mx.zeros((blocks, group_size // 5, N), dtype=mx.uint8)
    
    for i in range(5):
        w_packed = w_packed + (w_shifted[:, i::5, :] * (3 ** i)).astype(mx.uint8)
        
    # 4. Interleave Scales (2 bytes) and Weights (30 bytes)
    # MLX doesn't have a view_as(uint8) for FP16, so we use numpy for the exact byte conversion
    scales_np = np.array(scales, copy=False)
    scales_uint8 = mx.array(np.frombuffer(scales_np, dtype=np.uint8)).reshape(blocks, N, 2)
    
    w_packed_transposed = mx.transpose(w_packed, (0, 2, 1)) # (blocks, N, 30)
    
    interleaved = mx.concatenate([scales_uint8, w_packed_transposed], axis=2) # (blocks, N, 32)
    
    return interleaved, pad_k

def quantize_model(input_path: str, output_path: str):
    """
    Ingests an MLX safetensors model and quantizes all Linear layers.
    """
    print(f"Loading weights from {input_path}")
    weights = mx.load(input_path)
    
    quantized_weights = {}
    
    for key, w in weights.items():
        if w.ndim == 2: # Linear layer
            print(f"Quantizing {key} | Shape: {w.shape}")
            # Transpose to (K, N) if necessary, assuming (N, K) standard PyTorch/MLX linear
            # We pack along K
            interleaved, pad_k = pack_interleaved(w.T)
            quantized_weights[f"{key}.weight_packed"] = interleaved
            if pad_k > 0:
                quantized_weights[f"{key}.pad_k"] = mx.array(pad_k, dtype=mx.uint32)
        else:
            quantized_weights[key] = w
            
    print(f"Saving quantized model to {output_path}")
    mx.save_safetensors(output_path, quantized_weights)
    print("Done! Model is now running at 1.706 bits/weight.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input .safetensors")
    parser.add_argument("output", help="Path to output .safetensors")
    args = parser.parse_args()
    quantize_model(args.input, args.output)
