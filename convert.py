import mlx.core as mx
import numpy as np
import argparse
import time

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
    w_shifted = w_ternary + 1  
    w_shifted = w_shifted.astype(mx.uint32)
    
    pad_5 = (5 - (group_size % 5)) % 5
    if pad_5 > 0:
        pad_tensor = mx.ones((K // group_size, pad_5, N), dtype=mx.uint32)
        w_shifted = mx.concatenate([w_shifted, pad_tensor], axis=1)
        
    num_bytes = w_shifted.shape[1] // 5
    blocks = K // group_size
    w_packed = mx.zeros((blocks, num_bytes, N), dtype=mx.uint8)
    
    for i in range(5):
        w_packed = w_packed + (w_shifted[:, i::5, :] * (3 ** i)).astype(mx.uint8)
        
    scales_np = np.array(scales, copy=False)
    scales_uint8 = mx.array(np.frombuffer(scales_np, dtype=np.uint8)).reshape(blocks, N, 2)
    w_packed_transposed = mx.transpose(w_packed, (0, 2, 1))
    
    pad_bytes = 30 - num_bytes
    if pad_bytes > 0:
        padding = mx.zeros((blocks, N, pad_bytes), dtype=mx.uint8)
        interleaved = mx.concatenate([scales_uint8, w_packed_transposed, padding], axis=2)
    else:
        interleaved = mx.concatenate([scales_uint8, w_packed_transposed], axis=2)
        
    return interleaved, pad_k

def convert(input_path, output_path):
    print(f"Loading {input_path}...")
    weights = mx.load(input_path)
    new_weights = {}
    
    prefixes = set()
    for k in weights.keys():
        if ".weight" in k and weights[k].dtype == mx.uint32:
            prefixes.add(k.replace(".weight", ""))
            
    for k, v in weights.items():
        if "signs" in k:
            new_weights[k] = v 
            continue
            
        if any(k.startswith(p) for p in prefixes):
            continue 
            
        new_weights[k] = v 
        
    print(f"Found {len(prefixes)} layers to transcode.")
    start = time.time()
    
    for i, p in enumerate(prefixes):
        print(f"[{i+1}/{len(prefixes)}] Transcoding {p} ...", end="", flush=True)
        w = weights[f"{p}.weight"]
        s = weights[f"{p}.scales"]
        b = weights[f"{p}.biases"]
        
        w_fp16 = mx.dequantize(w, s, b, group_size=128, bits=2)
        w_interleaved, pad_k = pack_interleaved(w_fp16.T, group_size=128)
        mx.eval(w_interleaved)
        
        new_weights[f"{p}.weight_packed"] = w_interleaved
        del weights[f"{p}.weight"]
        del weights[f"{p}.scales"]
        del weights[f"{p}.biases"]
        if pad_k > 0:
            new_weights[f"{p}.pad_k"] = mx.array(pad_k, dtype=mx.uint32)
            
        print(" done.")
            
    print(f"Saving transcoded model to {output_path} (took {time.time()-start:.2f}s)...")
    mx.save_safetensors(output_path, new_weights)
    print("Done!")

if __name__ == "__main__":
    import sys
    convert(sys.argv[1], sys.argv[2])
