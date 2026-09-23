import mlx.core as mx
import mlx_ternary

def test_api_boundary():
    print("[1] Creating mock tensors in MLX...")
    x = mx.random.normal((32, 128)).astype(mx.float16)
    
    # Packed weights: (128 / 4, 64) -> (32, 64). We'll use uint8 for packing.
    # 85 is 01010101 in binary, which corresponds to four 1s (mapped to 1.0).
    w_packed = mx.full((32, 64), 85, dtype=mx.uint8) 
    
    scales = mx.ones((64,), dtype=mx.float16)
    
    print("[2] Hitting the C++ Extension Boundary...")
    try:
        # This will call mlx_ternary_ext.ternary_matmul internally
        out = mlx_ternary.ternary_matmul(x, w_packed, scales)
        
        # We need to evaluate the graph to force the GPU dispatch
        mx.eval(out)
        
        print("\n[3] Execution Successful!")
        print(f"Output shape: {out.shape}")
        print(f"Output mean: {out.mean().item()}")
        
    except Exception as e:
        print(f"\n[3] Caught error from C++ runtime:")
        print(f"--- {type(e).__name__} ---")
        print(e)
        print("--------------------")

if __name__ == "__main__":
    test_api_boundary()
