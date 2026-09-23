import mlx.core as mx
import mlx_ternary

def test_api_boundary():
    print("[1] Creating mock tensors in MLX...")
    x = mx.random.normal((32, 128))
    # Packed weights: (128 / 4, 64) -> (32, 64)
    w_packed = mx.random.uniform(shape=(32, 64))
    scales = mx.random.normal((64,))
    
    print("[2] Hitting the C++ Extension Boundary...")
    try:
        # This will call mlx_ternary_ext.ternary_matmul internally
        out = mlx_ternary.ternary_matmul(x, w_packed, scales)
        
        # We need to evaluate the graph to force the GPU dispatch
        mx.eval(out)
        print("Wait, it succeeded? This shouldn't happen because we put a throw std::runtime_error in eval_gpu!")
        
    except Exception as e:
        print(f"\n[3] Caught expected error from C++ runtime:")
        print(f"--- {type(e).__name__} ---")
        print(e)
        print("--------------------")
        
        if "Metal dispatch logic written but requires MLX MTL headers" in str(e):
            print("\nSUCCESS! The PyBind11 bindings successfully translated mx.array to mlx::core::array, built the graph, and executed our C++ Primitive!")
        else:
            print("\nWARNING: This is not the error we expected. Pybind11 might be failing to cast MLX arrays.")

if __name__ == "__main__":
    test_api_boundary()
