import time
import mlx.core as mx
import mlx_ternary
from tests.test_ops import pack_ternary_weights

def benchmark_matmul():
    # Large dimensions typical for an LLM
    B, M, K, N = 1, 1, 4096, 4096
    
    print(f"Benchmarking Matrix Multiplication (M={M}, K={K}, N={N})")
    print("-" * 50)
    
    # 1. Prepare inputs
    x = mx.random.normal((B, M, K)).astype(mx.float16)
    
    # Weights in [-1, 0, 1]
    w_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    scales = mx.random.uniform(0.5, 1.5, (N,)).astype(mx.float16)
    
    # Pack weights for custom kernel
    w_packed = pack_ternary_weights(w_raw)
    
    # Compile functions using mx.compile for maximum performance
    @mx.compile
    def standard_matmul(x_in, w_in, s_in):
        return mx.matmul(x_in, w_in) * s_in
        
    @mx.compile
    def custom_ternary_matmul(x_in, w_p, s_in):
        return mlx_ternary.ternary_matmul(x_in, w_p, s_in)
        
    # Warmup
    for _ in range(5):
        y_std = standard_matmul(x, w_raw, scales)
        mx.eval(y_std)
        
        y_cust = custom_ternary_matmul(x, w_packed, scales)
        mx.eval(y_cust)
        
    # Benchmark Standard
    iters = 100
    mx.metal.clear_cache()
    
    start_time = time.perf_counter()
    for _ in range(iters):
        y_std = standard_matmul(x, w_raw, scales)
        mx.eval(y_std)
    std_time = (time.perf_counter() - start_time) / iters
    
    # Benchmark Custom
    mx.metal.clear_cache()
    start_time = time.perf_counter()
    for _ in range(iters):
        y_cust = custom_ternary_matmul(x, w_packed, scales)
        mx.eval(y_cust)
    cust_time = (time.perf_counter() - start_time) / iters
    
    print(f"Standard FP16 Matmul: {std_time * 1000:.3f} ms")
    print(f"Custom Ternary Matmul: {cust_time * 1000:.3f} ms")
    print(f"Speedup: {std_time / cust_time:.2f}x")

if __name__ == "__main__":
    benchmark_matmul()
