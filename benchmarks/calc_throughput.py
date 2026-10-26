import time
import mlx.core as mx
import mlx_ternary
from tests.test_ops import pack_ternary_weights

def benchmark_throughput():
    # Amortize launch overhead by generating 50 tokens (M=50 sequentially, simulated via loop)
    B, M, K, N = 1, 1, 8192, 8192
    
    x = mx.random.normal((B, M, K)).astype(mx.float16)
    w_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    scales = mx.random.uniform(0.5, 1.5, (N,)).astype(mx.float16)
    w_packed = pack_ternary_weights(w_raw)
    
    @mx.compile
    def custom_ternary_matmul(x_in, w_p, s_in):
        return mlx_ternary.ternary_matmul(x_in, w_p, s_in)
        
    for _ in range(5):
        mx.eval(custom_ternary_matmul(x, w_packed, scales))
        
    iters = 100
    
    start_time = time.perf_counter()
    for _ in range(iters):
        y = custom_ternary_matmul(x, w_packed, scales)
        mx.eval(y)
    
    total_time = time.perf_counter() - start_time
    time_per_iter = total_time / iters
    
    mem_bytes = (K * N) / 5
    bandwidth = (mem_bytes / time_per_iter) / 1e9
    
    print(f"Time per forward pass: {time_per_iter * 1000:.3f} ms")
    print(f"Effective Memory Bandwidth: {bandwidth:.2f} GB/s")

if __name__ == "__main__":
    benchmark_throughput()
