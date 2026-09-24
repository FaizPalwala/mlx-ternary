import time
import mlx.core as mx
import mlx_ternary
from mlx_ternary.quantize import pack_interleaved

def benchmark():
    B, M, K, N = 1, 1, 4050, 4096
    
    x = mx.random.normal((B, M, K)).astype(mx.float16)
    w_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    w_interleaved, pad_k = pack_interleaved(w_raw)
    
    @mx.compile
    def custom_ternary_matmul(x_in, w_in):
        return mlx_ternary.ternary_matmul(x_in, w_in)
        
    @mx.compile
    def standard_fp16_matmul(x_in, w_in):
        return mx.matmul(x_in, w_in)
        
    print(f"Benchmarking Matrix Multiplication (M={M}, K={K}, N={N})")
    print("-" * 50)
    
    # Warmup
    for _ in range(10):
        mx.eval(standard_fp16_matmul(x, w_raw))
        mx.eval(custom_ternary_matmul(x, w_interleaved))
        
    iters = 1000
    
    # FP16
    start = time.perf_counter()
    for _ in range(iters):
        y = standard_fp16_matmul(x, w_raw)
        mx.eval(y)
    fp16_time = (time.perf_counter() - start) / iters
    
    # Custom
    start = time.perf_counter()
    for _ in range(iters):
        y = custom_ternary_matmul(x, w_interleaved)
        mx.eval(y)
    custom_time = (time.perf_counter() - start) / iters
    
    print(f"Standard FP16 Matmul: {fp16_time * 1000:.3f} ms")
    print(f"Custom Ternary Matmul: {custom_time * 1000:.3f} ms")
    print(f"Speedup: {fp16_time / custom_time:.2f}x")

if __name__ == "__main__":
    benchmark()
