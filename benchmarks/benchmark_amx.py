import mlx.core as mx
import mlx_ternary
import time
from mlx_ternary.quantize import pack_interleaved

def benchmark():
    B, M, K, N = 1, 1024, 4050, 4096
    
    x = mx.random.normal((B, M, K)).astype(mx.float16)
    w_raw = mx.random.normal((K, N)).astype(mx.float16)
    w_interleaved, _ = pack_interleaved(w_raw)
    
    @mx.compile
    def custom_ternary_matmul(x_in, w_in):
        return mlx_ternary.ternary_matmul(x_in, w_in)
        
    # Warmup
    for _ in range(5):
        mx.eval(custom_ternary_matmul(x, w_interleaved))
        
    # Timing
    iters = 100
    start = time.perf_counter()
    for _ in range(iters):
        mx.eval(custom_ternary_matmul(x, w_interleaved))
    end = time.perf_counter()
    
    ms_per_iter = (end - start) * 1000 / iters
    # Memory bandwidth:
    # A read: M * K * 2 bytes
    # B read: (K/150) * N * 32 bytes
    # C write: M * N * 2 bytes
    bytes_per_iter = (M * K * 2) + ((K // 150) * N * 32) + (M * N * 2)
    gb_s = (bytes_per_iter / 1e9) / (ms_per_iter / 1000)
    
    print(f"AMX Prefill Kernel (M={M}, K={K}, N={N})")
    print(f"Time: {ms_per_iter:.3f} ms")
    print(f"Bandwidth: {gb_s:.2f} GB/s")

if __name__ == "__main__":
    benchmark()
