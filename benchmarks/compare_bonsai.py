import time
import mlx.core as mx
import mlx_ternary
from mlx_ternary.quantize import pack_interleaved

def benchmark_shape(name: str, M: int, K: int, N: int, iters: int = 50):
    print(f"\n[{name}] Shape: M={M}, K={K}, N={N}")
    
    # 1. Prepare inputs
    x = mx.random.normal((M, K)).astype(mx.float16)
    
    # Weights for MLX standard 2-bit affine quantized_matmul (Normal Bonsai baseline)
    # transpose=True expects w: (N, K // 16), scales: (N, K // 128), biases: (N, K // 128)
    w_2bit = mx.random.randint(0, 255, (N, K // 16), dtype=mx.uint32)
    scales_2bit = mx.random.normal((N, K // 128)).astype(mx.float16)
    biases_2bit = mx.random.normal((N, K // 128)).astype(mx.float16)
    
    @mx.compile
    def mlx_quantized_f(act, w, s, b):
        return mx.quantized_matmul(act, w, s, b, transpose=True, group_size=128, bits=2)
    
    # Weights for mlx_ternary JIT Metal engine
    w_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    w_interleaved, _ = pack_interleaved(w_raw, group_size=128)
    
    def mlx_ternary_f(act, w_pack):
        return mlx_ternary.ternary_matmul(act, w_pack, group_size=128)
    
    # Warmup
    for _ in range(5):
        mx.eval(mlx_quantized_f(x, w_2bit, scales_2bit, biases_2bit))
        mx.eval(mlx_ternary_f(x, w_interleaved))
        
    # Benchmark Normal Bonsai 2-bit
    start = time.perf_counter()
    for _ in range(iters):
        out = mlx_quantized_f(x, w_2bit, scales_2bit, biases_2bit)
        mx.eval(out)
    time_normal = (time.perf_counter() - start) / iters
    
    # Benchmark mlx_ternary JIT Metal
    start = time.perf_counter()
    for _ in range(iters):
        out = mlx_ternary_f(x, w_interleaved)
        mx.eval(out)
    time_ternary = (time.perf_counter() - start) / iters
    
    bytes_normal = (K // 16) * N * 4 + (K // 128) * N * 4
    bytes_ternary = (K // 128) * N * 32
    
    bw_normal = (bytes_normal / 1e9) / time_normal
    bw_ternary = (bytes_ternary / 1e9) / time_ternary
    speedup = time_normal / time_ternary
    
    print(f"  * Normal Bonsai (MLX 2-bit): {time_normal * 1e6:8.2f} us | BW: {bw_normal:5.2f} GB/s | Data: {bytes_normal / 1024 / 1024:.2f} MB")
    print(f"  * mlx-ternary (1.6-bit JIT): {time_ternary * 1e6:8.2f} us | BW: {bw_ternary:5.2f} GB/s | Data: {bytes_ternary / 1024 / 1024:.2f} MB")
    print(f"  * Speedup: {speedup:.2f}x ({'+' if speedup > 1 else ''}{(speedup - 1) * 100:.1f}%)")
    
    return time_normal, time_ternary

def main():
    print("=" * 70)
    print("BONSAI 2 (27B) HARDWARE INFERENCE BENCHMARK")
    print("Target SoC: Apple M4 Pro (20-core GPU, 24GB Unified Memory)")
    print("Comparing: Normal Bonsai 2-bit (mx.quantized_matmul) vs mlx-ternary JIT")
    print("=" * 70)
    
    layers = [
        ("MLP Gate/Up Proj", 1, 5120, 17408),
        ("MLP Down Proj",    1, 17408, 5120),
        ("Attention Q Proj", 1, 5120, 3072),
        ("Attention KV Proj",1, 5120, 512),
        ("Attention O Proj", 1, 3072, 5120),
    ]
    
    total_time_normal_layer = 0.0
    total_time_ternary_layer = 0.0
    
    # Layer counts per transformer block:
    # 2x Gate/Up, 1x Down, 1x Q, 2x KV (K and V), 1x O
    layer_multipliers = {
        "MLP Gate/Up Proj": 2,
        "MLP Down Proj": 1,
        "Attention Q Proj": 1,
        "Attention KV Proj": 2,
        "Attention O Proj": 1,
    }
    
    for name, M, K, N in layers:
        t_norm, t_tern = benchmark_shape(name, M, K, N, iters=30)
        mult = layer_multipliers[name]
        total_time_normal_layer += t_norm * mult
        total_time_ternary_layer += t_tern * mult
        
    num_layers = 64
    full_model_time_normal = total_time_normal_layer * num_layers
    full_model_time_ternary = total_time_ternary_layer * num_layers
    
    tok_sec_normal = 1.0 / full_model_time_normal
    tok_sec_ternary = 1.0 / full_model_time_ternary
    
    # Speculative decoding projection (assuming draft acceptance length alpha = 3.5x over 5 speculative tokens)
    # Verification cost using AMX GEMM is ~1 sweep per 4-5 tokens
    tok_sec_speculative = tok_sec_ternary * 2.1
    
    print("\n" + "=" * 70)
    print("END-TO-END 27B PROJECTION ACROSS 64 LAYERS")
    print("=" * 70)
    print(f"Normal Bonsai 2-bit (Single Token Step): {full_model_time_normal * 1000:6.2f} ms | ~{tok_sec_normal:5.1f} tok/s")
    print(f"mlx-ternary Pure JIT (Single Token Step): {full_model_time_ternary * 1000:6.2f} ms | ~{tok_sec_ternary:5.1f} tok/s")
    print(f"mlx-ternary + Speculative Verification:   ~{1000.0 / tok_sec_speculative:6.2f} ms | ~{tok_sec_speculative:5.1f} tok/s")
    print(f"\nNet Speedup: {tok_sec_ternary / tok_sec_normal:.2f}x (Autoregressive) | {tok_sec_speculative / tok_sec_normal:.2f}x (Speculative)")
    print("=" * 70)

if __name__ == "__main__":
    main()
