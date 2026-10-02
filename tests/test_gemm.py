import mlx.core as mx
import mlx_ternary
import time
from mlx_ternary.quantize import pack_interleaved

def test_gemm():
    B, M, K, N = 1, 32, 150, 32
    
    x = mx.random.normal((B, M, K)).astype(mx.float16)
    w_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    w_interleaved, pad_k = pack_interleaved(w_raw)
    
    @mx.compile
    def custom_ternary_matmul(x_in, w_in):
        return mlx_ternary.ternary_matmul(x_in, w_in)
        
    K_blocks = K // 150
    w_grouped = w_raw.reshape(K_blocks, 150, N)
    scales = mx.abs(w_grouped).max(axis=1).astype(mx.float16)
    safe_scales = mx.where(scales == 0, 1.0, scales)
    w_norm = w_grouped / safe_scales[:, None, :]
    w_ternary = mx.where(w_norm > 0.33, 1, mx.where(w_norm < -0.33, -1, 0))
    w_reconstructed = (w_ternary * scales[:, None, :]).reshape(K, N)
    expected = mx.matmul(x, w_reconstructed)
    mx.eval(expected)

    actual = custom_ternary_matmul(x, w_interleaved)
    mx.eval(actual)
    
    print("Expected:")
    print(expected)
    print("Actual:")
    print(actual)

if __name__ == "__main__":
    test_gemm()
