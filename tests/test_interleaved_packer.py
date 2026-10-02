import mlx.core as mx
from mlx_ternary.quantize import pack_interleaved

def test_interleaved():
    K, N = 1500, 256
    w_raw = mx.random.uniform(-1.5, 1.5, (K, N))
    
    interleaved, pad_k = pack_interleaved(w_raw)
    
    print("Interleaved shape:", interleaved.shape)
    print("Expected footprint: 1.706 bits/weight")
    
    actual_bits = (interleaved.size * 8) / (K * N)
    print(f"Actual footprint: {actual_bits:.3f} bits/weight")
    assert abs(actual_bits - 1.706) < 0.01

if __name__ == "__main__":
    test_interleaved()
