import mlx.core as mx
from mlx_ternary import TernaryLinear

def test_ternary_linear():
    # Standard shapes
    B, M, K, N = 2, 4, 1024, 2048
    
    layer = TernaryLinear(K, N, bias=True)
    
    # Fake FP16 weights like you'd get from a state_dict
    w_raw = mx.random.normal((N, K)).astype(mx.float16)
    
    # Quantize and inject
    layer.quantize_from_raw(w_raw)
    
    # Generate random activations
    x = mx.random.normal((B, M, K)).astype(mx.float16)
    
    # Forward pass
    out = layer(x)
    mx.eval(out)
    
    assert out.shape == (B, M, N)
    print("TernaryLinear forward pass successful!")

if __name__ == "__main__":
    test_ternary_linear()
