import math
import mlx.core as mx
import pytest

from mlx_ternary import (
    fwht,
    TernaryLinear,
    TernarySwiGLU,
    TernaryQKV,
    ternary_swiglu,
    ternary_fused_qkv,
)


def test_fwht_numerical_inversion():
    """Verify numerical inversion of the normalized Sylvester-Walsh-Hadamard Transform."""
    mx.random.seed(42)
    K = 512
    block = 128

    x = mx.random.normal((2, K)).astype(mx.float16)
    signs = mx.where(mx.random.uniform(shape=(K,)) > 0.5, 1.0, -1.0).astype(mx.float32)

    x_fwd = fwht(x, block=block, signs=signs, inverse=False)
    x_rec = fwht(x_fwd, block=block, signs=signs, inverse=True)
    mx.eval(x, x_fwd, x_rec)

    assert mx.allclose(x, x_rec, atol=1e-3).item()


def test_hadamard_linear_forward():
    """Verify forward execution of TernaryLinear with Hadamard activation rotation."""
    mx.random.seed(42)
    K, N = 256, 128
    block = 128

    w_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    signs = mx.where(mx.random.uniform(shape=(K,)) > 0.5, 1.0, -1.0).astype(mx.float32)

    layer = TernaryLinear.from_raw(w_raw, block=block, signs=signs)
    x = mx.random.normal((1, K)).astype(mx.float16)

    out = layer(x)
    mx.eval(out)

    assert out.shape == (1, N)
    assert mx.isfinite(out).all().item()


def test_hadamard_swiglu_parity():
    """Verify TernarySwiGLU with Hadamard rotation matches pre-rotated ternary_swiglu."""
    mx.random.seed(42)
    K, N = 256, 128
    block = 128

    w_gate = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    w_up = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    signs = mx.where(mx.random.uniform(shape=(K,)) > 0.5, 1.0, -1.0).astype(mx.float32)

    layer = TernarySwiGLU.from_raw(w_gate, w_up, block=block, signs=signs)
    x = mx.random.normal((1, K)).astype(mx.float16)

    out = layer(x)
    x_rot = fwht(x, block=block, signs=signs)
    expected = ternary_swiglu(x_rot, layer.gate_packed, layer.up_packed, group_size=128)
    mx.eval(out, expected)

    assert mx.allclose(out, expected, atol=1e-3).item()


def test_hadamard_qkv_parity():
    """Verify TernaryQKV with Hadamard rotation matches pre-rotated ternary_fused_qkv."""
    mx.random.seed(42)
    K = 256
    q_dim = 128
    k_dim = 64
    v_dim = 64
    block = 128

    w_q = mx.random.randint(0, 3, (K, q_dim)).astype(mx.float16) - 1.0
    w_k = mx.random.randint(0, 3, (K, k_dim)).astype(mx.float16) - 1.0
    w_v = mx.random.randint(0, 3, (K, v_dim)).astype(mx.float16) - 1.0
    signs = mx.where(mx.random.uniform(shape=(K,)) > 0.5, 1.0, -1.0).astype(mx.float32)

    layer = TernaryQKV.from_raw(w_q, w_k, w_v, split=True, block=block, signs=signs)
    x = mx.random.normal((1, K)).astype(mx.float16)

    q, k, v = layer(x)
    x_rot = fwht(x, block=block, signs=signs)
    out_exp = ternary_fused_qkv(x_rot, layer.qkv_packed, group_size=128)
    q_exp = out_exp[..., :layer.q_dim]
    k_exp = out_exp[..., layer.q_dim : layer.q_dim + layer.k_dim]
    v_exp = out_exp[..., layer.q_dim + layer.k_dim :]
    mx.eval(q, k, v, q_exp, k_exp, v_exp)

    assert mx.allclose(q, q_exp, atol=1e-3).item()
    assert mx.allclose(k, k_exp, atol=1e-3).item()
    assert mx.allclose(v, v_exp, atol=1e-3).item()


def test_zero_overhead_invariant():
    """Verify block=0 produces exact bitwise array equality against unrotated baseline."""
    mx.random.seed(42)
    K, N = 256, 128

    w = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    x = mx.random.normal((1, K)).astype(mx.float16)

    layer_unrot = TernaryLinear.from_raw(w, block=0, signs=None)
    layer_default = TernaryLinear.from_raw(w)

    out_unrot = layer_unrot(x)
    out_default = layer_default(x)
    mx.eval(out_unrot, out_default)

    assert mx.array_equal(out_unrot, out_default).item()
