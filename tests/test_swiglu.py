import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_ternary import ternary_swiglu, TernarySwiGLU
from mlx_ternary.quantize import pack_interleaved


def _compute_reference_swiglu(x: mx.array, w_gate_raw: mx.array, w_up_raw: mx.array, group_size: int) -> mx.array:
    """Compute mathematical reference SwiGLU using exact unpacked ternary representation."""
    K, N = w_gate_raw.shape
    K_blocks = K // group_size

    def reconstruct_weights(w_raw):
        w_grouped = w_raw.reshape(K_blocks, group_size, N)
        scales = mx.abs(w_grouped).max(axis=1).astype(mx.float16)
        safe_scales = mx.where(scales == 0, 1.0, scales)
        w_norm = w_grouped / safe_scales[:, None, :]
        w_ternary = mx.where(w_norm > 0.33, 1, mx.where(w_norm < -0.33, -1, 0))
        return (w_ternary * scales[:, None, :]).reshape(K, N)

    w_g_recon = reconstruct_weights(w_gate_raw)
    w_u_recon = reconstruct_weights(w_up_raw)

    g = mx.matmul(x, w_g_recon)
    u = mx.matmul(x, w_u_recon)
    ref = (nn.silu(g.astype(mx.float32)) * u.astype(mx.float32)).astype(mx.float16)
    mx.eval(ref)
    return ref


def test_swiglu_parity_group128():
    """Verify numerical parity of fused SwiGLU against reference for group_size=128."""
    mx.random.seed(42)
    M, K, N = 1, 512, 1024
    G = 128

    x = mx.random.normal((M, K)).astype(mx.float16)
    w_g_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    w_u_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0

    w_g_packed, _ = pack_interleaved(w_g_raw, group_size=G)
    w_u_packed, _ = pack_interleaved(w_u_raw, group_size=G)

    expected = _compute_reference_swiglu(x, w_g_raw, w_u_raw, group_size=G)
    actual = ternary_swiglu(x, w_g_packed, w_u_packed, group_size=G)
    mx.eval(expected, actual)

    assert actual.shape == expected.shape
    rel_diff = (mx.abs(actual - expected) / (mx.abs(expected) + 1e-3)).mean().item()
    assert rel_diff < 0.03, f"Group 128 SwiGLU rel diff exceeded threshold: {rel_diff}"


def test_swiglu_parity_group150():
    """Verify numerical parity of fused SwiGLU against reference for group_size=150."""
    mx.random.seed(42)
    M, K, N = 1, 600, 1024
    G = 150

    x = mx.random.normal((M, K)).astype(mx.float16)
    w_g_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    w_u_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0

    w_g_packed, _ = pack_interleaved(w_g_raw, group_size=G)
    w_u_packed, _ = pack_interleaved(w_u_raw, group_size=G)

    expected = _compute_reference_swiglu(x, w_g_raw, w_u_raw, group_size=G)
    actual = ternary_swiglu(x, w_g_packed, w_u_packed, group_size=G)
    mx.eval(expected, actual)

    assert actual.shape == expected.shape
    rel_diff = (mx.abs(actual - expected) / (mx.abs(expected) + 1e-3)).mean().item()
    assert rel_diff < 0.03, f"Group 150 SwiGLU rel diff exceeded threshold: {rel_diff}"


def test_swiglu_batched_input():
    """Verify fused SwiGLU correctly handles batched inputs of shape (batch, seq, K)."""
    mx.random.seed(42)
    B, S, K, N = 2, 4, 512, 1024
    G = 128

    x = mx.random.normal((B, S, K)).astype(mx.float16)
    w_g_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    w_u_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0

    w_g_packed, _ = pack_interleaved(w_g_raw, group_size=G)
    w_u_packed, _ = pack_interleaved(w_u_raw, group_size=G)

    expected = _compute_reference_swiglu(x, w_g_raw, w_u_raw, group_size=G)
    actual = ternary_swiglu(x, w_g_packed, w_u_packed, group_size=G)
    mx.eval(expected, actual)

    assert actual.shape == (B, S, N)
    rel_diff = (mx.abs(actual - expected) / (mx.abs(expected) + 1e-3)).mean().item()
    assert rel_diff < 0.03, f"Batched SwiGLU rel diff exceeded threshold: {rel_diff}"


def test_ternary_swiglu_module_initialization():
    """Verify high-level TernarySwiGLU nn.Module initialization and execution."""
    mx.random.seed(42)
    K, N = 256, 512
    G = 128

    w_g_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    w_u_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0

    layer = TernarySwiGLU(input_dims_or_gate=K, output_dims_or_up=N, group_size=G)
    layer.quantize_from_raw(w_g_raw, w_u_raw)

    x = mx.random.normal((1, K)).astype(mx.float16)
    out = layer(x)
    mx.eval(out)

    assert out.shape == (1, N)
    expected = _compute_reference_swiglu(x, w_g_raw, w_u_raw, group_size=G)
    rel_diff = (mx.abs(out - expected) / (mx.abs(expected) + 1e-3)).mean().item()
    assert rel_diff < 0.03, f"TernarySwiGLU module output diff exceeded threshold: {rel_diff}"


def test_ternary_swiglu_module_from_packed():
    """Verify TernarySwiGLU nn.Module initialized directly from pre-packed tensors."""
    mx.random.seed(42)
    K, N = 256, 512
    G = 128

    w_g_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    w_u_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0

    w_g_packed, _ = pack_interleaved(w_g_raw, group_size=G)
    w_u_packed, _ = pack_interleaved(w_u_raw, group_size=G)

    layer = TernarySwiGLU(w_g_packed, w_u_packed, group_size=G)
    x = mx.random.normal((1, K)).astype(mx.float16)
    out = layer(x)
    mx.eval(out)

    assert out.shape == (1, N)
