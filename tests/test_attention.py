import mlx.core as mx
import pytest

from mlx_ternary import ternary_fused_kv, ternary_fused_qkv, TernaryFusedKV, TernaryQKV
from mlx_ternary.quantize import pack_interleaved


def _compute_reference_matmul(x: mx.array, w_raw: mx.array, group_size: int) -> mx.array:
    """Compute mathematical reference matmul using exact unpacked ternary representation."""
    K, N = w_raw.shape
    K_blocks = K // group_size
    w_grouped = w_raw.reshape(K_blocks, group_size, N)
    scales = mx.abs(w_grouped).max(axis=1).astype(mx.float16)
    safe_scales = mx.where(scales == 0, 1.0, scales)
    w_norm = w_grouped / safe_scales[:, None, :]
    w_ternary = mx.where(w_norm > 0.33, 1, mx.where(w_norm < -0.33, -1, 0))
    w_reconstructed = (w_ternary * scales[:, None, :]).reshape(K, N)
    expected = mx.matmul(x, w_reconstructed)
    mx.eval(expected)
    return expected


def test_fused_kv_parity_group128():
    """Verify numerical parity of dual-accumulator fused KV against reference for group_size=128."""
    mx.random.seed(42)
    M, K, N_kv = 1, 512, 256
    G = 128

    x = mx.random.normal((M, K)).astype(mx.float16)
    w_k_raw = mx.random.randint(0, 3, (K, N_kv)).astype(mx.float16) - 1.0
    w_v_raw = mx.random.randint(0, 3, (K, N_kv)).astype(mx.float16) - 1.0

    w_k_packed, _ = pack_interleaved(w_k_raw, group_size=G)
    w_v_packed, _ = pack_interleaved(w_v_raw, group_size=G)

    k_exp = _compute_reference_matmul(x, w_k_raw, group_size=G)
    v_exp = _compute_reference_matmul(x, w_v_raw, group_size=G)

    k_act, v_act = ternary_fused_kv(x, w_k_packed, w_v_packed, group_size=G)
    mx.eval(k_exp, v_exp, k_act, v_act)

    assert k_act.shape == k_exp.shape
    assert v_act.shape == v_exp.shape

    k_diff = (mx.abs(k_act - k_exp) / (mx.abs(k_exp) + 1e-3)).mean().item()
    v_diff = (mx.abs(v_act - v_exp) / (mx.abs(v_exp) + 1e-3)).mean().item()

    assert k_diff < 0.03, f"Group 128 Fused K rel diff exceeded threshold: {k_diff}"
    assert v_diff < 0.03, f"Group 128 Fused V rel diff exceeded threshold: {v_diff}"


def test_fused_kv_parity_group150():
    """Verify numerical parity of dual-accumulator fused KV against reference for group_size=150."""
    mx.random.seed(42)
    M, K, N_kv = 1, 600, 256
    G = 150

    x = mx.random.normal((M, K)).astype(mx.float16)
    w_k_raw = mx.random.randint(0, 3, (K, N_kv)).astype(mx.float16) - 1.0
    w_v_raw = mx.random.randint(0, 3, (K, N_kv)).astype(mx.float16) - 1.0

    w_k_packed, _ = pack_interleaved(w_k_raw, group_size=G)
    w_v_packed, _ = pack_interleaved(w_v_raw, group_size=G)

    k_exp = _compute_reference_matmul(x, w_k_raw, group_size=G)
    v_exp = _compute_reference_matmul(x, w_v_raw, group_size=G)

    k_act, v_act = ternary_fused_kv(x, w_k_packed, w_v_packed, group_size=G)
    mx.eval(k_exp, v_exp, k_act, v_act)

    assert k_act.shape == k_exp.shape
    assert v_act.shape == v_exp.shape

    k_diff = (mx.abs(k_act - k_exp) / (mx.abs(k_exp) + 1e-3)).mean().item()
    v_diff = (mx.abs(v_act - v_exp) / (mx.abs(v_exp) + 1e-3)).mean().item()

    assert k_diff < 0.03, f"Group 150 Fused K rel diff exceeded threshold: {k_diff}"
    assert v_diff < 0.03, f"Group 150 Fused V rel diff exceeded threshold: {v_diff}"


def test_fused_qkv_parity_group128():
    """Verify numerical parity of unified QKV against reference for group_size=128."""
    mx.random.seed(42)
    M, K = 1, 512
    N_q, N_k, N_v = 768, 128, 128
    N_total = N_q + N_k + N_v
    G = 128

    x = mx.random.normal((M, K)).astype(mx.float16)
    w_q_raw = mx.random.randint(0, 3, (K, N_q)).astype(mx.float16) - 1.0
    w_k_raw = mx.random.randint(0, 3, (K, N_k)).astype(mx.float16) - 1.0
    w_v_raw = mx.random.randint(0, 3, (K, N_v)).astype(mx.float16) - 1.0
    w_qkv_raw = mx.concatenate([w_q_raw, w_k_raw, w_v_raw], axis=1)

    w_qkv_packed, _ = pack_interleaved(w_qkv_raw, group_size=G)

    q_exp = _compute_reference_matmul(x, w_q_raw, group_size=G)
    k_exp = _compute_reference_matmul(x, w_k_raw, group_size=G)
    v_exp = _compute_reference_matmul(x, w_v_raw, group_size=G)
    qkv_exp = mx.concatenate([q_exp, k_exp, v_exp], axis=1)

    qkv_act = ternary_fused_qkv(x, w_qkv_packed, group_size=G)
    mx.eval(qkv_exp, qkv_act)

    assert qkv_act.shape == (M, N_total)
    rel_diff = (mx.abs(qkv_act - qkv_exp) / (mx.abs(qkv_exp) + 1e-3)).mean().item()
    assert rel_diff < 0.03, f"Group 128 QKV rel diff exceeded threshold: {rel_diff}"


def test_fused_qkv_parity_group150():
    """Verify numerical parity of unified QKV against reference for group_size=150."""
    mx.random.seed(42)
    M, K = 1, 600
    N_q, N_k, N_v = 768, 128, 128
    N_total = N_q + N_k + N_v
    G = 150

    x = mx.random.normal((M, K)).astype(mx.float16)
    w_q_raw = mx.random.randint(0, 3, (K, N_q)).astype(mx.float16) - 1.0
    w_k_raw = mx.random.randint(0, 3, (K, N_k)).astype(mx.float16) - 1.0
    w_v_raw = mx.random.randint(0, 3, (K, N_v)).astype(mx.float16) - 1.0
    w_qkv_raw = mx.concatenate([w_q_raw, w_k_raw, w_v_raw], axis=1)

    w_qkv_packed, _ = pack_interleaved(w_qkv_raw, group_size=G)

    q_exp = _compute_reference_matmul(x, w_q_raw, group_size=G)
    k_exp = _compute_reference_matmul(x, w_k_raw, group_size=G)
    v_exp = _compute_reference_matmul(x, w_v_raw, group_size=G)
    qkv_exp = mx.concatenate([q_exp, k_exp, v_exp], axis=1)

    qkv_act = ternary_fused_qkv(x, w_qkv_packed, group_size=G)
    mx.eval(qkv_exp, qkv_act)

    assert qkv_act.shape == (M, N_total)
    rel_diff = (mx.abs(qkv_act - qkv_exp) / (mx.abs(qkv_exp) + 1e-3)).mean().item()
    assert rel_diff < 0.03, f"Group 150 QKV rel diff exceeded threshold: {rel_diff}"


def test_attention_layers_and_shapes():
    """Verify TernaryFusedKV and TernaryQKV layers across 1D, 2D, and 3D shapes and .from_raw()."""
    mx.random.seed(42)
    K = 256
    N_q, N_k, N_v = 384, 128, 128
    G = 128

    w_q_raw = mx.random.randint(0, 3, (K, N_q)).astype(mx.float16) - 1.0
    w_k_raw = mx.random.randint(0, 3, (K, N_k)).astype(mx.float16) - 1.0
    w_v_raw = mx.random.randint(0, 3, (K, N_v)).astype(mx.float16) - 1.0

    # 1. Test TernaryFusedKV .from_raw()
    kv_layer = TernaryFusedKV.from_raw(w_k_raw, w_v_raw, group_size=G)

    # 1D input (K,)
    x_1d = mx.random.normal((K,)).astype(mx.float16)
    k_1d, v_1d = kv_layer(x_1d)
    mx.eval(k_1d, v_1d)
    assert k_1d.shape == (N_k,)
    assert v_1d.shape == (N_v,)

    # 2D input (M, K)
    x_2d = mx.random.normal((3, K)).astype(mx.float16)
    k_2d, v_2d = kv_layer(x_2d)
    mx.eval(k_2d, v_2d)
    assert k_2d.shape == (3, N_k)
    assert v_2d.shape == (3, N_v)

    # 3D input (B, S, K)
    x_3d = mx.random.normal((2, 4, K)).astype(mx.float16)
    k_3d, v_3d = kv_layer(x_3d)
    mx.eval(k_3d, v_3d)
    assert k_3d.shape == (2, 4, N_k)
    assert v_3d.shape == (2, 4, N_v)

    # 2. Test TernaryQKV .from_raw(split=True)
    qkv_layer = TernaryQKV.from_raw(w_q_raw, w_k_raw, w_v_raw, split=True, group_size=G)
    q_out, k_out, v_out = qkv_layer(x_3d)
    mx.eval(q_out, k_out, v_out)
    assert q_out.shape == (2, 4, N_q)
    assert k_out.shape == (2, 4, N_k)
    assert v_out.shape == (2, 4, N_v)

    # 3. Test TernaryQKV .from_raw(split=False)
    qkv_unsplit = TernaryQKV.from_raw(w_q_raw, w_k_raw, w_v_raw, split=False, group_size=G)
    qkv_total = qkv_unsplit(x_2d)
    mx.eval(qkv_total)
    assert qkv_total.shape == (3, N_q + N_k + N_v)
