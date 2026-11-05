"""
Prototype: Universal Hadamard-Aware Fused Layers (Hadamard Activation Folding)
Target Models: Ternary-Bonsai-2-27B (PrismML 1.58-bit / 2-bit Checkpoints)
Objective: Evaluate Fast Walsh-Hadamard Transform (FWHT) activation rotation once per fused layer.

Prototypes:
1. HadamardTernaryLinear: Drop-in for TernaryLinear with optional FWHT.
2. HadamardTernarySwiGLU: Fused Gate-Up-SiLU evaluating FWHT exactly ONCE for Gate and Up.
3. HadamardTernaryQKV: Fused Attention QKV evaluating FWHT exactly ONCE for Q, K, and V.

Acceptance Gates:
- Parity vs Bonsai Packed layer on folded weights (mean_rel_diff < 0.01, max_rel_diff < 0.05).
- Parity vs un-rotated reference when block=0 and signs=None.
- Live latency profiling: 1 FWHT vs Bonsai multiple FWHT dispatches.
"""

import sys
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Setup import paths
CWD = Path.cwd()
SRC_CANDIDATES = [
    CWD / "src",
    CWD / "mlx-ternary" / "src",
    Path(__file__).resolve().parent.parent / "src",
    Path(__file__).resolve().parent.parent / "mlx-ternary" / "src",
]
for p in SRC_CANDIDATES:
    if p.is_dir():
        sys.path.insert(0, str(p))
        break

from mlx_ternary.ops import ternary_matmul, ternary_swiglu, ternary_fused_qkv
from mlx_ternary.quantize import pack_interleaved


def fwht(x: mx.array, block: int, signs: mx.array = None, inverse: bool = False) -> mx.array:
    """
    Fast Walsh-Hadamard Transform (FWHT) with activation signs.
    Uses MLX's native mx.hadamard_transform.
    """
    if block <= 0:
        return x
    shape, dtype = x.shape, x.dtype
    if shape[-1] % block != 0:
        raise ValueError(f"Hadamard block size {block} does not divide activation width {shape[-1]}")
    x_f32 = x.astype(mx.float32)
    if signs is not None and not inverse:
        x_f32 = x_f32 * signs
    x_rot = mx.hadamard_transform(x_f32.reshape(-1, block), scale=1.0 / math.sqrt(block)).reshape(shape)
    if signs is not None and inverse:
        x_rot = x_rot * signs
    return x_rot.astype(dtype)


class BonsaiPacked(nn.Module):
    """Reference implementation matching Bonsai runtime.py Packed layer."""
    def __init__(self, weight, block: int = 0, signs: mx.array = None):
        super().__init__()
        self.weight = weight
        self.block = block
        self.signs = signs

    def __call__(self, x: mx.array) -> mx.array:
        if self.block:
            x = fwht(x, self.block, self.signs)
        return ternary_matmul(x, self.weight, group_size=128)


class HadamardTernaryLinear(nn.Module):
    """Prototype TernaryLinear layer with optional Hadamard activation folding."""
    def __init__(
        self,
        input_dims_or_weight,
        output_dims: int = None,
        bias: bool = False,
        group_size: int = 128,
        pad_k: int = 0,
        block: int = 0,
        signs: mx.array = None,
    ):
        super().__init__()
        self.group_size = group_size
        self.pad_k = pad_k
        self.bias = None
        self.block = block
        self.signs = signs

        if isinstance(input_dims_or_weight, mx.array):
            self.weight_packed = input_dims_or_weight
            self.input_dims = None
            self.output_dims = self.weight_packed.shape[1]
        else:
            self.input_dims = input_dims_or_weight
            self.output_dims = output_dims
            self.pad_k = (group_size - (self.input_dims % group_size)) % group_size
            padded_k = self.input_dims + self.pad_k
            blocks = padded_k // group_size
            self.weight_packed = mx.zeros((blocks, output_dims, 32), dtype=mx.uint8)
            if bias:
                self.bias = mx.zeros((output_dims,), dtype=mx.float16)

    def __call__(self, x: mx.array) -> mx.array:
        if self.block:
            x = fwht(x, self.block, self.signs)
        if self.pad_k > 0:
            *batch, k = x.shape
            x = mx.concatenate([x, mx.zeros((*batch, self.pad_k), dtype=x.dtype)], axis=-1)
        out = ternary_matmul(x, self.weight_packed, group_size=self.group_size)
        if self.bias is not None:
            out = out + self.bias
        return out


class HadamardTernarySwiGLU(nn.Module):
    """
    Prototype TernarySwiGLU layer with Hadamard activation folding.
    Evaluates FWHT exactly ONCE for both Gate and Up projections.
    """
    def __init__(
        self,
        input_dims_or_gate,
        output_dims_or_up=None,
        output_dims: int = None,
        group_size: int = 128,
        pad_k: int = 0,
        block: int = 0,
        signs: mx.array = None,
    ):
        super().__init__()
        self.group_size = group_size
        self.pad_k = pad_k
        self.block = block
        self.signs = signs

        if isinstance(input_dims_or_gate, mx.array) and isinstance(output_dims_or_up, mx.array):
            self.gate_packed = input_dims_or_gate
            self.up_packed = output_dims_or_up
            self.input_dims = None
            self.output_dims = self.gate_packed.shape[1]
        else:
            self.input_dims = input_dims_or_gate
            self.output_dims = output_dims if output_dims is not None else output_dims_or_up
            self.pad_k = (group_size - (self.input_dims % group_size)) % group_size
            padded_k = self.input_dims + self.pad_k
            blocks = padded_k // group_size
            self.gate_packed = mx.zeros((blocks, self.output_dims, 32), dtype=mx.uint8)
            self.up_packed = mx.zeros((blocks, self.output_dims, 32), dtype=mx.uint8)

    def __call__(self, x: mx.array) -> mx.array:
        if self.block:
            x = fwht(x, self.block, self.signs)
        if self.pad_k > 0:
            *batch, k = x.shape
            x = mx.concatenate([x, mx.zeros((*batch, self.pad_k), dtype=x.dtype)], axis=-1)
        return ternary_swiglu(x, self.gate_packed, self.up_packed, group_size=self.group_size)


class HadamardTernaryQKV(nn.Module):
    """
    Prototype TernaryQKV layer with Hadamard activation folding.
    Evaluates FWHT exactly ONCE for Query, Key, and Value projections.
    """
    def __init__(
        self,
        input_dims_or_qkv,
        q_dim: int = None,
        k_dim: int = None,
        v_dim: int = None,
        split: bool = True,
        group_size: int = 128,
        pad_k: int = 0,
        block: int = 0,
        signs: mx.array = None,
    ):
        super().__init__()
        self.group_size = group_size
        self.pad_k = pad_k
        self.split = split
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.block = block
        self.signs = signs

        if isinstance(input_dims_or_qkv, mx.array):
            self.qkv_packed = input_dims_or_qkv
            self.input_dims = None
            self.output_dims = self.qkv_packed.shape[1]
        else:
            self.input_dims = input_dims_or_qkv
            assert q_dim is not None and k_dim is not None and v_dim is not None
            self.output_dims = q_dim + k_dim + v_dim
            self.pad_k = (group_size - (self.input_dims % group_size)) % group_size
            padded_k = self.input_dims + self.pad_k
            blocks = padded_k // group_size
            self.qkv_packed = mx.zeros((blocks, self.output_dims, 32), dtype=mx.uint8)

    def __call__(self, x: mx.array):
        if self.block:
            x = fwht(x, self.block, self.signs)
        if self.pad_k > 0:
            *batch, k = x.shape
            x = mx.concatenate([x, mx.zeros((*batch, self.pad_k), dtype=x.dtype)], axis=-1)
        out = ternary_fused_qkv(x, self.qkv_packed, group_size=self.group_size)
        if self.split and self.q_dim is not None and self.k_dim is not None:
            q = out[..., :self.q_dim]
            k = out[..., self.q_dim:self.q_dim + self.k_dim]
            v = out[..., self.q_dim + self.k_dim:]
            return q, k, v
        return out


def find_checkpoint_path() -> Path:
    candidates = [
        Path("testing-artifacts/models/Ternary-Bonsai-2-27B-mlx-2bit/model_interleaved.safetensors"),
        Path("../testing-artifacts/models/Ternary-Bonsai-2-27B-mlx-2bit/model_interleaved.safetensors"),
        Path(__file__).resolve().parent.parent / "testing-artifacts/models/Ternary-Bonsai-2-27B-mlx-2bit/model_interleaved.safetensors",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError("Could not locate model_interleaved.safetensors in testing-artifacts.")


def main():
    print("=" * 80)
    print("PROTOTYPE: UNIVERSAL HADAMARD-AWARE FUSED LAYERS IN SCRATCH")
    print("=" * 80)

    checkpoint_path = find_checkpoint_path()
    print(f"Loading checkpoint weights from: {checkpoint_path}")
    weights = mx.load(str(checkpoint_path))

    # --------------------------------------------------------------------------
    # 1. Verification of Shared Signs Invariant
    # --------------------------------------------------------------------------
    print("\n[1/5] Verifying Shared Sign Invariants across Layers...")
    
    # MLP Layer 0 Gate and Up
    gate_signs_l0 = weights["language_model.model.layers.0.mlp.gate_proj.signs"]
    up_signs_l0 = weights["language_model.model.layers.0.mlp.up_proj.signs"]
    down_signs_l0 = weights["language_model.model.layers.0.mlp.down_proj.signs"]

    assert mx.array_equal(gate_signs_l0, up_signs_l0).item(), "Gate and Up signs must match!"
    print(f"  ✓ Layer 0 Gate and Up signs match exactly: width = {gate_signs_l0.shape[0]}")
    print(f"  ✓ Layer 0 Down projection signs width = {down_signs_l0.shape[0]}")

    # Attention Layer 11 Q, K, V
    q_signs_l11 = weights["language_model.model.layers.11.self_attn.q_proj.signs"]
    k_signs_l11 = weights["language_model.model.layers.11.self_attn.k_proj.signs"]
    v_signs_l11 = weights["language_model.model.layers.11.self_attn.v_proj.signs"]

    assert mx.array_equal(q_signs_l11, k_signs_l11).item(), "Q and K signs must match!"
    assert mx.array_equal(q_signs_l11, v_signs_l11).item(), "Q and V signs must match!"
    print(f"  ✓ Layer 11 Q, K, V signs match exactly: width = {q_signs_l11.shape[0]} (D_model)")

    # --------------------------------------------------------------------------
    # 2. Numerical Parity: TernaryLinear with Hadamard (block=1024) vs Bonsai
    # --------------------------------------------------------------------------
    print("\n[2/5] Testing Numerical Parity: HadamardTernaryLinear vs Bonsai Packed...")
    gate_w = weights["language_model.model.layers.0.mlp.gate_proj.weight_packed"]
    down_w = weights["language_model.model.layers.0.mlp.down_proj.weight_packed"]

    mx.random.seed(42)
    x_gate = mx.random.normal((1, 5120)).astype(mx.float16)
    x_down = mx.random.normal((1, 17408)).astype(mx.float16)

    # Gate projection (K=5120 -> N=17408)
    bonsai_linear_gate = BonsaiPacked(gate_w, block=1024, signs=gate_signs_l0)
    hadamard_linear_gate = HadamardTernaryLinear(gate_w, block=1024, signs=gate_signs_l0)
    exp_gate = bonsai_linear_gate(x_gate)
    act_gate = hadamard_linear_gate(x_gate)
    mx.eval(exp_gate, act_gate)
    diff_gate = (mx.abs(act_gate - exp_gate) / (mx.abs(exp_gate) + 1e-3)).mean().item()
    max_rel_gate = (mx.abs(act_gate - exp_gate).max() / mx.abs(exp_gate.astype(mx.float32)).mean()).item()
    print(f"  ✓ Gate Linear (K=5120): mean_rel_diff = {diff_gate:.8f}, max_rel_diff = {max_rel_gate:.8f} (PASS: {diff_gate < 0.01 and max_rel_gate < 0.05})")
    assert diff_gate < 0.01 and max_rel_gate < 0.05

    # Down projection (K=17408 -> N=5120)
    bonsai_linear_down = BonsaiPacked(down_w, block=1024, signs=down_signs_l0)
    hadamard_linear_down = HadamardTernaryLinear(down_w, block=1024, signs=down_signs_l0)
    exp_down = bonsai_linear_down(x_down)
    act_down = hadamard_linear_down(x_down)
    mx.eval(exp_down, act_down)
    diff_down = (mx.abs(act_down - exp_down) / (mx.abs(exp_down) + 1e-3)).mean().item()
    max_rel_down = (mx.abs(act_down - exp_down).max() / mx.abs(exp_down.astype(mx.float32)).mean()).item()
    print(f"  ✓ Down Linear (K=17408): mean_rel_diff = {diff_down:.8f}, max_rel_diff = {max_rel_down:.8f} (PASS: {diff_down < 0.01 and max_rel_down < 0.05})")
    assert diff_down < 0.01 and max_rel_down < 0.05

    # --------------------------------------------------------------------------
    # 3. Numerical Parity: HadamardTernarySwiGLU vs Bonsai
    # --------------------------------------------------------------------------
    print("\n[3/5] Testing Numerical Parity: HadamardTernarySwiGLU (1 FWHT) vs Bonsai (2 FWHTs)...")
    up_w = weights["language_model.model.layers.0.mlp.up_proj.weight_packed"]

    bonsai_up = BonsaiPacked(up_w, block=1024, signs=gate_signs_l0)
    g_exp = bonsai_linear_gate(x_gate) # FWHT 1
    u_exp = bonsai_up(x_gate)          # FWHT 2
    swiglu_exp = (nn.silu(g_exp.astype(mx.float32)) * u_exp.astype(mx.float32)).astype(mx.float16)

    hadamard_swiglu = HadamardTernarySwiGLU(gate_w, up_w, block=1024, signs=gate_signs_l0)
    swiglu_act = hadamard_swiglu(x_gate) # FWHT exactly once
    mx.eval(swiglu_exp, swiglu_act)

    diff_swiglu = (mx.abs(swiglu_act - swiglu_exp) / (mx.abs(swiglu_exp) + 1e-3)).mean().item()
    max_rel_swiglu = (mx.abs(swiglu_act - swiglu_exp).max() / mx.abs(swiglu_exp.astype(mx.float32)).mean()).item()
    print(f"  ✓ SwiGLU: mean_rel_diff = {diff_swiglu:.8f}, max_rel_diff = {max_rel_swiglu:.8f} (PASS: {diff_swiglu < 0.01 and max_rel_swiglu < 0.05})")
    assert diff_swiglu < 0.01 and max_rel_swiglu < 0.05

    # --------------------------------------------------------------------------
    # 4. Numerical Parity: HadamardTernaryQKV vs Bonsai
    # --------------------------------------------------------------------------
    print("\n[4/5] Testing Numerical Parity: HadamardTernaryQKV (1 FWHT) vs Bonsai (3 FWHTs)...")
    qw = weights["language_model.model.layers.11.self_attn.q_proj.weight_packed"]
    kw = weights["language_model.model.layers.11.self_attn.k_proj.weight_packed"]
    vw = weights["language_model.model.layers.11.self_attn.v_proj.weight_packed"]

    q_dim, k_dim, v_dim = qw.shape[1], kw.shape[1], vw.shape[1]
    qkv_w = mx.concatenate([qw, kw, vw], axis=1)

    bonsai_q = BonsaiPacked(qw, block=1024, signs=q_signs_l11) # FWHT 1
    bonsai_k = BonsaiPacked(kw, block=1024, signs=q_signs_l11) # FWHT 2
    bonsai_v = BonsaiPacked(vw, block=1024, signs=q_signs_l11) # FWHT 3

    q_exp = bonsai_q(x_gate)
    k_exp = bonsai_k(x_gate)
    v_exp = bonsai_v(x_gate)

    hadamard_qkv = HadamardTernaryQKV(
        qkv_w,
        q_dim=q_dim,
        k_dim=k_dim,
        v_dim=v_dim,
        split=True,
        block=1024,
        signs=q_signs_l11,
    )
    q_act, k_act, v_act = hadamard_qkv(x_gate) # FWHT exactly once
    mx.eval(q_exp, k_exp, v_exp, q_act, k_act, v_act)

    diff_q = (mx.abs(q_act - q_exp) / (mx.abs(q_exp) + 1e-3)).mean().item()
    diff_k = (mx.abs(k_act - k_exp) / (mx.abs(k_exp) + 1e-3)).mean().item()
    diff_v = (mx.abs(v_act - v_exp) / (mx.abs(v_exp) + 1e-3)).mean().item()
    max_rel_q = (mx.abs(q_act - q_exp).max() / mx.abs(q_exp.astype(mx.float32)).mean()).item()
    max_rel_k = (mx.abs(k_act - k_exp).max() / mx.abs(k_exp.astype(mx.float32)).mean()).item()
    max_rel_v = (mx.abs(v_act - v_exp).max() / mx.abs(v_exp.astype(mx.float32)).mean()).item()

    print(f"  ✓ Q Projection (N=12288): mean_rel_diff = {diff_q:.8f}, max_rel_diff = {max_rel_q:.8f}")
    print(f"  ✓ K Projection (N=1024):  mean_rel_diff = {diff_k:.8f}, max_rel_diff = {max_rel_k:.8f}")
    print(f"  ✓ V Projection (N=1024):  mean_rel_diff = {diff_v:.8f}, max_rel_diff = {max_rel_v:.8f}")
    assert diff_q < 0.01 and diff_k < 0.01 and diff_v < 0.01
    assert max_rel_q < 0.05 and max_rel_k < 0.05 and max_rel_v < 0.05

    # --------------------------------------------------------------------------
    # 5. Zero-Overhead Un-Rotated Reference Check (block=0, signs=None)
    # --------------------------------------------------------------------------
    print("\n[5/5] Testing Zero-Overhead Un-Rotated Invariant (block=0, signs=None)...")
    lin_unrot = HadamardTernaryLinear(gate_w, block=0, signs=None)
    swiglu_unrot = HadamardTernarySwiGLU(gate_w, up_w, block=0, signs=None)
    qkv_unrot = HadamardTernaryQKV(qkv_w, q_dim=q_dim, k_dim=k_dim, v_dim=v_dim, split=True, block=0, signs=None)

    ref_lin = ternary_matmul(x_gate, gate_w, group_size=128)
    ref_swiglu = ternary_swiglu(x_gate, gate_w, up_w, group_size=128)
    ref_qkv_total = ternary_fused_qkv(x_gate, qkv_w, group_size=128)
    ref_q = ref_qkv_total[:, :q_dim]
    ref_k = ref_qkv_total[:, q_dim:q_dim + k_dim]
    ref_v = ref_qkv_total[:, q_dim + k_dim:]

    out_lin_unrot = lin_unrot(x_gate)
    out_swiglu_unrot = swiglu_unrot(x_gate)
    out_q, out_k, out_v = qkv_unrot(x_gate)

    mx.eval(out_lin_unrot, out_swiglu_unrot, out_q, out_k, out_v, ref_lin, ref_swiglu, ref_q, ref_k, ref_v)
    assert mx.array_equal(out_lin_unrot, ref_lin).item(), "Linear un-rotated mismatch!"
    assert mx.array_equal(out_swiglu_unrot, ref_swiglu).item(), "SwiGLU un-rotated mismatch!"
    assert mx.array_equal(out_q, ref_q).item(), "Q un-rotated mismatch!"
    assert mx.array_equal(out_k, ref_k).item(), "K un-rotated mismatch!"
    assert mx.array_equal(out_v, ref_v).item(), "V un-rotated mismatch!"
    print("  ✓ All block=0 un-rotated outputs match reference kernels with bit-exact equality (zero overhead)!")

    # --------------------------------------------------------------------------
    # Latency Profiling: Fused FWHT vs Bonsai Multiple FWHT Calls
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LIVE LATENCY PROFILING (1 FWHT Fused vs Bonsai Multiple FWHT)")
    print("=" * 80)

    warmup = 15
    iters = 50

    # 1. SwiGLU Latency
    for _ in range(warmup):
        g = bonsai_linear_gate(x_gate)
        u = bonsai_up(x_gate)
        mx.eval((nn.silu(g.astype(mx.float32)) * u.astype(mx.float32)).astype(mx.float16))

    t0 = time.perf_counter()
    for _ in range(iters):
        g = bonsai_linear_gate(x_gate)
        u = bonsai_up(x_gate)
        mx.eval((nn.silu(g.astype(mx.float32)) * u.astype(mx.float32)).astype(mx.float16))
    lat_bonsai_swiglu = (time.perf_counter() - t0) / iters * 1e6

    for _ in range(warmup):
        mx.eval(hadamard_swiglu(x_gate))

    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(hadamard_swiglu(x_gate))
    lat_fused_swiglu = (time.perf_counter() - t0) / iters * 1e6

    savings_swiglu = lat_bonsai_swiglu - lat_fused_swiglu
    speedup_swiglu = lat_bonsai_swiglu / lat_fused_swiglu

    print(f"MLP SwiGLU (5120 -> 17408):")
    print(f"  • Bonsai (2 FWHTs + 2 GEMVs + SiLU):  {lat_bonsai_swiglu:6.2f} µs")
    print(f"  • Fused (1 FWHT + Fused SwiGLU):      {lat_fused_swiglu:6.2f} µs")
    print(f"  • Savings per layer:                  {savings_swiglu:6.2f} µs (Speedup: {speedup_swiglu:.2f}x)")

    # 2. QKV Latency
    for _ in range(warmup):
        mx.eval(bonsai_q(x_gate), bonsai_k(x_gate), bonsai_v(x_gate))

    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(bonsai_q(x_gate), bonsai_k(x_gate), bonsai_v(x_gate))
    lat_bonsai_qkv = (time.perf_counter() - t0) / iters * 1e6

    for _ in range(warmup):
        mx.eval(*hadamard_qkv(x_gate))

    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(*hadamard_qkv(x_gate))
    lat_fused_qkv = (time.perf_counter() - t0) / iters * 1e6

    savings_qkv = lat_bonsai_qkv - lat_fused_qkv
    speedup_qkv = lat_bonsai_qkv / lat_fused_qkv

    print(f"\nAttention QKV (5120 -> 12288 + 1024 + 1024 = 14336):")
    print(f"  • Bonsai (3 FWHTs + 3 GEMVs):         {lat_bonsai_qkv:6.2f} µs")
    print(f"  • Fused (1 FWHT + Fused QKV):         {lat_fused_qkv:6.2f} µs")
    print(f"  • Savings per layer:                  {savings_qkv:6.2f} µs (Speedup: {speedup_qkv:.2f}x)")

    total_layer_savings = savings_swiglu + savings_qkv
    print("\n" + "=" * 80)
    print(f"TOTAL PRE-ATTENTION + PRE-MLP SAVINGS PER LAYER: {total_layer_savings:6.2f} µs")
    print(f"PROJECTED 64-LAYER FULL MODEL SAVINGS PER TOKEN: {total_layer_savings * 64 / 1000:6.2f} ms/token")
    print("=" * 80)


if __name__ == "__main__":
    main()
