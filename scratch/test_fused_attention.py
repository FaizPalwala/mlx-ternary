"""
Phase 3D Prototype: Attention Projection Fusion (Fused KV & Unified QKV)
Apple Silicon Architecture: Metal Shading Language with 0-Barrier SIMD GEMV

Workload (Ternary-Bonsai-2-27B Attention):
  - Q:   5120 -> 3072
  - K:   5120 -> 512
  - V:   5120 -> 512
  - QKV: 5120 -> 4096 (Unified Contiguous Projection)

Architectural Implementations:
1. Dual-Accumulator Fused KV:
   - Loads activation chunk once in registers, executes concurrent dual-issue FMAs into acc_k and acc_v.
   - Outputs out_k: (1, 512) and out_v: (1, 512).
2. Unified Fused QKV:
   - Packed weights: [w_q, w_k, w_v] along column axis -> (40, 4096, 32).
   - Single-pass 0-barrier SIMD-warp grid: (4096, 1, 1), threadgroup: (64, 1, 1).
   - Zero-copy tensor slicing into Q, K, V.
"""

import sys
import os
import time
import subprocess
from pathlib import Path

import mlx.core as mx
import mlx.core.fast as fast
import mlx.nn as nn

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "src"))

from mlx_ternary.ops import _LUT_HEADER, ternary_matmul
from mlx_ternary.quantize import pack_interleaved


def get_gpu_core_count() -> int:
    """Dynamically query GPU core count on Apple Silicon."""
    try:
        out = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], timeout=3).decode()
        for line in out.splitlines():
            if "Total Number of Cores" in line:
                return int(line.split(":")[-1].strip())
    except Exception:
        pass
    dev_name = mx.device_info().get("device_name", "")
    if "Max" in dev_name:
        return 40
    if "Ultra" in dev_name:
        return 80
    if "Pro" in dev_name:
        return 20
    return 10


def make_fused_kv_source(M: int, K: int, N: int, G_SIZE: int = 128) -> str:
    """
    Fused K and V projection Metal kernel.
    - Grid: (N, 1, M) where N = 512
    - Threadgroup: (64, 1, 1) or (128, 1, 1)
    - Reuses activation chunks loaded once in registers across dual-issue accumulators.
    - Outputs: out_k and out_v
    """
    source = f"""
    uint col = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;

    if (col >= {N} || row >= {M}) return;

    uint K_blocks = {K} / {G_SIZE};
    half acc_k = 0;
    half acc_v = 0;

    device const half* act_block = activations + row * {K};
    device const uint4* w_k_ptr = ((device const uint4*)k_weights) + col * 2;
    device const uint4* w_v_ptr = ((device const uint4*)v_weights) + col * 2;
    uint w_stride = {N} * 2;

    // Asynchronous prologue: pre-load block 0 weights for both K and V
    uint4 cur_k0 = w_k_ptr[0];
    uint4 cur_k1 = w_k_ptr[1];
    w_k_ptr += w_stride;

    uint4 cur_v0 = w_v_ptr[0];
    uint4 cur_v1 = w_v_ptr[1];
    w_v_ptr += w_stride;

    for (uint k_block = 0; k_block < K_blocks; k_block++) {{
        // Prefetch next block weights
        uint4 next_k0 = uint4(0), next_k1 = uint4(0);
        uint4 next_v0 = uint4(0), next_v1 = uint4(0);
        if (k_block + 1 < K_blocks) {{
            next_k0 = w_k_ptr[0];
            next_k1 = w_k_ptr[1];
            w_k_ptr += w_stride;

            next_v0 = w_v_ptr[0];
            next_v1 = w_v_ptr[1];
            w_v_ptr += w_stride;
        }}

        half scale_k = as_type<half>((uint16_t)(cur_k0.x & 0xFFFF));
        half scale_v = as_type<half>((uint16_t)(cur_v0.x & 0xFFFF));

        struct PackedBlock {{ uint4 v0, v1; }};
        PackedBlock blk_k = {{cur_k0, cur_k1}};
        PackedBlock blk_v = {{cur_v0, cur_v1}};
        thread const uchar* bytes_k = (thread const uchar*)&blk_k;
        thread const uchar* bytes_v = (thread const uchar*)&blk_v;
    """

    if G_SIZE == 128:
        source += f"""
        half k_acc0 = 0, k_acc1 = 0, k_acc2 = 0, k_acc3 = 0;
        half v_acc0 = 0, v_acc1 = 0, v_acc2 = 0, v_acc3 = 0;

        device const half4* act_vec = (device const half4*)act_block;

        #pragma unroll(6)
        for (uint c = 0; c < 6; c++) {{
            half4 x0 = act_vec[c * 5 + 0];
            half4 x1 = act_vec[c * 5 + 1];
            half4 x2 = act_vec[c * 5 + 2];
            half4 x3 = act_vec[c * 5 + 3];
            half4 x4 = act_vec[c * 5 + 4];

            // 1. K Chunk FMAs
            uint8_t kb0 = bytes_k[2 + c * 4 + 0];
            uint8_t kb1 = bytes_k[2 + c * 4 + 1];
            uint8_t kb2 = bytes_k[2 + c * 4 + 2];
            uint8_t kb3 = bytes_k[2 + c * 4 + 3];

            half4 kw0_low = base3_lut_low[kb0]; half kw0_high = base3_lut_high[kb0];
            half4 kw1_low = base3_lut_low[kb1]; half kw1_high = base3_lut_high[kb1];
            half4 kw2_low = base3_lut_low[kb2]; half kw2_high = base3_lut_high[kb2];
            half4 kw3_low = base3_lut_low[kb3]; half kw3_high = base3_lut_high[kb3];

            k_acc0 += dot(x0, kw0_low) + x1.x * kw0_high;
            k_acc1 += dot(x1.yzw, kw1_low.xyz) + x2.x * kw1_low.w + x2.y * kw1_high;
            k_acc2 += dot(x2.zw, kw2_low.xy) + dot(x3.xyz, half3(kw2_low.z, kw2_low.w, kw2_high));
            k_acc3 += x3.w * kw3_low.x + dot(x4, half4(kw3_low.y, kw3_low.z, kw3_low.w, kw3_high));

            // 2. V Chunk FMAs (reuses x0..x4, dual-issue concurrent with K)
            uint8_t vb0 = bytes_v[2 + c * 4 + 0];
            uint8_t vb1 = bytes_v[2 + c * 4 + 1];
            uint8_t vb2 = bytes_v[2 + c * 4 + 2];
            uint8_t vb3 = bytes_v[2 + c * 4 + 3];

            half4 vw0_low = base3_lut_low[vb0]; half vw0_high = base3_lut_high[vb0];
            half4 vw1_low = base3_lut_low[vb1]; half vw1_high = base3_lut_high[vb1];
            half4 vw2_low = base3_lut_low[vb2]; half vw2_high = base3_lut_high[vb2];
            half4 vw3_low = base3_lut_low[vb3]; half vw3_high = base3_lut_high[vb3];

            v_acc0 += dot(x0, vw0_low) + x1.x * vw0_high;
            v_acc1 += dot(x1.yzw, vw1_low.xyz) + x2.x * vw1_low.w + x2.y * vw1_high;
            v_acc2 += dot(x2.zw, vw2_low.xy) + dot(x3.xyz, half3(vw2_low.z, vw2_low.w, vw2_high));
            v_acc3 += x3.w * vw3_low.x + dot(x4, half4(vw3_low.y, vw3_low.z, vw3_low.w, vw3_high));
        }}

        half4 tail0 = act_vec[30];
        half4 tail1 = act_vec[31];

        // Tail K
        uint8_t kb24 = bytes_k[2 + 24];
        uint8_t kb25 = bytes_k[2 + 25];
        k_acc0 += dot(tail0, base3_lut_low[kb24]) + tail1.x * base3_lut_high[kb24];
        k_acc1 += dot(tail1.yzw, base3_lut_low[kb25].xyz);

        // Tail V
        uint8_t vb24 = bytes_v[2 + 24];
        uint8_t vb25 = bytes_v[2 + 25];
        v_acc0 += dot(tail0, base3_lut_low[vb24]) + tail1.x * base3_lut_high[vb24];
        v_acc1 += dot(tail1.yzw, base3_lut_low[vb25].xyz);

        acc_k += ((k_acc0 + k_acc1) + (k_acc2 + k_acc3)) * scale_k;
        acc_v += ((v_acc0 + v_acc1) + (v_acc2 + v_acc3)) * scale_v;

        act_block += 128;
        cur_k0 = next_k0; cur_k1 = next_k1;
        cur_v0 = next_v0; cur_v1 = next_v1;
        """
    else:
        # G_SIZE == 150
        source += f"""
        half block_acc_k = 0;
        half block_acc_v = 0;

        #pragma unroll(30)
        for (uint i = 0; i < 30; i++) {{
            uint curr_x = i * 5;
            half4 x_low = half4(act_block[curr_x], act_block[curr_x + 1], act_block[curr_x + 2], act_block[curr_x + 3]);
            half x_high = act_block[curr_x + 4];

            uint8_t kb = bytes_k[2 + i];
            half4 kw_low = base3_lut_low[kb]; half kw_high = base3_lut_high[kb];
            block_acc_k += dot(x_low, kw_low) + x_high * kw_high;

            uint8_t vb = bytes_v[2 + i];
            half4 vw_low = base3_lut_low[vb]; half vw_high = base3_lut_high[vb];
            block_acc_v += dot(x_low, vw_low) + x_high * vw_high;
        }}

        acc_k += block_acc_k * scale_k;
        acc_v += block_acc_v * scale_v;

        act_block += 150;
        cur_k0 = next_k0; cur_k1 = next_k1;
        cur_v0 = next_v0; cur_v1 = next_v1;
        """

    source += f"""
    }}

    out_k[row * {N} + col] = acc_k;
    out_v[row * {N} + col] = acc_v;
    """
    return source


def make_fused_qkv_source(M: int, K: int, N: int, G_SIZE: int = 128) -> str:
    """
    Unified Fused QKV 0-barrier SIMD GEMV source.
    - Grid: (N, 1, M) where N = 4096
    - Threadgroup: (64, 1, 1) or (128, 1, 1)
    - Pre-fetches weights, aligned 20-trit vector loads, 4-way ILP accumulators
    - Direct write to output[row * N + col]
    """
    source = f"""
    uint col = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;

    if (col >= {N} || row >= {M}) return;

    uint K_blocks = {K} / {G_SIZE};
    half acc = 0;

    device const half* act_block = activations + row * {K};
    device const uint4* w_block = ((device const uint4*)interleaved_weights) + col * 2;
    uint w_stride = {N} * 2;

    uint4 cur_vec0 = w_block[0];
    uint4 cur_vec1 = w_block[1];
    w_block += w_stride;

    for (uint k_block = 0; k_block < K_blocks; k_block++) {{
        uint4 next_vec0 = uint4(0);
        uint4 next_vec1 = uint4(0);
        if (k_block + 1 < K_blocks) {{
            next_vec0 = w_block[0];
            next_vec1 = w_block[1];
            w_block += w_stride;
        }}

        half scale = as_type<half>((uint16_t)(cur_vec0.x & 0xFFFF));

        struct PackedBlock {{ uint4 v0, v1; }};
        PackedBlock blk = {{cur_vec0, cur_vec1}};
        thread const uchar* blk_bytes = (thread const uchar*)&blk;
    """

    if G_SIZE == 128:
        source += f"""
        half block_acc0 = 0, block_acc1 = 0, block_acc2 = 0, block_acc3 = 0;
        device const half4* act_vec = (device const half4*)act_block;

        #pragma unroll(6)
        for (uint c = 0; c < 6; c++) {{
            half4 x0 = act_vec[c * 5 + 0];
            half4 x1 = act_vec[c * 5 + 1];
            half4 x2 = act_vec[c * 5 + 2];
            half4 x3 = act_vec[c * 5 + 3];
            half4 x4 = act_vec[c * 5 + 4];

            uint8_t b0 = blk_bytes[2 + c * 4 + 0];
            uint8_t b1 = blk_bytes[2 + c * 4 + 1];
            uint8_t b2 = blk_bytes[2 + c * 4 + 2];
            uint8_t b3 = blk_bytes[2 + c * 4 + 3];

            half4 w0_low = base3_lut_low[b0]; half w0_high = base3_lut_high[b0];
            half4 w1_low = base3_lut_low[b1]; half w1_high = base3_lut_high[b1];
            half4 w2_low = base3_lut_low[b2]; half w2_high = base3_lut_high[b2];
            half4 w3_low = base3_lut_low[b3]; half w3_high = base3_lut_high[b3];

            block_acc0 += dot(x0, w0_low) + x1.x * w0_high;
            block_acc1 += dot(x1.yzw, w1_low.xyz) + x2.x * w1_low.w + x2.y * w1_high;
            block_acc2 += dot(x2.zw, w2_low.xy) + dot(x3.xyz, half3(w2_low.z, w2_low.w, w2_high));
            block_acc3 += x3.w * w3_low.x + dot(x4, half4(w3_low.y, w3_low.z, w3_low.w, w3_high));
        }}

        device const half4* tail_vec = (device const half4*)(act_block + 120);
        half4 tail0 = tail_vec[0];
        half4 tail1 = tail_vec[1];

        uint8_t b24 = blk_bytes[2 + 24];
        uint8_t b25 = blk_bytes[2 + 25];

        half4 w24_low = base3_lut_low[b24];
        half w24_high = base3_lut_high[b24];
        half4 w25_low = base3_lut_low[b25];

        block_acc0 += dot(tail0, w24_low) + tail1.x * w24_high;
        block_acc1 += dot(tail1.yzw, w25_low.xyz);

        half block_acc = (block_acc0 + block_acc1) + (block_acc2 + block_acc3);
        acc += block_acc * scale;
        act_block += 128;
        cur_vec0 = next_vec0;
        cur_vec1 = next_vec1;
        """
    else:
        source += f"""
        half block_acc = 0;

        #pragma unroll(30)
        for (uint i = 0; i < 30; i++) {{
            uint8_t b = blk_bytes[2 + i];
            half4 w_low = base3_lut_low[b];
            half w_high = base3_lut_high[b];
            uint curr_x = i * 5;
            half4 x_low = half4(act_block[curr_x], act_block[curr_x + 1], act_block[curr_x + 2], act_block[curr_x + 3]);
            half x_high = act_block[curr_x + 4];
            block_acc += dot(x_low, w_low) + x_high * w_high;
        }}
        acc += block_acc * scale;
        act_block += 150;
        cur_vec0 = next_vec0;
        cur_vec1 = next_vec1;
        """

    source += f"""
    }}

    output[row * {N} + col] = acc;
    """
    return source


def compute_unquantized_reference(x: mx.array, w_raw: mx.array, G: int) -> mx.array:
    """Exact reference unquantized projection."""
    K, N = w_raw.shape
    K_blocks = K // G
    w_grouped = w_raw.reshape(K_blocks, G, N)
    scales = mx.abs(w_grouped).max(axis=1).astype(mx.float16)
    safe_scales = mx.where(scales == 0, 1.0, scales)
    w_norm = w_grouped / safe_scales[:, None, :]
    w_ternary = mx.where(w_norm > 0.33, 1, mx.where(w_norm < -0.33, -1, 0))
    w_reconstructed = (w_ternary * scales[:, None, :]).reshape(K, N)
    expected = mx.matmul(x, w_reconstructed)
    mx.eval(expected)
    return expected


def run_benchmark():
    print("=" * 80)
    print("ATTENTION PROJECTION FUSION BENCHMARK: FUSED KV & UNIFIED QKV")
    print("=" * 80)

    cores = get_gpu_core_count()
    dev_name = mx.device_info().get("device_name", "Apple Silicon")
    print(f"Target Hardware:  {dev_name} ({cores} GPU cores detected)")

    M = 1
    K = 5120
    N_q = 3072
    N_k = 512
    N_v = 512
    N_qkv = N_q + N_k + N_v  # 4096
    G = 128

    print(f"Contractions:     Q ({K} -> {N_q}), K ({K} -> {N_k}), V ({K} -> {N_v}) | QKV: {K} -> {N_qkv}")

    # Generate random test weights
    mx.random.seed(42)
    x = mx.random.normal((M, K)).astype(mx.float16)

    w_q_raw = mx.random.randint(0, 3, (K, N_q)).astype(mx.float16) - 1.0
    w_k_raw = mx.random.randint(0, 3, (K, N_k)).astype(mx.float16) - 1.0
    w_v_raw = mx.random.randint(0, 3, (K, N_v)).astype(mx.float16) - 1.0

    w_q_packed, _ = pack_interleaved(w_q_raw, group_size=G)
    w_k_packed, _ = pack_interleaved(w_k_raw, group_size=G)
    w_v_packed, _ = pack_interleaved(w_v_raw, group_size=G)

    w_qkv_raw = mx.concatenate([w_q_raw, w_k_raw, w_v_raw], axis=1)
    w_qkv_packed = mx.concatenate([w_q_packed, w_k_packed, w_v_packed], axis=1)

    # 1. Compute references
    q_ref = compute_unquantized_reference(x, w_q_raw, G)
    k_ref = compute_unquantized_reference(x, w_k_raw, G)
    v_ref = compute_unquantized_reference(x, w_v_raw, G)
    qkv_ref = compute_unquantized_reference(x, w_qkv_raw, G)

    warmup = 25
    iters = 100

    results = {}

    # --------------------------------------------------------------------------
    # Benchmark 1: Live Bonsai 2-bit Baseline (QuantizedLinear Q, K, V)
    # --------------------------------------------------------------------------
    print("\n[1/4] Benchmarking Live PrismML Bonsai 2-bit Attention Projections...")
    w_q_fp16 = mx.random.normal((N_q, K)).astype(mx.float16)
    w_k_fp16 = mx.random.normal((N_k, K)).astype(mx.float16)
    w_v_fp16 = mx.random.normal((N_v, K)).astype(mx.float16)

    bq_w, bq_s, bq_b = mx.quantize(w_q_fp16, group_size=G, bits=2)
    bk_w, bk_s, bk_b = mx.quantize(w_k_fp16, group_size=G, bits=2)
    bv_w, bv_s, bv_b = mx.quantize(w_v_fp16, group_size=G, bits=2)

    bonsai_q = nn.QuantizedLinear(K, N_q, bits=2, group_size=G, bias=False)
    bonsai_q.weight, bonsai_q.scales, bonsai_q.biases = bq_w, bq_s, bq_b

    bonsai_k = nn.QuantizedLinear(K, N_k, bits=2, group_size=G, bias=False)
    bonsai_k.weight, bonsai_k.scales, bonsai_k.biases = bk_w, bk_s, bk_b

    bonsai_v = nn.QuantizedLinear(K, N_v, bits=2, group_size=G, bias=False)
    bonsai_v.weight, bonsai_v.scales, bonsai_v.biases = bv_w, bv_s, bv_b
    mx.eval(bonsai_q.parameters(), bonsai_k.parameters(), bonsai_v.parameters())

    for _ in range(warmup):
        mx.eval(bonsai_q(x), bonsai_k(x), bonsai_v(x))

    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(bonsai_q(x), bonsai_k(x), bonsai_v(x))
    lat_bonsai = (time.perf_counter() - t0) / iters * 1e6
    results["bonsai"] = {"lat": lat_bonsai}
    print(f"  ✓ Live Bonsai 2-bit Q+K+V:    {lat_bonsai:6.2f} µs")

    # --------------------------------------------------------------------------
    # Benchmark 2: Baseline Isolated Projections
    # --------------------------------------------------------------------------
    print("\n[2/4] Benchmarking Isolated ternary_matmul (Q, K, V)...")
    # Isolated dispatches alone
    for _ in range(warmup):
        mx.eval(ternary_matmul(x, w_k_packed, group_size=G))
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(ternary_matmul(x, w_k_packed, group_size=G))
    lat_k_alone = (time.perf_counter() - t0) / iters * 1e6

    for _ in range(warmup):
        mx.eval(ternary_matmul(x, w_v_packed, group_size=G))
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(ternary_matmul(x, w_v_packed, group_size=G))
    lat_v_alone = (time.perf_counter() - t0) / iters * 1e6

    for _ in range(warmup):
        mx.eval(ternary_matmul(x, w_q_packed, group_size=G))
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(ternary_matmul(x, w_q_packed, group_size=G))
    lat_q_alone = (time.perf_counter() - t0) / iters * 1e6

    lat_isolated_kv_sum = lat_k_alone + lat_v_alone
    lat_isolated_total_sum = lat_q_alone + lat_k_alone + lat_v_alone

    # Batched sequence execution
    for _ in range(warmup):
        q_out = ternary_matmul(x, w_q_packed, group_size=G)
        k_out = ternary_matmul(x, w_k_packed, group_size=G)
        v_out = ternary_matmul(x, w_v_packed, group_size=G)
        mx.eval(q_out, k_out, v_out)

    t0 = time.perf_counter()
    for _ in range(iters):
        q_out = ternary_matmul(x, w_q_packed, group_size=G)
        k_out = ternary_matmul(x, w_k_packed, group_size=G)
        v_out = ternary_matmul(x, w_v_packed, group_size=G)
        mx.eval(q_out, k_out, v_out)
    lat_isolated_seq = (time.perf_counter() - t0) / iters * 1e6

    print(f"  ✓ Isolated Individual Dispatches: Q={lat_q_alone:6.2f} µs, K={lat_k_alone:6.2f} µs, V={lat_v_alone:6.2f} µs")
    print(f"  ✓ Sum of Isolated Dispatches:     K+V={lat_isolated_kv_sum:6.2f} µs | Q+K+V={lat_isolated_total_sum:6.2f} µs")
    print(f"  ✓ Batched Sequence (Q + K + V):   {lat_isolated_seq:6.2f} µs")

    # --------------------------------------------------------------------------
    # Benchmark 3: Kernel A: Dual-Accumulator Fused KV
    # --------------------------------------------------------------------------
    print("\n[3/4] Benchmarking Kernel A: Dual-Accumulator Fused KV...")
    src_fused_kv = make_fused_kv_source(M, K, N_k, G)
    k_fused_kv = fast.metal_kernel(
        name="fused_kv",
        input_names=["activations", "k_weights", "v_weights"],
        output_names=["out_k", "out_v"],
        source=src_fused_kv,
        header=_LUT_HEADER
    )

    grid_kv = (N_k, 1, M)
    best_tg_kv = 128

    out_k, out_v = k_fused_kv(
        inputs=[x, w_k_packed, w_v_packed],
        grid=grid_kv,
        threadgroup=(best_tg_kv, 1, 1),
        output_shapes=[(M, N_k), (M, N_k)],
        output_dtypes=[mx.float16, mx.float16]
    )
    mx.eval(out_k, out_v)

    # Parity check
    k_diff = (mx.abs(out_k - k_ref) / (mx.abs(k_ref) + 1e-3)).mean().item()
    v_diff = (mx.abs(out_v - v_ref) / (mx.abs(v_ref) + 1e-3)).mean().item()
    k_max_rel = (mx.abs(out_k - k_ref).max() / mx.abs(k_ref.astype(mx.float32)).mean()).item()
    v_max_rel = (mx.abs(out_v - v_ref).max() / mx.abs(v_ref.astype(mx.float32)).mean()).item()

    for _ in range(warmup):
        k_out, v_out = k_fused_kv(inputs=[x, w_k_packed, w_v_packed], grid=grid_kv, threadgroup=(best_tg_kv, 1, 1), output_shapes=[(M, N_k), (M, N_k)], output_dtypes=[mx.float16, mx.float16])
        mx.eval(k_out, v_out)

    t0 = time.perf_counter()
    for _ in range(iters):
        k_out, v_out = k_fused_kv(inputs=[x, w_k_packed, w_v_packed], grid=grid_kv, threadgroup=(best_tg_kv, 1, 1), output_shapes=[(M, N_k), (M, N_k)], output_dtypes=[mx.float16, mx.float16])
        mx.eval(k_out, v_out)
    lat_fused_kv = (time.perf_counter() - t0) / iters * 1e6

    # Fused KV sequence: Q + FusedKV
    for _ in range(warmup):
        q_out = ternary_matmul(x, w_q_packed, group_size=G)
        k_out, v_out = k_fused_kv(inputs=[x, w_k_packed, w_v_packed], grid=grid_kv, threadgroup=(best_tg_kv, 1, 1), output_shapes=[(M, N_k), (M, N_k)], output_dtypes=[mx.float16, mx.float16])
        mx.eval(q_out, k_out, v_out)

    t0 = time.perf_counter()
    for _ in range(iters):
        q_out = ternary_matmul(x, w_q_packed, group_size=G)
        k_out, v_out = k_fused_kv(inputs=[x, w_k_packed, w_v_packed], grid=grid_kv, threadgroup=(best_tg_kv, 1, 1), output_shapes=[(M, N_k), (M, N_k)], output_dtypes=[mx.float16, mx.float16])
        mx.eval(q_out, k_out, v_out)
    lat_fused_kv_seq = (time.perf_counter() - t0) / iters * 1e6

    kv_savings_dispatch = lat_isolated_kv_sum - lat_fused_kv
    kv_savings_seq = lat_isolated_seq - lat_fused_kv_seq

    print(f"  ✓ Fused KV Kernel (Alone):     {lat_fused_kv:6.2f} µs | Savings vs K+V Dispatches: {kv_savings_dispatch:6.2f} µs")
    print(f"  ✓ Q + Fused KV Sequence:       {lat_fused_kv_seq:6.2f} µs | Savings vs Batched Q+K+V:    {kv_savings_seq:6.2f} µs")
    print(f"  ✓ Numerical Parity K: mean_rel = {k_diff:.6f}, max_rel = {k_max_rel:.6f}")
    print(f"  ✓ Numerical Parity V: mean_rel = {v_diff:.6f}, max_rel = {v_max_rel:.6f}")

    # --------------------------------------------------------------------------
    # Benchmark 4: Kernel B: Unified Fused QKV (Contiguous 4096-Column)
    # --------------------------------------------------------------------------
    print("\n[4/4] Benchmarking Kernel B: Unified Fused QKV (5120 -> 4096)...")
    src_fused_qkv = make_fused_qkv_source(M, K, N_qkv, G)
    k_fused_qkv = fast.metal_kernel(
        name="fused_qkv",
        input_names=["activations", "interleaved_weights"],
        output_names=["output"],
        source=src_fused_qkv,
        header=_LUT_HEADER
    )

    grid_qkv = (N_qkv, 1, M)
    best_tg_qkv = 64

    out_qkv = k_fused_qkv(
        inputs=[x, w_qkv_packed],
        grid=grid_qkv,
        threadgroup=(best_tg_qkv, 1, 1),
        output_shapes=[(M, N_qkv)],
        output_dtypes=[mx.float16]
    )[0]
    mx.eval(out_qkv)

    # Parity check on slices
    q_slice = out_qkv[:, :N_q]
    k_slice = out_qkv[:, N_q:N_q + N_k]
    v_slice = out_qkv[:, N_q + N_k:]

    qkv_diff = (mx.abs(out_qkv - qkv_ref) / (mx.abs(qkv_ref) + 1e-3)).mean().item()
    qkv_max_rel = (mx.abs(out_qkv - qkv_ref).max() / mx.abs(qkv_ref.astype(mx.float32)).mean()).item()

    for _ in range(warmup):
        out = k_fused_qkv(inputs=[x, w_qkv_packed], grid=grid_qkv, threadgroup=(best_tg_qkv, 1, 1), output_shapes=[(M, N_qkv)], output_dtypes=[mx.float16])[0]
        q_s, k_s, v_s = out[:, :N_q], out[:, N_q:N_q + N_k], out[:, N_q + N_k:]
        mx.eval(q_s, k_s, v_s)

    t0 = time.perf_counter()
    for _ in range(iters):
        out = k_fused_qkv(inputs=[x, w_qkv_packed], grid=grid_qkv, threadgroup=(best_tg_qkv, 1, 1), output_shapes=[(M, N_qkv)], output_dtypes=[mx.float16])[0]
        q_s, k_s, v_s = out[:, :N_q], out[:, N_q:N_q + N_k], out[:, N_q + N_k:]
        mx.eval(q_s, k_s, v_s)
    lat_fused_qkv = (time.perf_counter() - t0) / iters * 1e6
    bw_qkv = (K * N_qkv * 2 / 8) / (lat_fused_qkv * 1e-6) / 1e9

    qkv_savings_dispatch = lat_isolated_total_sum - lat_fused_qkv
    qkv_savings_seq = lat_isolated_seq - lat_fused_qkv

    print(f"  ✓ Unified Fused QKV:           {lat_fused_qkv:6.2f} µs | Bandwidth: {bw_qkv:5.1f} GB/s")
    print(f"  ✓ Savings vs Isolated Sum:     {qkv_savings_dispatch:6.2f} µs (Speedup: {lat_isolated_total_sum / lat_fused_qkv:.2f}x)")
    print(f"  ✓ Savings vs Batched Seq:      {qkv_savings_seq:6.2f} µs (Speedup: {lat_isolated_seq / lat_fused_qkv:.2f}x)")
    print(f"  ✓ Speedup vs Live Bonsai:      {lat_bonsai / lat_fused_qkv:.2f}x ({lat_fused_qkv:.2f} µs vs {lat_bonsai:.2f} µs)")
    print(f"  ✓ Numerical Parity QKV: mean_rel = {qkv_diff:.6f}, max_rel = {qkv_max_rel:.6f}")

    # --------------------------------------------------------------------------
    # Group Size 150 Validation
    # --------------------------------------------------------------------------
    print("\n[5/5] Validating Group Size 150 Numerical Precision...")
    K_150 = 5100  # Multiple of 150 (34 blocks)
    x_150 = mx.random.normal((M, K_150)).astype(mx.float16)

    w_k_150 = mx.random.randint(0, 3, (K_150, N_k)).astype(mx.float16) - 1.0
    w_v_150 = mx.random.randint(0, 3, (K_150, N_v)).astype(mx.float16) - 1.0
    w_qkv_150 = mx.random.randint(0, 3, (K_150, N_qkv)).astype(mx.float16) - 1.0

    pk_150, _ = pack_interleaved(w_k_150, group_size=150)
    pv_150, _ = pack_interleaved(w_v_150, group_size=150)
    pqkv_150, _ = pack_interleaved(w_qkv_150, group_size=150)

    k_ref_150 = compute_unquantized_reference(x_150, w_k_150, 150)
    v_ref_150 = compute_unquantized_reference(x_150, w_v_150, 150)
    qkv_ref_150 = compute_unquantized_reference(x_150, w_qkv_150, 150)

    # Fused KV (G=150)
    src_kv_150 = make_fused_kv_source(M, K_150, N_k, 150)
    k_kv_150 = fast.metal_kernel(
        name="fused_kv_150",
        input_names=["activations", "k_weights", "v_weights"],
        output_names=["out_k", "out_v"],
        source=src_kv_150,
        header=_LUT_HEADER
    )
    ok_150, ov_150 = k_kv_150(inputs=[x_150, pk_150, pv_150], grid=(N_k, 1, M), threadgroup=(best_tg_kv, 1, 1), output_shapes=[(M, N_k), (M, N_k)], output_dtypes=[mx.float16, mx.float16])
    mx.eval(ok_150, ov_150)

    diff_k_150 = (mx.abs(ok_150 - k_ref_150) / (mx.abs(k_ref_150) + 1e-3)).mean().item()
    diff_v_150 = (mx.abs(ov_150 - v_ref_150) / (mx.abs(v_ref_150) + 1e-3)).mean().item()
    max_k_150 = (mx.abs(ok_150 - k_ref_150).max() / mx.abs(k_ref_150.astype(mx.float32)).mean()).item()
    max_v_150 = (mx.abs(ov_150 - v_ref_150).max() / mx.abs(v_ref_150.astype(mx.float32)).mean()).item()

    # Fused QKV (G=150)
    src_qkv_150 = make_fused_qkv_source(M, K_150, N_qkv, 150)
    k_qkv_150 = fast.metal_kernel(
        name="fused_qkv_150",
        input_names=["activations", "interleaved_weights"],
        output_names=["output"],
        source=src_qkv_150,
        header=_LUT_HEADER
    )
    oqkv_150 = k_qkv_150(inputs=[x_150, pqkv_150], grid=(N_qkv, 1, M), threadgroup=(best_tg_qkv, 1, 1), output_shapes=[(M, N_qkv)], output_dtypes=[mx.float16])[0]
    mx.eval(oqkv_150)

    diff_qkv_150 = (mx.abs(oqkv_150 - qkv_ref_150) / (mx.abs(qkv_ref_150) + 1e-3)).mean().item()
    max_qkv_150 = (mx.abs(oqkv_150 - qkv_ref_150).max() / mx.abs(qkv_ref_150.astype(mx.float32)).mean()).item()

    print(f"  ✓ G=150 Fused KV:  K mean_rel = {diff_k_150:.6f}, max_rel = {max_k_150:.6f} | V mean_rel = {diff_v_150:.6f}, max_rel = {max_v_150:.6f}")
    print(f"  ✓ G=150 Fused QKV: QKV mean_rel = {diff_qkv_150:.6f}, max_rel = {max_qkv_150:.6f}")

    # --------------------------------------------------------------------------
    # Acceptance Gates
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ACCEPTANCE GATES VERIFICATION")
    print("=" * 80)

    # Gate 1: Numerical Precision
    gate1_pass = (
        k_diff < 0.03 and k_max_rel < 0.15 and
        v_diff < 0.03 and v_max_rel < 0.15 and
        qkv_diff < 0.03 and qkv_max_rel < 0.15 and
        diff_k_150 < 0.03 and max_k_150 < 0.15 and
        diff_v_150 < 0.03 and max_v_150 < 0.15 and
        diff_qkv_150 < 0.03 and max_qkv_150 < 0.15
    )
    print(f"Gate 1 [Numerical Parity (< 0.03 mean, < 0.15 max)]: {'PASSED' if gate1_pass else 'FAILED'}")

    # Gate 2a: Fused KV Latency Dividend (shave at least 40 µs off isolated K+V dispatches)
    gate2a_pass = kv_savings_dispatch >= 40.0
    print(f"Gate 2a [Fused KV Latency Savings >= 40 µs]: {'PASSED' if gate2a_pass else 'FAILED'} (Dispatch: {kv_savings_dispatch:.2f} µs saved | Sequence: {kv_savings_seq:.2f} µs saved)")

    # Gate 2b: Unified Fused QKV Latency Dividend (shave at least 75 µs off isolated Q+K+V dispatches)
    gate2b_pass = qkv_savings_dispatch >= 75.0
    print(f"Gate 2b [Fused QKV Latency Savings >= 75 µs]: {'PASSED' if gate2b_pass else 'FAILED'} (Dispatch: {qkv_savings_dispatch:.2f} µs saved | Sequence: {qkv_savings_seq:.2f} µs saved)")

    print("=" * 80)
    return {
        "gate1": gate1_pass,
        "gate2a": gate2a_pass,
        "gate2b": gate2b_pass,
        "lat_isolated_total_sum": lat_isolated_total_sum,
        "lat_isolated_kv_sum": lat_isolated_kv_sum,
        "lat_isolated_seq": lat_isolated_seq,
        "lat_fused_kv": lat_fused_kv,
        "lat_fused_kv_seq": lat_fused_kv_seq,
        "lat_fused_qkv": lat_fused_qkv,
        "lat_bonsai": lat_bonsai,
    }


if __name__ == "__main__":
    run_benchmark()
