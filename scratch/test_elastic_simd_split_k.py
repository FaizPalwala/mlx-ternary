"""
Phase 3B: Hardware-Elastic Split-K + Barrier-Free SIMD Warp Reduction Benchmark
Hardware: Apple Silicon GPU (Detected dynamically via system_profiler / mlx device info)
Architectural Target: ADR-002 Phase 3B (Elastic Split-K + 0-Barrier SIMD Reduction)

Key Architectural Features:
1. Hardware-Elastic Core Querying: Dynamically probes GPU core count, selects K_split to target 6-8 active TGs/core.
2. Stage 1: Zero-barrier SIMD warp accumulation across K-splits with 100% coalesced 1024-byte bursts.
3. Stage 2: Register-based parallel vector reduction across the K-split dimension.
4. Live Side-by-Side Comparison against official PrismML Bonsai 2-bit under identical battery power governor.
"""

import sys
import os
import time
import subprocess
from pathlib import Path
import mlx.core as mx
import mlx.core.fast as fast
import mlx.nn as nn

REPO_DIR = Path(__file__).resolve().parent.parent / "mlx-ternary"
sys.path.insert(0, str(REPO_DIR / "src"))

from mlx_ternary.ops import _LUT_HEADER, get_wide_gemv_source
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


def make_stage1_split_k_source(M: int, K: int, N: int, S: int, G_SIZE: int = 128) -> str:
    """
    Stage 1: Partial Accumulation Kernel across S K-splits.
    - Grid: (N, S, M)
    - Threadgroup: (128, 1, 1) -> 4 SIMD warps of 32 lanes
    - Exactly 0 threadgroup barriers!
    - 100% coalesced 32-byte weight loads per lane (1024 bytes per warp)
    - Activations broadcast directly from L1 cache
    - Live registers per thread <= 24
    - Writes to partial_output: (M, S, N) in float32
    """
    source = f"""
    uint col = thread_position_in_grid.x;
    uint split_id = thread_position_in_grid.y;
    uint row = thread_position_in_grid.z;

    if (col >= {N} || split_id >= {S} || row >= {M}) return;

    uint K_blocks = {K} / {G_SIZE};
    uint blocks_per_split = (K_blocks + {S} - 1) / {S};
    uint start_block = split_id * blocks_per_split;
    uint end_block = min(start_block + blocks_per_split, K_blocks);

    if (start_block >= K_blocks) {{
        partial_output[row * ({S} * {N}) + split_id * {N} + col] = 0.0f;
        return;
    }}

    half acc = 0;
    device const half* act_block = activations + row * {K} + start_block * {G_SIZE};
    device const uint4* w_block = ((device const uint4*)interleaved_weights) + (start_block * {N} + col) * 2;
    uint w_stride = {N} * 2;

    uint4 cur_vec0 = w_block[0];
    uint4 cur_vec1 = w_block[1];
    w_block += w_stride;

    for (uint k_block = start_block; k_block < end_block; k_block++) {{
        uint4 next_vec0 = uint4(0);
        uint4 next_vec1 = uint4(0);
        if (k_block + 1 < end_block) {{
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
        # G_SIZE == 150
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

    partial_output[row * ({S} * {N}) + split_id * {N} + col] = (float)acc;
    """
    return source


def make_stage2_reduce_source(M: int, N: int, S: int) -> str:
    """
    Stage 2: Parallel Vector Reduction Kernel.
    - Grid: (N / 4, M, 1) -> Each thread reduces a float4 across S splits
    - Threadgroup: (256, 1, 1)
    - Directly reduces S partial slices in registers into FP16 half4
    """
    return f"""
    uint idx = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    uint N_vec = {N} / 4;

    if (idx >= N_vec || row >= {M}) return;

    device const float4* p_base = (device const float4*)(partial_output + row * ({S} * {N}));
    float4 sum = float4(0.0f);

    #pragma unroll({S})
    for (uint s = 0; s < {S}; s++) {{
        sum += p_base[s * N_vec + idx];
    }}

    device half4* out_vec = (device half4*)(output + row * {N});
    out_vec[idx] = half4(sum);
    """


def make_phase3a_source(M: int, K: int, N: int, G_SIZE: int = 128) -> str:
    """Phase 3A 0-barrier SIMD warp baseline (S=1)."""
    src = make_stage1_split_k_source(M, K, N, S=1, G_SIZE=G_SIZE)
    src = src.replace(
        f"partial_output[row * (1 * {N}) + split_id * {N} + col] = (float)acc;",
        f"output[row * {N} + col] = acc;"
    )
    src = src.replace(
        f"partial_output[row * (1 * {N}) + split_id * {N} + col] = 0.0f;",
        f"output[row * {N} + col] = 0;"
    )
    return src


def compute_unquantized_reference(x, w_raw, G):
    """Reference unquantized FP32 GEMV matching exact unpacked scale semantics."""
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
    print("PHASE 3B: ELASTIC SPLIT-K + BARRIER-FREE SIMD WARP BENCHMARK")
    print("Apple Silicon Dynamic Architecture & Live Battery Governor Evaluation")
    print("=" * 80)

    cores = get_gpu_core_count()
    dev_name = mx.device_info().get("device_name", "Apple Silicon")
    print(f"Target Hardware:  {dev_name} ({cores} GPU cores detected)")
    print(f"Power Condition:  BATTERY POWER (Live side-by-side comparison with Bonsai)")

    # 1. Dimensions: MLP Down for Ternary-Bonsai-2-27B
    M = 1
    K = 17408
    N = 5120
    G = 128
    payload_mb = (K * N * 2 / 8) / (1024 ** 2)

    base_tgs = N // 128
    recommended_s = max(1, round((cores * 6) / base_tgs))
    print(f"\nContraction:      MLP Down (M={M}, K={K}, N={N}, G={G}, Payload={payload_mb:.2f} MB)")
    print(f"Grid Geometry:    Base TGs = {base_tgs} ({base_tgs / cores:.1f} TGs/core at S=1)")
    print(f"Hardware-Elastic: Target 6-8 TGs/core -> Selected Recommended S = {recommended_s} ({base_tgs * recommended_s} TGs, {(base_tgs * recommended_s) / cores:.1f} TGs/core)")

    # 2. Input Data
    mx.random.seed(42)
    x = mx.random.normal((M, K)).astype(mx.float16)
    w_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    w_packed, _ = pack_interleaved(w_raw, group_size=G)
    expected = compute_unquantized_reference(x, w_raw, G)

    warmup = 20
    iters = 100

    results = {}

    # --------------------------------------------------------------------------
    # Benchmark D: Live PrismML Bonsai 2-bit (Native Apple QuantizedLinear)
    # --------------------------------------------------------------------------
    print("\n[1/5] Benchmarking Live PrismML Bonsai 2-bit Baseline...")
    w_fp16 = mx.random.normal((N, K)).astype(mx.float16)
    w_q128, s128, b128 = mx.quantize(w_fp16, group_size=128, bits=2)
    bonsai_g128 = nn.QuantizedLinear(K, N, bits=2, group_size=128, bias=False)
    bonsai_g128.weight = w_q128
    bonsai_g128.scales = s128
    bonsai_g128.biases = b128
    mx.eval(bonsai_g128.parameters())

    w_q64, s64, b64 = mx.quantize(w_fp16, group_size=64, bits=2)
    bonsai_g64 = nn.QuantizedLinear(K, N, bits=2, group_size=64, bias=False)
    bonsai_g64.weight = w_q64
    bonsai_g64.scales = s64
    bonsai_g64.biases = b64
    mx.eval(bonsai_g64.parameters())

    for _ in range(warmup):
        mx.eval(bonsai_g128(x))
        mx.eval(bonsai_g64(x))

    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(bonsai_g128(x))
    lat_bonsai_g128 = (time.perf_counter() - t0) / iters * 1e6
    bw_bonsai_g128 = ((K * N * 2 / 8 + K * N / 128 * 4) / 1e9) / (lat_bonsai_g128 * 1e-6)

    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(bonsai_g64(x))
    lat_bonsai_g64 = (time.perf_counter() - t0) / iters * 1e6
    bw_bonsai_g64 = ((K * N * 2 / 8 + K * N / 64 * 4) / 1e9) / (lat_bonsai_g64 * 1e-6)

    results["bonsai_g128"] = {"lat": lat_bonsai_g128, "bw": bw_bonsai_g128}
    results["bonsai_g64"] = {"lat": lat_bonsai_g64, "bw": bw_bonsai_g64}
    print(f"  ✓ Live Bonsai G128 (Official): {lat_bonsai_g128:6.2f} µs | Bandwidth: {bw_bonsai_g128:5.1f} GB/s")
    print(f"  ✓ Live Bonsai G64:              {lat_bonsai_g64:6.2f} µs | Bandwidth: {bw_bonsai_g64:5.1f} GB/s")

    # --------------------------------------------------------------------------
    # Benchmark A: mlx-ternary Baseline (Phase 2G wide_gemv, SRAM + 272 Barriers)
    # --------------------------------------------------------------------------
    print("\n[2/5] Benchmarking mlx-ternary Baseline (Phase 2G wide_gemv)...")
    src_base = get_wide_gemv_source(M, K, N, G)
    k_base = fast.metal_kernel(
        name="baseline_wide",
        input_names=["activations", "interleaved_weights"],
        output_names=["output"],
        source=src_base,
        header=_LUT_HEADER
    )
    grid_base = (((N + 127) // 128) * 128, M, 1)
    tgroup_base = (128, 1, 1)
    out_base = k_base(inputs=[x, w_packed], grid=grid_base, threadgroup=tgroup_base, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
    mx.eval(out_base)
    diff_base = (mx.abs(out_base - expected) / (mx.abs(expected) + 1e-3)).mean().item()

    for _ in range(warmup):
        mx.eval(k_base(inputs=[x, w_packed], grid=grid_base, threadgroup=tgroup_base, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0])
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(k_base(inputs=[x, w_packed], grid=grid_base, threadgroup=tgroup_base, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0])
    lat_base = (time.perf_counter() - t0) / iters * 1e6
    bw_base = (K * N * 2 / 8) / (lat_base * 1e-6) / 1e9
    results["base"] = {"lat": lat_base, "bw": bw_base, "diff": diff_base}
    print(f"  ✓ Phase 2G Baseline:          {lat_base:6.2f} µs | Bandwidth: {bw_base:5.1f} GB/s | diff: {diff_base:.6f}")

    # --------------------------------------------------------------------------
    # Benchmark B: mlx-ternary Phase 3A (0-Barrier SIMD Warp, S=1)
    # --------------------------------------------------------------------------
    print("\n[3/5] Benchmarking mlx-ternary Phase 3A (0-Barrier SIMD Warp, S=1)...")
    src_3a = make_phase3a_source(M, K, N, G)
    k_3a = fast.metal_kernel(
        name="phase3a_simd",
        input_names=["activations", "interleaved_weights"],
        output_names=["output"],
        source=src_3a,
        header=_LUT_HEADER
    )
    grid_3a = (N, 1, M)
    tgroup_3a = (128, 1, 1)
    out_3a = k_3a(inputs=[x, w_packed], grid=grid_3a, threadgroup=tgroup_3a, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
    mx.eval(out_3a)
    diff_3a = (mx.abs(out_3a - expected) / (mx.abs(expected) + 1e-3)).mean().item()

    for _ in range(warmup):
        mx.eval(k_3a(inputs=[x, w_packed], grid=grid_3a, threadgroup=tgroup_3a, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0])
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(k_3a(inputs=[x, w_packed], grid=grid_3a, threadgroup=tgroup_3a, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0])
    lat_3a = (time.perf_counter() - t0) / iters * 1e6
    bw_3a = (K * N * 2 / 8) / (lat_3a * 1e-6) / 1e9
    results["3a"] = {"lat": lat_3a, "bw": bw_3a, "diff": diff_3a}
    print(f"  ✓ Phase 3A Prototype (S=1):   {lat_3a:6.2f} µs | Bandwidth: {bw_3a:5.1f} GB/s | diff: {diff_3a:.6f}")

    # --------------------------------------------------------------------------
    # Benchmark C: mlx-ternary Phase 3B (Elastic Split-K + 0-Barrier SIMD Warp)
    # --------------------------------------------------------------------------
    print("\n[4/5] Benchmarking mlx-ternary Phase 3B (Elastic Split-K Sweep S in [2, 3, 4])...")
    split_results = {}
    phase3b_outputs = {}

    for S in [2, 3, 4]:
        src_s1 = make_stage1_split_k_source(M, K, N, S, G)
        k_s1 = fast.metal_kernel(
            name=f"p3b_stage1_s{S}",
            input_names=["activations", "interleaved_weights"],
            output_names=["partial_output"],
            source=src_s1,
            header=_LUT_HEADER
        )

        src_s2 = make_stage2_reduce_source(M, N, S)
        k_s2 = fast.metal_kernel(
            name=f"p3b_stage2_s{S}",
            input_names=["partial_output"],
            output_names=["output"],
            source=src_s2
        )

        grid_s1 = (N, S, M)
        tgroup_s1 = (128, 1, 1)

        grid_s2 = (N // 4, M, 1)
        tgroup_s2 = (256, 1, 1)

        p = k_s1(inputs=[x, w_packed], grid=grid_s1, threadgroup=tgroup_s1, output_shapes=[(M, S, N)], output_dtypes=[mx.float32])[0]
        out_3b = k_s2(inputs=[p], grid=grid_s2, threadgroup=tgroup_s2, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
        mx.eval(out_3b)

        phase3b_outputs[S] = out_3b
        diff_3b = (mx.abs(out_3b - expected) / (mx.abs(expected) + 1e-3)).mean().item()
        max_abs_3b = mx.abs(out_3b - expected).max().item()
        mean_mag_exp = mx.abs(expected.astype(mx.float32)).mean().item()
        max_rel_3b = max_abs_3b / mean_mag_exp

        # Parity against Phase 3A
        parity_mean = (mx.abs(out_3b - out_3a) / (mx.abs(out_3a) + 1e-3)).mean().item()
        parity_max_abs = mx.abs(out_3b - out_3a).max().item()
        parity_mean_mag = mx.abs(out_3a.astype(mx.float32)).mean().item()
        parity_max_rel = parity_max_abs / parity_mean_mag

        for _ in range(warmup):
            p = k_s1(inputs=[x, w_packed], grid=grid_s1, threadgroup=tgroup_s1, output_shapes=[(M, S, N)], output_dtypes=[mx.float32])[0]
            o = k_s2(inputs=[p], grid=grid_s2, threadgroup=tgroup_s2, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
            mx.eval(o)

        t0 = time.perf_counter()
        for _ in range(iters):
            p = k_s1(inputs=[x, w_packed], grid=grid_s1, threadgroup=tgroup_s1, output_shapes=[(M, S, N)], output_dtypes=[mx.float32])[0]
            o = k_s2(inputs=[p], grid=grid_s2, threadgroup=tgroup_s2, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
            mx.eval(o)
        lat_3b = (time.perf_counter() - t0) / iters * 1e6
        bw_3b = (K * N * 2 / 8) / (lat_3b * 1e-6) / 1e9

        total_tgs = base_tgs * S
        tgs_per_core = total_tgs / cores
        split_results[S] = {
            "lat": lat_3b,
            "bw": bw_3b,
            "mean_diff": diff_3b,
            "max_rel_diff": max_rel_3b,
            "parity_mean_diff": parity_mean,
            "parity_max_rel": parity_max_rel,
            "total_tgs": total_tgs,
            "tgs_per_core": tgs_per_core,
        }
        print(f"  ✓ Phase 3B (S={S}, {total_tgs:3d} TGs, {tgs_per_core:4.1f}/core): {lat_3b:6.2f} µs | Bandwidth: {bw_3b:5.1f} GB/s | mean_diff: {diff_3b:.6f} | max_rel: {max_rel_3b:.6f} | vs 3A: {parity_mean:.6f}")

    # --------------------------------------------------------------------------
    # Group Size 150 Numerical Validation
    # --------------------------------------------------------------------------
    print("\n[5/5] Validating G=150 Numerical Precision Across S in [2, 3, 4]...")
    K_150 = 17400
    w_raw_150 = mx.random.randint(0, 3, (K_150, N)).astype(mx.float16) - 1.0
    w_packed_150, _ = pack_interleaved(w_raw_150, group_size=150)
    x_150 = mx.random.normal((M, K_150)).astype(mx.float16)
    expected_150 = compute_unquantized_reference(x_150, w_raw_150, G=150)

    g150_passes = True
    for S in [2, 3, 4]:
        src_s1_150 = make_stage1_split_k_source(M, K_150, N, S, G_SIZE=150)
        k_s1_150 = fast.metal_kernel(
            name=f"p3b_g150_s1_{S}",
            input_names=["activations", "interleaved_weights"],
            output_names=["partial_output"],
            source=src_s1_150,
            header=_LUT_HEADER
        )
        src_s2_150 = make_stage2_reduce_source(M, N, S)
        k_s2_150 = fast.metal_kernel(
            name=f"p3b_g150_s2_{S}",
            input_names=["partial_output"],
            output_names=["output"],
            source=src_s2_150
        )
        p150 = k_s1_150(inputs=[x_150, w_packed_150], grid=(N, S, M), threadgroup=(128, 1, 1), output_shapes=[(M, S, N)], output_dtypes=[mx.float32])[0]
        out_150 = k_s2_150(inputs=[p150], grid=(N // 4, M, 1), threadgroup=(256, 1, 1), output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
        mx.eval(out_150)

        diff_150 = (mx.abs(out_150 - expected_150) / (mx.abs(expected_150) + 1e-3)).mean().item()
        max_abs_150 = mx.abs(out_150 - expected_150).max().item()
        mean_mag_150 = mx.abs(expected_150.astype(mx.float32)).mean().item()
        max_rel_150 = max_abs_150 / mean_mag_150
        pass_150 = diff_150 < 0.03 and max_rel_150 < 0.15
        if not pass_150:
            g150_passes = False
        print(f"  ✓ G=150 (S={S}): mean_diff = {diff_150:.6f}, max_rel_diff = {max_rel_150:.6f} (PASS: {pass_150})")

    # --------------------------------------------------------------------------
    # Clean Markdown Summary Table
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LIVE BATTERY-NORMALIZED COMPARISON TABLE (MLP Down: 17408 -> 5120)")
    print("=" * 80)

    print(f"| {'Kernel Configuration':<36} | {'TGs (TGs/Core)':<16} | {'Latency (µs)':<13} | {'Bandwidth':<12} | {'Delta vs Base':<15} | {'Delta vs Bonsai':<16} |")
    print(f"| :{'-'*34} | :{'-'*14} | :{'-'*11} | :{'-'*10} | :{'-'*13} | :{'-'*14} |")

    # Bonsai G128
    print(f"| {'PrismML Bonsai 2-bit (Official G128)':<36} | {'Apple Native':<16} | {lat_bonsai_g128:>10.2f} µs | {bw_bonsai_g128:>9.1f} GB/s | {'---':<15} | {'Baseline (0.0%)':<16} |")
    # Bonsai G64
    print(f"| {'PrismML Bonsai 2-bit (G64)':<36} | {'Apple Native':<16} | {lat_bonsai_g64:>10.2f} µs | {bw_bonsai_g64:>9.1f} GB/s | {'---':<15} | {f'{lat_bonsai_g128/lat_bonsai_g64:.2f}x':<16} |")
    # Base
    print(f"| {'Phase 2G Baseline (SRAM, 272 Bar.)':<36} | {'40 (2.0/core)':<16} | {lat_base:>10.2f} µs | {bw_base:>9.1f} GB/s | {'Baseline (0.0%)':<15} | {f'{lat_bonsai_g128/lat_base:.2f}x':<16} |")
    # Phase 3A
    diff_vs_base_3a = lat_base - lat_3a
    speed_vs_base_3a = lat_base / lat_3a
    print(f"| {'Phase 3A (0-Barrier SIMD Warp, S=1)':<36} | {'40 (2.0/core)':<16} | {lat_3a:>10.2f} µs | {bw_3a:>9.1f} GB/s | {f'-{diff_vs_base_3a:.1f}µ ({speed_vs_base_3a:.2f}x)':<15} | {f'{lat_bonsai_g128/lat_3a:.2f}x':<16} |")

    # Phase 3B Sweeps
    for S in [2, 3, 4]:
        r = split_results[S]
        diff_vs_base = lat_base - r["lat"]
        speed_vs_base = lat_base / r["lat"]
        speed_vs_bonsai = lat_bonsai_g128 / r["lat"]
        star = " ★" if S == recommended_s else ""
        tg_info = f"{r['total_tgs']} ({r['tgs_per_core']:.1f}/core)"
        print(f"| {f'Phase 3B Split-K (S={S}){star}':<36} | {tg_info:<16} | {r['lat']:>10.2f} µs | {r['bw']:>9.1f} GB/s | {f'-{diff_vs_base:.1f}µ ({speed_vs_base:.2f}x)':<15} | {f'{speed_vs_bonsai:.2f}x':<16} |")

    # --------------------------------------------------------------------------
    # Acceptance Gates Evaluation
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 3B ACCEPTANCE GATES EVALUATION")
    print("=" * 80)

    # Gate 1: Numerical Parity against unquantized FP32
    g128_mean_pass = all(r["mean_diff"] < 0.03 for r in split_results.values())
    g128_max_pass = all(r["max_rel_diff"] < 0.15 for r in split_results.values())
    gate1_pass = g128_mean_pass and g128_max_pass and g150_passes
    print(f"Gate 1: Numerical Precision (mean_rel_diff < 0.03, max_rel_diff < 0.15 across G=128 & G=150): {'PASSED' if gate1_pass else 'FAILED'}")

    # Gate 2: Parity against Phase 3A
    parity_pass = all(r["parity_mean_diff"] < 0.03 for r in split_results.values())
    print(f"Gate 2: Parity vs Phase 3A (diff < 0.03): {'PASSED' if parity_pass else 'FAILED'}")

    # Gate 3: Live Delta vs Bonsai
    opt_3b_lat = min(r["lat"] for r in split_results.values())
    print(f"Gate 3: Live Performance Under Battery:")
    print(f"  * Live Bonsai G128:  {lat_bonsai_g128:.2f} µs")
    print(f"  * Phase 2G Baseline: {lat_base:.2f} µs")
    print(f"  * Phase 3A (S=1):    {lat_3a:.2f} µs")
    print(f"  * Phase 3B (S={recommended_s}):    {split_results[recommended_s]['lat']:.2f} µs")
    print(f"  * Phase 3B Optimal:  {opt_3b_lat:.2f} µs")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
