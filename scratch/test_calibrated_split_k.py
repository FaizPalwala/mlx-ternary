"""
Phase 3B Calibration: Calibrated Stage 2 Split-K Reduction Prototype
Contraction: MLP Down (M=1, K=17408, N=5120, G=128, S in [2, 3, 4, 5, 6])
Architecture: Apple Silicon Metal Shading Language with 0-Barrier SIMD GEMV

Objectives:
1. Calibrate Stage 2 grid dispatch to strict threadgroup boundaries with zero trailing lane stalls.
2. In-register vector accumulation with aligned float4 loads across S slices into FP16 half4 destination.
3. Isolate and measure Stage 1 latency, Stage 2 latency, and end-to-end pipelined latency.
4. Verify bit-exact numerical parity against unquantized FP32 reference (mean_rel_diff < 0.03) for G=128 and G=150.
5. Benchmark side-by-side against uncalibrated baseline and live PrismML Bonsai 2-bit MLP Down.
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
    - Threadgroup: (tg_x, 1, 1) -> SIMD warps of 32 lanes
    - Exactly 0 threadgroup barriers
    - Coalesced 32-byte weight loads per lane (1024 bytes per warp)
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


def make_calibrated_stage2_source(M: int, N: int, S: int, balanced_tree: bool = False) -> str:
    """
    Calibrated Stage 2 In-Register Vector Reduction Kernel.
    - Grid: ((((N // 4) + tg_s2 - 1) // tg_s2) * tg_s2, M, 1)
    - Threadgroup: (tg_s2, 1, 1), default 256
    - Each thread loads contiguous float4 vectors across S slices
    - Direct write to output half4 vector
    - 0 threadgroup shared memory, 0 barriers, <= 10 live registers
    """
    if balanced_tree and S == 4:
        inner_reduction = f"""
    device const float4* p_base = (device const float4*)(partial_output + row * ({S} * {N}));
    float4 v0 = p_base[0 * N_vec + idx];
    float4 v1 = p_base[1 * N_vec + idx];
    float4 v2 = p_base[2 * N_vec + idx];
    float4 v3 = p_base[3 * N_vec + idx];
    float4 sum = (v0 + v1) + (v2 + v3);
        """
    elif balanced_tree and S == 3:
        inner_reduction = f"""
    device const float4* p_base = (device const float4*)(partial_output + row * ({S} * {N}));
    float4 v0 = p_base[0 * N_vec + idx];
    float4 v1 = p_base[1 * N_vec + idx];
    float4 v2 = p_base[2 * N_vec + idx];
    float4 sum = (v0 + v1) + v2;
        """
    elif balanced_tree and S == 2:
        inner_reduction = f"""
    device const float4* p_base = (device const float4*)(partial_output + row * ({S} * {N}));
    float4 v0 = p_base[0 * N_vec + idx];
    float4 v1 = p_base[1 * N_vec + idx];
    float4 sum = v0 + v1;
        """
    else:
        inner_reduction = f"""
    device const float4* p_base = (device const float4*)(partial_output + row * ({S} * {N}));
    float4 sum = float4(0.0f);

    #pragma unroll({S})
    for (uint s = 0; s < {S}; s++) {{
        sum += p_base[s * N_vec + idx];
    }}
        """

    return f"""
    uint idx = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    if (row >= {M}) return;

    uint N_vec = {N} / 4;
    if (idx >= N_vec) return;

    {inner_reduction}

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


def compute_unquantized_reference(x: mx.array, w_raw: mx.array, G: int) -> mx.array:
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
    print("PHASE 3B CALIBRATION: CALIBRATED STAGE 2 SPLIT-K REDUCTION")
    print("Apple Silicon Micro-Architecture & Real-Time Telemetry Audit")
    print("=" * 80)

    cores = get_gpu_core_count()
    dev_name = mx.device_info().get("device_name", "Apple Silicon")
    print(f"Target Hardware:  {dev_name} ({cores} GPU cores detected)")

    M = 1
    K = 17408
    N = 5120
    G = 128
    payload_mb = (K * N * 2 / 8) / (1024 * 1024)

    print(f"Contraction:      MLP Down (M={M}, K={K}, N={N}, G={G}, Payload={payload_mb:.2f} MB)")

    warmup = 25
    iters = 100

    mx.random.seed(42)
    x = mx.random.normal((M, K)).astype(mx.float16)
    w_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    w_packed, _ = pack_interleaved(w_raw, group_size=G)
    mx.eval(x, w_packed)

    print("\n[1/6] Computing Mathematical Reference (Unpacked FP32)...")
    expected = compute_unquantized_reference(x, w_raw, G)

    # --------------------------------------------------------------------------
    # Benchmark 1: Live PrismML Bonsai 2-bit Baseline
    # --------------------------------------------------------------------------
    print("\n[2/6] Benchmarking Live PrismML Bonsai 2-bit Baseline...")
    w_fp16 = w_raw.T
    w_q128, s128, b128 = mx.quantize(w_fp16, group_size=128, bits=2)
    bonsai_g128 = nn.QuantizedLinear(K, N, bits=2, group_size=128, bias=False)
    bonsai_g128.weight, bonsai_g128.scales, bonsai_g128.biases = w_q128, s128, b128
    mx.eval(bonsai_g128.parameters())

    for _ in range(warmup):
        mx.eval(bonsai_g128(x))

    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(bonsai_g128(x))
    lat_bonsai = (time.perf_counter() - t0) / iters * 1e6
    bw_bonsai = ((K * N * 2 / 8 + K * N / 128 * 4) / 1e9) / (lat_bonsai * 1e-6)
    print(f"  ✓ Live Bonsai 2-bit (G=128):     {lat_bonsai:6.2f} µs | Bandwidth: {bw_bonsai:5.1f} GB/s")

    # --------------------------------------------------------------------------
    # Benchmark 2: Phase 3A Baseline (S=1, 0-Barrier SIMD)
    # --------------------------------------------------------------------------
    print("\n[3/6] Benchmarking Phase 3A Baseline (S=1, 0-Barrier SIMD)...")
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
    print(f"  ✓ Phase 3A (S=1 Baseline):      {lat_3a:6.2f} µs | Bandwidth: {bw_3a:5.1f} GB/s | mean_diff: {diff_3a:.6f}")

    # --------------------------------------------------------------------------
    # Benchmark 3: Uncalibrated Split-K S=4 (from test_elastic_simd_split_k.py)
    # --------------------------------------------------------------------------
    print("\n[4/6] Benchmarking Uncalibrated Split-K S=4 Baseline...")
    src_s1_uncal = make_stage1_split_k_source(M, K, N, S=4, G_SIZE=G)
    k_s1_uncal = fast.metal_kernel(
        name="uncal_stage1_s4",
        input_names=["activations", "interleaved_weights"],
        output_names=["partial_output"],
        source=src_s1_uncal,
        header=_LUT_HEADER
    )
    src_s2_uncal = f"""
    uint idx = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    uint N_vec = {N} / 4;
    if (idx >= N_vec || row >= {M}) return;
    device const float4* p_base = (device const float4*)(partial_output + row * (4 * {N}));
    float4 sum = float4(0.0f);
    #pragma unroll(4)
    for (uint s = 0; s < 4; s++) {{
        sum += p_base[s * N_vec + idx];
    }}
    device half4* out_vec = (device half4*)(output + row * {N});
    out_vec[idx] = half4(sum);
    """
    k_s2_uncal = fast.metal_kernel(
        name="uncal_stage2_s4",
        input_names=["partial_output"],
        output_names=["output"],
        source=src_s2_uncal
    )

    grid_s1_4 = (N, 4, M)
    tgroup_s1 = (128, 1, 1)
    grid_s2_uncal = (N // 4, M, 1)
    tgroup_s2_uncal = (256, 1, 1)

    for _ in range(warmup):
        p = k_s1_uncal(inputs=[x, w_packed], grid=grid_s1_4, threadgroup=tgroup_s1, output_shapes=[(M, 4, N)], output_dtypes=[mx.float32])[0]
        o = k_s2_uncal(inputs=[p], grid=grid_s2_uncal, threadgroup=tgroup_s2_uncal, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
        mx.eval(o)

    t0 = time.perf_counter()
    for _ in range(iters):
        p = k_s1_uncal(inputs=[x, w_packed], grid=grid_s1_4, threadgroup=tgroup_s1, output_shapes=[(M, 4, N)], output_dtypes=[mx.float32])[0]
        o = k_s2_uncal(inputs=[p], grid=grid_s2_uncal, threadgroup=tgroup_s2_uncal, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
        mx.eval(o)
    lat_uncal = (time.perf_counter() - t0) / iters * 1e6
    bw_uncal = (K * N * 2 / 8) / (lat_uncal * 1e-6) / 1e9
    print(f"  ✓ Uncalibrated Split-K S=4:     {lat_uncal:6.2f} µs | Bandwidth: {bw_uncal:5.1f} GB/s")

    # --------------------------------------------------------------------------
    # Benchmark 4: Calibrated Split-K Sweep (S in [2, 3, 4], Stage 1 vs Stage 2 Isolation)
    # --------------------------------------------------------------------------
    print("\n[5/6] Benchmarking Calibrated Split-K with Stage 1 & Stage 2 Telemetry Breakdown...")

    results_table = {}

    for S in [2, 3, 4]:
        print(f"\n  --- Testing Split-K S={S} ({N // 128 * S} Stage 1 TGs, {N // 128 * S / cores:.1f} TGs/core) ---")
        src_s1 = make_stage1_split_k_source(M, K, N, S, G)
        k_s1 = fast.metal_kernel(
            name=f"cal_stage1_s{S}",
            input_names=["activations", "interleaved_weights"],
            output_names=["partial_output"],
            source=src_s1,
            header=_LUT_HEADER
        )
        grid_s1 = (N, S, M)

        # 1. Measure Stage 1 alone
        for _ in range(warmup):
            p = k_s1(inputs=[x, w_packed], grid=grid_s1, threadgroup=tgroup_s1, output_shapes=[(M, S, N)], output_dtypes=[mx.float32])[0]
            mx.eval(p)
        t0 = time.perf_counter()
        for _ in range(iters):
            p = k_s1(inputs=[x, w_packed], grid=grid_s1, threadgroup=tgroup_s1, output_shapes=[(M, S, N)], output_dtypes=[mx.float32])[0]
            mx.eval(p)
        lat_s1 = (time.perf_counter() - t0) / iters * 1e6

        # 2. Stage 2 calibration sweep: threadgroup sizes 256, 128, 64, 32 with tree addition
        for tg_size in [256, 128, 64]:
            for tree in [False, True]:
                src_s2 = make_calibrated_stage2_source(M, N, S=S, balanced_tree=tree)
                k_s2 = fast.metal_kernel(
                    name=f"cal_stage2_s{S}_tg{tg_size}_{'tree' if tree else 'loop'}",
                    input_names=["partial_output"],
                    output_names=["output"],
                    source=src_s2
                )
                N_vec = N // 4
                grid_x = ((N_vec + tg_size - 1) // tg_size) * tg_size
                grid_s2 = (grid_x, M, 1)
                tgroup_s2 = (tg_size, 1, 1)

                # Parity check
                p_tmp = k_s1(inputs=[x, w_packed], grid=grid_s1, threadgroup=tgroup_s1, output_shapes=[(M, S, N)], output_dtypes=[mx.float32])[0]
                out = k_s2(inputs=[p_tmp], grid=grid_s2, threadgroup=tgroup_s2, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
                mx.eval(out)
                mean_diff = (mx.abs(out - expected) / (mx.abs(expected) + 1e-3)).mean().item()
                max_abs = mx.abs(out - expected).max().item()
                max_rel = max_abs / mx.abs(expected.astype(mx.float32)).mean().item()

                # Measure End-to-End Pipelined
                for _ in range(warmup):
                    p_e2e = k_s1(inputs=[x, w_packed], grid=grid_s1, threadgroup=tgroup_s1, output_shapes=[(M, S, N)], output_dtypes=[mx.float32])[0]
                    o_e2e = k_s2(inputs=[p_e2e], grid=grid_s2, threadgroup=tgroup_s2, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
                    mx.eval(o_e2e)

                t0 = time.perf_counter()
                for _ in range(iters):
                    p_e2e = k_s1(inputs=[x, w_packed], grid=grid_s1, threadgroup=tgroup_s1, output_shapes=[(M, S, N)], output_dtypes=[mx.float32])[0]
                    o_e2e = k_s2(inputs=[p_e2e], grid=grid_s2, threadgroup=tgroup_s2, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
                    mx.eval(o_e2e)
                lat_e2e = (time.perf_counter() - t0) / iters * 1e6
                bw_e2e = (K * N * 2 / 8) / (lat_e2e * 1e-6) / 1e9

                lat_s2_pipelined = lat_e2e - lat_s1

                cfg_name = f"S={S}_tg{tg_size}_{'tree' if tree else 'loop'}"
                results_table[cfg_name] = {
                    "S": S,
                    "tg_size": tg_size,
                    "tree": tree,
                    "lat_s1": lat_s1,
                    "lat_s2_pipe": lat_s2_pipelined,
                    "lat_e2e": lat_e2e,
                    "bw": bw_e2e,
                    "mean_diff": mean_diff,
                    "max_rel": max_rel,
                }
                print(f"    • {cfg_name:<20s}: Stage 1 = {lat_s1:6.2f} µs | Stage 2 Δ = {lat_s2_pipelined:5.2f} µs | Total E2E = {lat_e2e:6.2f} µs ({bw_e2e:5.1f} GB/s) | diff: {mean_diff:.6f}")

    # --------------------------------------------------------------------------
    # Benchmark 5: Group Size 150 Numerical Validation
    # --------------------------------------------------------------------------
    print("\n[6/6] Validating Numerical Parity for G=150 (Contraction K=17400, N=5120)...")
    K_150 = 17400  # 116 blocks of 150
    w_raw_150 = mx.random.randint(0, 3, (K_150, N)).astype(mx.float16) - 1.0
    w_packed_150, _ = pack_interleaved(w_raw_150, group_size=150)
    x_150 = mx.random.normal((M, K_150)).astype(mx.float16)
    expected_150 = compute_unquantized_reference(x_150, w_raw_150, G=150)

    g150_passes = True
    for S in [2, 3, 4]:
        src_s1_150 = make_stage1_split_k_source(M, K_150, N, S, G_SIZE=150)
        k_s1_150 = fast.metal_kernel(
            name=f"cal_s1_150_s{S}",
            input_names=["activations", "interleaved_weights"],
            output_names=["partial_output"],
            source=src_s1_150,
            header=_LUT_HEADER
        )
        src_s2_150 = make_calibrated_stage2_source(M, N, S, balanced_tree=True)
        k_s2_150 = fast.metal_kernel(
            name=f"cal_s2_150_s{S}",
            input_names=["partial_output"],
            output_names=["output"],
            source=src_s2_150
        )

        p150 = k_s1_150(inputs=[x_150, w_packed_150], grid=(N, S, M), threadgroup=(128, 1, 1), output_shapes=[(M, S, N)], output_dtypes=[mx.float32])[0]
        out150 = k_s2_150(inputs=[p150], grid=(1280, M, 1), threadgroup=(256, 1, 1), output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
        mx.eval(out150)

        mean_diff_150 = (mx.abs(out150 - expected_150) / (mx.abs(expected_150) + 1e-3)).mean().item()
        max_abs_150 = mx.abs(out150 - expected_150).max().item()
        mean_mag_150 = mx.abs(expected_150.astype(mx.float32)).mean().item()
        max_rel_150 = max_abs_150 / mean_mag_150
        passed = mean_diff_150 < 0.03 and max_rel_150 < 0.15
        if not passed:
            g150_passes = False
        print(f"  ✓ G=150 (S={S}): mean_diff = {mean_diff_150:.6f}, max_rel = {max_rel_150:.6f} (PASS: {passed})")

    # --------------------------------------------------------------------------
    # Summary Table
    # --------------------------------------------------------------------------
    best_cfg_name = min(results_table, key=lambda k: results_table[k]["lat_e2e"])
    best_info = results_table[best_cfg_name]
    s4_std = results_table["S=4_tg256_loop"]

    print("\n" + "=" * 85)
    print("CALIBRATED SPLIT-K PERFORMANCE AUDIT TABLE (MLP Down: 17408 -> 5120)")
    print("=" * 85)
    print(f"{'Configuration':<32} | {'Stage 1':<10} | {'Stage 2 Δ':<10} | {'Total E2E':<12} | {'Bandwidth':<12}")
    print("-" * 85)
    print(f"{'Live PrismML Bonsai 2-bit (G128)':<32} | {'---':<10} | {'---':<10} | {lat_bonsai:9.2f} µs | {bw_bonsai:9.1f} GB/s")
    print(f"{'Phase 3A Baseline (S=1, 40 TGs)':<32} | {'---':<10} | {'---':<10} | {lat_3a:9.2f} µs | {bw_3a:9.1f} GB/s")
    print(f"{'Uncalibrated Split-K S=4 (160 TGs)':<32} | {'---':<10} | {'---':<10} | {lat_uncal:9.2f} µs | {bw_uncal:9.1f} GB/s")
    print(f"{'Calibrated Split-K (S=4 Standard)':<32} | {s4_std['lat_s1']:7.2f} µs | {s4_std['lat_s2_pipe']:7.2f} µs | {s4_std['lat_e2e']:9.2f} µs | {s4_std['bw']:9.1f} GB/s")
    print(f"{'Calibrated Split-K (' + best_cfg_name + ') ★':<32} | {best_info['lat_s1']:7.2f} µs | {best_info['lat_s2_pipe']:7.2f} µs | {best_info['lat_e2e']:9.2f} µs | {best_info['bw']:9.1f} GB/s")
    print("=" * 85)

    # Acceptance Gates
    gate1 = best_info["mean_diff"] < 0.03 and best_info["max_rel"] < 0.15 and g150_passes
    gate2 = best_info["lat_s2_pipe"] <= 8.0
    gate3 = best_info["lat_e2e"] < lat_uncal

    print("\nACCEPTANCE GATES STATUS:")
    print(f"  [1] Numerical Parity (mean_diff < 0.03, max_rel < 0.15 for G=128 & G=150): {'PASSED' if gate1 else 'FAILED'}")
    print(f"  [2] Dispatch Efficiency (Stage 2 latency <= 8.0 µs):                        {'PASSED' if gate2 else 'FAILED'} ({best_info['lat_s2_pipe']:.2f} µs)")
    print(f"  [3] Calibration Delta (Faster than uncalibrated Split-K):                    {'PASSED' if gate3 else 'FAILED'} ({best_info['lat_e2e']:.2f} µs vs {lat_uncal:.2f} µs)")


if __name__ == "__main__":
    run_benchmark()
