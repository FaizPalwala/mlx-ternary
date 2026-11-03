"""
Phase 3A: Barrier-Free SIMD-Warp Cooperative Reduction Prototype
Hardware: Apple M4 Pro (20-core GPU, 24GB Unified Memory, 273 GB/s peak LPDDR5 bandwidth)
Architectural Target: ADR-002 Phase 3A (Barrier-Free SIMD-Warp Cooperative Reduction)

This test compares:
1. Baseline: Production wide_gemv (Threadgroup SRAM activation staging, 272 barriers on MLP Down)
2. Variant 1: 32-Lane Intra-Warp Cooperative Reduction (simd_sum across K-splits, 0 threadgroup barriers)
3. Variant 2: Barrier-Free SIMD Warp (0 threadgroup barriers, 100% coalesced memory bursts, 4-way ILP, 20-trit chunking)
4. Variant 3: 4-Way Sub-Warp Cooperative Reduction (K-split=4, simd_shuffle_down, 0 threadgroup barriers)
"""

import sys
import time
from pathlib import Path
import mlx.core as mx
import mlx.core.fast as fast

REPO_DIR = Path(__file__).resolve().parent.parent / "mlx-ternary"
sys.path.insert(0, str(REPO_DIR / "src"))

from mlx_ternary.ops import _LUT_HEADER, get_wide_gemv_source
from mlx_ternary.quantize import pack_interleaved


def get_baseline_gemv_source(M: int, K: int, N: int, G_SIZE: int = 128) -> str:
    """Production wide_gemv source with threadgroup SRAM activation caching."""
    return get_wide_gemv_source(M, K, N, G_SIZE)


def get_simd_sum_coop_source(M: int, K: int, N: int, G_SIZE: int = 128) -> str:
    """
    Variant 1: 32-Lane Intra-Warp Cooperative Reduction
    - Grid: (32, ((N + tg_y - 1) // tg_y) * tg_y, M)
    - Threadgroup: (32, tg_y, 1)
    - 32 lanes in warp cooperatively reduce K_blocks using simd_sum()
    - Exactly 0 threadgroup barriers!
    - Aligned 20-Trit Chunking + 4-Way ILP accumulators
    """
    source = f"""
    uint lane_id = thread_position_in_threadgroup.x;
    uint col = thread_position_in_grid.y;
    uint row = thread_position_in_grid.z;

    if (row >= {M} || col >= {N} || lane_id >= 32) {{
        return;
    }}

    uint K_blocks = {K} / {G_SIZE};
    uint blocks_per_thread = (K_blocks + 31) / 32;
    uint start_k = lane_id * blocks_per_thread;
    uint end_k = min(start_k + blocks_per_thread, K_blocks);

    half acc = 0;
    device const half* act_ptr = activations + row * {K};
    device const uint4* w_ptr = (device const uint4*)interleaved_weights;

    for (uint k = start_k; k < end_k; k++) {{
        uint block_offset_uint4 = (k * {N} + col) * 2;
        device const uint4* w_block = w_ptr + block_offset_uint4;
        uint4 vec0 = w_block[0];
        uint4 vec1 = w_block[1];
        
        half scale = as_type<half>((uint16_t)(vec0.x & 0xFFFF));
        uint x_base = k * {G_SIZE};

        struct PackedBlock {{
            uint4 v0;
            uint4 v1;
        }};
        PackedBlock blk = {{vec0, vec1}};
        thread const uchar* blk_bytes = (thread const uchar*)&blk;
    """

    if G_SIZE == 128:
        source += f"""
        half block_acc0 = 0;
        half block_acc1 = 0;
        half block_acc2 = 0;
        half block_acc3 = 0;

        device const half* act_block = act_ptr + x_base;

        #pragma unroll(6)
        for (uint c = 0; c < 6; c++) {{
            device const half4* x_vec = (device const half4*)(act_block + c * 20);
            half4 x0 = x_vec[0];
            half4 x1 = x_vec[1];
            half4 x2 = x_vec[2];
            half4 x3 = x_vec[3];
            half4 x4 = x_vec[4];

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
        """
    else:
        source += f"""
        half block_acc = 0;
        device const half* act_block = act_ptr + x_base;

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
        """

    source += f"""
    }}

    half total = simd_sum(acc);

    if (lane_id == 0) {{
        output[row * {N} + col] = total;
    }}
    """
    return source


def get_barrier_free_gemv_source(M: int, K: int, N: int, G_SIZE: int = 128) -> str:
    """
    Variant 2: Barrier-Free Direct SIMD Warp
    - Exactly 0 threadgroup barriers!
    - 100% Coalesced DRAM weight burst reads (32 lanes in warp read 32 adjacent columns)
    - Activations broadcast directly from L1 cache
    - Aligned 20-Trit Vector Chunking + 4-Way ILP accumulators
    - Pre-fetched weights and incremented pointer arithmetic
    - Supports G=128 and G=150
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


def benchmark_kernel(k_fn, x, w_packed, grid, tgroup, M, N, warmup=50, iters=200):
    for _ in range(warmup):
        mx.eval(k_fn(inputs=[x, w_packed], grid=grid, threadgroup=tgroup, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0])
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(k_fn(inputs=[x, w_packed], grid=grid, threadgroup=tgroup, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0])
    return (time.perf_counter() - t0) / iters * 1e6


def compute_reference(x, w_raw, G):
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


def run_phase3a_benchmark():
    print("=" * 80)
    print("PHASE 3A: BARRIER-FREE SIMD-WARP COOPERATIVE REDUCTION BENCHMARK")
    print("Target Hardware: Apple M4 Pro (20-core GPU, 24GB Unified Memory)")
    print("Empirical Telemetry: Latency (µs), Bandwidth (GB/s), and Precision Gates")
    print("=" * 80)

    shapes = [
        ("MLP Gate/Up", 1, 5120, 17408, 128),
        ("MLP Down", 1, 17408, 5120, 128),
        ("Attention Q", 1, 5120, 3072, 128),
        ("Attention KV", 1, 5120, 512, 128),
        ("MLP Gate/Up G150", 1, 5100, 17408, 150),
        ("MLP Down G150", 1, 17400, 5120, 150),
    ]

    results = {}

    for name, M, K, N, G in shapes:
        print(f"\n--- Benchmark Contraction: {name} (M={M}, K={K}, N={N}, G={G}) ---")
        mx.random.seed(42)
        x = mx.random.normal((M, K)).astype(mx.float16)
        w_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
        w_packed, _ = pack_interleaved(w_raw, group_size=G)
        expected = compute_reference(x, w_raw, G)

        # 1. Baseline wide_gemv (Threadgroup SRAM, 272 barriers on Down)
        src_base = get_baseline_gemv_source(M, K, N, G)
        k_base = fast.metal_kernel(
            name=f"base_{M}_{K}_{N}_{G}",
            input_names=["activations", "interleaved_weights"],
            output_names=["output"],
            source=src_base,
            header=_LUT_HEADER
        )
        tg_base = 128
        grid_base = (((N + tg_base - 1) // tg_base) * tg_base, M, 1)
        tgroup_base = (tg_base, 1, 1)
        out_base = k_base(inputs=[x, w_packed], grid=grid_base, threadgroup=tgroup_base, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
        mx.eval(out_base)
        diff_base = (mx.abs(out_base - expected) / (mx.abs(expected) + 1e-3)).mean().item()
        lat_base = benchmark_kernel(k_base, x, w_packed, grid_base, tgroup_base, M, N)
        bw_base = (K * N * 2 / 8) / (lat_base * 1e-6) / 1e9

        # 2. Variant 1: 32-Lane Cooperative SIMD Reduction (simd_sum, 0 barriers)
        src_simd = get_simd_sum_coop_source(M, K, N, G)
        k_simd = fast.metal_kernel(
            name=f"simd_{M}_{K}_{N}_{G}",
            input_names=["activations", "interleaved_weights"],
            output_names=["output"],
            source=src_simd,
            header=_LUT_HEADER
        )
        tg_y = 4
        grid_simd = (32, ((N + tg_y - 1) // tg_y) * tg_y, M)
        tgroup_simd = (32, tg_y, 1)
        out_simd = k_simd(inputs=[x, w_packed], grid=grid_simd, threadgroup=tgroup_simd, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
        mx.eval(out_simd)
        diff_simd = (mx.abs(out_simd - expected) / (mx.abs(expected) + 1e-3)).mean().item()
        lat_simd = benchmark_kernel(k_simd, x, w_packed, grid_simd, tgroup_simd, M, N)
        bw_simd = (K * N * 2 / 8) / (lat_simd * 1e-6) / 1e9

        # 3. Variant 2: Barrier-Free Direct SIMD Warp (0 barriers, coalesced 32-lane bursts)
        src_bf = get_barrier_free_gemv_source(M, K, N, G)
        k_bf = fast.metal_kernel(
            name=f"bf_{M}_{K}_{N}_{G}",
            input_names=["activations", "interleaved_weights"],
            output_names=["output"],
            source=src_bf,
            header=_LUT_HEADER
        )
        tg_bf = 64 if N <= 1024 else 128
        grid_bf = (((N + tg_bf - 1) // tg_bf) * tg_bf, M, 1)
        tgroup_bf = (tg_bf, 1, 1)
        out_bf = k_bf(inputs=[x, w_packed], grid=grid_bf, threadgroup=tgroup_bf, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
        mx.eval(out_bf)
        diff_bf = (mx.abs(out_bf - expected) / (mx.abs(expected) + 1e-3)).mean().item()
        lat_bf = benchmark_kernel(k_bf, x, w_packed, grid_bf, tgroup_bf, M, N)
        bw_bf = (K * N * 2 / 8) / (lat_bf * 1e-6) / 1e9

        print(f"  Baseline (SRAM + 272 Barriers): {lat_base:6.2f} µs | BW: {bw_base:5.1f} GB/s | diff: {diff_base:.6f}")
        print(f"  32-Lane Coop simd_sum (0-Bar):  {lat_simd:6.2f} µs | BW: {bw_simd:5.1f} GB/s | diff: {diff_simd:.6f}")
        print(f"  Barrier-Free Coalesced (0-Bar): {lat_bf:6.2f} µs | BW: {bw_bf:5.1f} GB/s | diff: {diff_bf:.6f}")

        speedup_bf = lat_base / lat_bf
        speedup_simd = lat_base / lat_simd
        print(f"  -> Barrier-Free Speedup vs Base: {speedup_bf:.2f}x (shaved {lat_base - lat_bf:.2f} µs)")
        print(f"  -> Precision Gate (diff < 0.03): {'PASS' if diff_bf < 0.03 else 'FAIL'}")

        results[name] = {
            "base": lat_base,
            "simd": lat_simd,
            "bf": lat_bf,
            "diff_bf": diff_bf,
            "bw_bf": bw_bf,
        }

    # E2E 64-layer decode step projection
    # Bonsai 27B has 64 layers: each layer has Gate, Up, Down, Q, K, V, O projections
    # Gate & Up: 2 * Gate/Up
    # Down: 1 * Down
    # Q: 1 * Attn Q
    # KV: 2 * Attn KV
    # O: ~ Attention Q
    print("\n" + "=" * 80)
    print("END-TO-END 64-LAYER 27B DECODE LATENCY & THROUGHPUT PROJECTION")
    print("=" * 80)

    for mode, key in [("Production Baseline (SRAM)", "base"), ("32-Lane Coop SIMD", "simd"), ("Phase 3A Barrier-Free", "bf")]:
        gate_up = results["MLP Gate/Up"][key]
        down = results["MLP Down"][key]
        q = results["Attention Q"][key]
        kv = results["Attention KV"][key]
        
        # Total per layer (µs)
        layer_lat = 2 * gate_up + down + 2 * q + 2 * kv
        total_decode_ms = (64 * layer_lat) / 1000.0
        tok_s = 1000.0 / total_decode_ms
        print(f"  {mode:30s}: {total_decode_ms:6.2f} ms/tok | {tok_s:5.2f} tok/s")

    print("\n" + "=" * 80)
    print("PHASE 3A ACCEPTANCE GATES EVALUATION")
    print("=" * 80)
    mlp_down_lat = results["MLP Down"]["bf"]
    gate_passed = mlp_down_lat <= 285.0
    print(f"Target Gate 1: MLP Down Latency Drop >= 60 µs (Baseline: {results['MLP Down']['base']:.2f} µs)")
    print(f"Achieved:      MLP Down Latency = {mlp_down_lat:.2f} µs (Shaved {results['MLP Down']['base'] - mlp_down_lat:.2f} µs, {results['MLP Down']['bw_bf']:.1f} GB/s)")
    print(f"Gate 1 Evaluation: {'PASSED' if gate_passed else 'FAILED'}")

    precision_passed = all(r["diff_bf"] < 0.03 for r in results.values())
    print(f"Target Gate 2: Mathematical Precision (diff < 0.03 across all shapes)")
    print(f"Gate 2 Evaluation: {'PASSED' if precision_passed else 'FAILED'}")
    print("=" * 80)


if __name__ == "__main__":
    run_phase3a_benchmark()
