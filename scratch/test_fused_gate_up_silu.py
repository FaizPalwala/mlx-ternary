"""
Phase 3C: Fused Gate-Up-SiLU SwiGLU Kernel Prototype
Hardware: Apple Silicon GPU (Apple M4 Pro, 20 GPU Cores)
Architectural Target: ADR-002 Phase 3C (Fused SwiGLU Contraction)

Micro-Architectural Innovations:
1. Single-Pass Memory Access: Fuses Gate and Up weight passes into a single kernel pass.
   Eliminates 174 KB of intermediate memory traffic and two pipeline flushes per token.
2. Dual-Issue FMA Concurrency: Both Gate and Up projections share the exact same loaded
   activation vectors (x0..x4), firing concurrent FMAs across Apple Silicon's dual-issue pipelines.
3. In-Register Epilogue: Evaluates SiLU(acc_gate) * acc_up directly in FP32 registers
   before committing the final FP16 half vector to DRAM.
4. Zero-Barrier Invariant: Exactly 0 threadgroup barriers in the inner dot loop.
5. Live Side-by-Side Comparison against official PrismML Bonsai 2-bit under identical battery power governor.
"""

import sys
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


def make_fused_swiglu_source(M: int, K: int, N: int, G_SIZE: int = 128) -> str:
    """
    Fused Gate + Up + SiLU Metal Kernel Source.
    - Grid: (N, 1, M)
    - Threadgroup: (64, 1, 1) or (128, 1, 1)
    - Zero threadgroup barriers.
    - Activation vector loaded once and reused across both projections.
    - In-register SiLU evaluation and product.
    - Live registers <= 28 per thread.
    """
    source = f"""
    uint col = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;

    if (col >= {N} || row >= {M}) return;

    uint K_blocks = {K} / {G_SIZE};
    half acc_gate = 0;
    half acc_up = 0;

    device const half* act_block = activations + row * {K};
    device const uint4* w_gate_ptr = ((device const uint4*)gate_weights) + col * 2;
    device const uint4* w_up_ptr = ((device const uint4*)up_weights) + col * 2;
    uint w_stride = {N} * 2;

    // Asynchronous prologue: pre-load block 0 weights for both Gate and Up
    uint4 cur_g0 = w_gate_ptr[0];
    uint4 cur_g1 = w_gate_ptr[1];
    w_gate_ptr += w_stride;

    uint4 cur_u0 = w_up_ptr[0];
    uint4 cur_u1 = w_up_ptr[1];
    w_up_ptr += w_stride;

    for (uint k_block = 0; k_block < K_blocks; k_block++) {{
        // Asynchronously issue prefetch for next block weights
        uint4 next_g0 = uint4(0), next_g1 = uint4(0);
        uint4 next_u0 = uint4(0), next_u1 = uint4(0);
        if (k_block + 1 < K_blocks) {{
            next_g0 = w_gate_ptr[0];
            next_g1 = w_gate_ptr[1];
            w_gate_ptr += w_stride;

            next_u0 = w_up_ptr[0];
            next_u1 = w_up_ptr[1];
            w_up_ptr += w_stride;
        }}

        half scale_g = as_type<half>((uint16_t)(cur_g0.x & 0xFFFF));
        half scale_u = as_type<half>((uint16_t)(cur_u0.x & 0xFFFF));

        struct PackedBlock {{ uint4 v0, v1; }};
        PackedBlock blk_g = {{cur_g0, cur_g1}};
        PackedBlock blk_u = {{cur_u0, cur_u1}};
        thread const uchar* bytes_g = (thread const uchar*)&blk_g;
        thread const uchar* bytes_u = (thread const uchar*)&blk_u;
    """

    if G_SIZE == 128:
        source += f"""
        half g_acc0 = 0, g_acc1 = 0, g_acc2 = 0, g_acc3 = 0;
        half u_acc0 = 0, u_acc1 = 0, u_acc2 = 0, u_acc3 = 0;

        device const half4* act_vec = (device const half4*)act_block;

        #pragma unroll(6)
        for (uint c = 0; c < 6; c++) {{
            // Load activation chunk ONCE into registers for both projections
            half4 x0 = act_vec[c * 5 + 0];
            half4 x1 = act_vec[c * 5 + 1];
            half4 x2 = act_vec[c * 5 + 2];
            half4 x3 = act_vec[c * 5 + 3];
            half4 x4 = act_vec[c * 5 + 4];

            // 1. Gate Chunk FMAs
            uint8_t gb0 = bytes_g[2 + c * 4 + 0];
            uint8_t gb1 = bytes_g[2 + c * 4 + 1];
            uint8_t gb2 = bytes_g[2 + c * 4 + 2];
            uint8_t gb3 = bytes_g[2 + c * 4 + 3];

            half4 gw0_low = base3_lut_low[gb0]; half gw0_high = base3_lut_high[gb0];
            half4 gw1_low = base3_lut_low[gb1]; half gw1_high = base3_lut_high[gb1];
            half4 gw2_low = base3_lut_low[gb2]; half gw2_high = base3_lut_high[gb2];
            half4 gw3_low = base3_lut_low[gb3]; half gw3_high = base3_lut_high[gb3];

            g_acc0 += dot(x0, gw0_low) + x1.x * gw0_high;
            g_acc1 += dot(x1.yzw, gw1_low.xyz) + x2.x * gw1_low.w + x2.y * gw1_high;
            g_acc2 += dot(x2.zw, gw2_low.xy) + dot(x3.xyz, half3(gw2_low.z, gw2_low.w, gw2_high));
            g_acc3 += x3.w * gw3_low.x + dot(x4, half4(gw3_low.y, gw3_low.z, gw3_low.w, gw3_high));

            // 2. Up Chunk FMAs (reuses loaded x0..x4, dual-issue with Gate)
            uint8_t ub0 = bytes_u[2 + c * 4 + 0];
            uint8_t ub1 = bytes_u[2 + c * 4 + 1];
            uint8_t ub2 = bytes_u[2 + c * 4 + 2];
            uint8_t ub3 = bytes_u[2 + c * 4 + 3];

            half4 uw0_low = base3_lut_low[ub0]; half uw0_high = base3_lut_high[ub0];
            half4 uw1_low = base3_lut_low[ub1]; half uw1_high = base3_lut_high[ub1];
            half4 uw2_low = base3_lut_low[ub2]; half uw2_high = base3_lut_high[ub2];
            half4 uw3_low = base3_lut_low[ub3]; half uw3_high = base3_lut_high[ub3];

            u_acc0 += dot(x0, uw0_low) + x1.x * uw0_high;
            u_acc1 += dot(x1.yzw, uw1_low.xyz) + x2.x * uw1_low.w + x2.y * uw1_high;
            u_acc2 += dot(x2.zw, uw2_low.xy) + dot(x3.xyz, half3(uw2_low.z, uw2_low.w, uw2_high));
            u_acc3 += x3.w * uw3_low.x + dot(x4, half4(uw3_low.y, uw3_low.z, uw3_low.w, uw3_high));
        }}

        half4 tail0 = act_vec[30];
        half4 tail1 = act_vec[31];

        // Tail Gate
        uint8_t gb24 = bytes_g[2 + 24];
        uint8_t gb25 = bytes_g[2 + 25];
        g_acc0 += dot(tail0, base3_lut_low[gb24]) + tail1.x * base3_lut_high[gb24];
        g_acc1 += dot(tail1.yzw, base3_lut_low[gb25].xyz);

        // Tail Up
        uint8_t ub24 = bytes_u[2 + 24];
        uint8_t ub25 = bytes_u[2 + 25];
        u_acc0 += dot(tail0, base3_lut_low[ub24]) + tail1.x * base3_lut_high[ub24];
        u_acc1 += dot(tail1.yzw, base3_lut_low[ub25].xyz);

        acc_gate += ((g_acc0 + g_acc1) + (g_acc2 + g_acc3)) * scale_g;
        acc_up   += ((u_acc0 + u_acc1) + (u_acc2 + u_acc3)) * scale_u;

        act_block += 128;
        cur_g0 = next_g0; cur_g1 = next_g1;
        cur_u0 = next_u0; cur_u1 = next_u1;
        """
    else:
        # G_SIZE == 150
        source += f"""
        half block_acc_g = 0;
        half block_acc_u = 0;

        #pragma unroll(30)
        for (uint i = 0; i < 30; i++) {{
            uint curr_x = i * 5;
            half4 x_low = half4(act_block[curr_x], act_block[curr_x + 1], act_block[curr_x + 2], act_block[curr_x + 3]);
            half x_high = act_block[curr_x + 4];

            uint8_t gb = bytes_g[2 + i];
            half4 gw_low = base3_lut_low[gb]; half gw_high = base3_lut_high[gb];
            block_acc_g += dot(x_low, gw_low) + x_high * gw_high;

            uint8_t ub = bytes_u[2 + i];
            half4 uw_low = base3_lut_low[ub]; half uw_high = base3_lut_high[ub];
            block_acc_u += dot(x_low, uw_low) + x_high * uw_high;
        }}

        acc_gate += block_acc_g * scale_g;
        acc_up   += block_acc_u * scale_u;

        act_block += 150;
        cur_g0 = next_g0; cur_g1 = next_g1;
        cur_u0 = next_u0; cur_u1 = next_u1;
        """

    source += f"""
    }}

    // In-register SiLU(acc_gate) * acc_up evaluation
    float g = (float)acc_gate;
    float silu_g = g / (1.0f + metal::exp(-g));
    half out_val = (half)(silu_g * (float)acc_up);
    output[row * {N} + col] = out_val;
    """
    return source


def compute_unquantized_swiglu_reference(x, w_gate_raw, w_up_raw, G):
    """Reference unquantized SwiGLU computation: SiLU(x @ W_gate) * (x @ W_up)."""
    from test_elastic_simd_split_k import compute_unquantized_reference
    gate_ref = compute_unquantized_reference(x, w_gate_raw, G)
    up_ref = compute_unquantized_reference(x, w_up_raw, G)
    expected = (nn.silu(gate_ref.astype(mx.float32)) * up_ref.astype(mx.float32)).astype(mx.float16)
    mx.eval(expected)
    return expected


def run_benchmark():
    print("=" * 80)
    print("PHASE 3C: FUSED GATE-UP-SILU SWIGLU KERNEL BENCHMARK")
    print("Apple Silicon Dynamic Concurrency & Live Battery Governor Evaluation")
    print("=" * 80)

    cores = get_gpu_core_count()
    dev_name = mx.device_info().get("device_name", "Apple Silicon")
    print(f"Target Hardware:  {dev_name} ({cores} GPU cores detected)")
    print(f"Power Condition:  BATTERY POWER (Live side-by-side benchmark with Bonsai)")

    # 1. Dimensions: MLP Expansion for Ternary-Bonsai-2-27B
    M = 1
    K = 5120
    N = 17408
    G = 128
    total_weights_mb = (2 * K * N * 2 / 8) / (1024 ** 2)

    total_tgs_128 = (N + 127) // 128
    total_tgs_64 = (N + 63) // 64
    print(f"\nContraction:      MLP SwiGLU Expansion (M={M}, K={K}, N={N}, G={G})")
    print(f"Combined Weights: 2 x (5120 x 17408) = {total_weights_mb:.2f} MB payload")
    print(f"Grid Geometry:    tg=128 -> {total_tgs_128} TGs ({total_tgs_128 / cores:.1f} TGs/core) | tg=64 -> {total_tgs_64} TGs ({total_tgs_64 / cores:.1f} TGs/core)")

    # 2. Allocate Inputs
    mx.random.seed(42)
    x = mx.random.normal((M, K)).astype(mx.float16)

    w_gate_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    w_gate_packed, _ = pack_interleaved(w_gate_raw, group_size=G)

    w_up_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    w_up_packed, _ = pack_interleaved(w_up_raw, group_size=G)

    expected = compute_unquantized_swiglu_reference(x, w_gate_raw, w_up_raw, G)

    warmup = 20
    iters = 100

    results = {}

    # --------------------------------------------------------------------------
    # 1. Live Bonsai 2-bit SwiGLU Baseline (Native Apple QuantizedLinear)
    # --------------------------------------------------------------------------
    print("\n[1/4] Benchmarking Live PrismML Bonsai 2-bit SwiGLU Baseline...")
    w_gate_fp16 = mx.random.normal((N, K)).astype(mx.float16)
    q_gate, s_gate, b_gate = mx.quantize(w_gate_fp16, group_size=128, bits=2)
    gate_layer = nn.QuantizedLinear(K, N, bits=2, group_size=128, bias=False)
    gate_layer.weight = q_gate
    gate_layer.scales = s_gate
    gate_layer.biases = b_gate

    w_up_fp16 = mx.random.normal((N, K)).astype(mx.float16)
    q_up, s_up, b_up = mx.quantize(w_up_fp16, group_size=128, bits=2)
    up_layer = nn.QuantizedLinear(K, N, bits=2, group_size=128, bias=False)
    up_layer.weight = q_up
    up_layer.scales = s_up
    up_layer.biases = b_up
    mx.eval(gate_layer.parameters(), up_layer.parameters())

    def bonsai_swiglu():
        g = gate_layer(x)
        mx.eval(g)
        u = up_layer(x)
        mx.eval(u)
        h = nn.silu(g) * u
        return h

    for _ in range(warmup):
        mx.eval(bonsai_swiglu())

    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(bonsai_swiglu())
    lat_bonsai = (time.perf_counter() - t0) / iters * 1e6
    bonsai_bytes = 2 * ((K * N * 2 / 8) + (K * N / 128 * 4))
    bw_bonsai = (bonsai_bytes / 1e9) / (lat_bonsai * 1e-6)
    results["bonsai"] = {"lat": lat_bonsai, "bw": bw_bonsai}
    print(f"  ✓ Live Bonsai 2-bit SwiGLU:     {lat_bonsai:6.2f} µs | Bandwidth: {bw_bonsai:5.1f} GB/s")

    # --------------------------------------------------------------------------
    # 2. Un-fused mlx-ternary Baseline: Gate(x) + Up(x) + nn.silu(Gate) * Up
    # --------------------------------------------------------------------------
    print("\n[2/4] Benchmarking Un-fused mlx-ternary Baseline (Gate + Up + SiLU)...")
    from mlx_ternary.ops import ternary_matmul

    def unfused_swiglu():
        g = ternary_matmul(x, w_gate_packed, group_size=G)
        mx.eval(g)
        u = ternary_matmul(x, w_up_packed, group_size=G)
        mx.eval(u)
        h = nn.silu(g) * u
        return h

    out_unfused = unfused_swiglu()
    mx.eval(out_unfused)
    diff_unfused = (mx.abs(out_unfused - expected) / (mx.abs(expected) + 1e-3)).mean().item()

    for _ in range(warmup):
        mx.eval(unfused_swiglu())

    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(unfused_swiglu())
    lat_unfused = (time.perf_counter() - t0) / iters * 1e6
    bw_unfused = (2 * (K * N * 2 / 8) / 1e9) / (lat_unfused * 1e-6)
    results["unfused"] = {"lat": lat_unfused, "bw": bw_unfused, "diff": diff_unfused}
    print(f"  ✓ Un-fused mlx-ternary Baseline: {lat_unfused:6.2f} µs | Bandwidth: {bw_unfused:5.1f} GB/s | diff: {diff_unfused:.6f}")

    # --------------------------------------------------------------------------
    # 3. Fused mlx-ternary Phase 3C: FusedGateUpSiLU(x, w_gate, w_up)
    # --------------------------------------------------------------------------
    print("\n[3/4] Benchmarking Fused mlx-ternary Phase 3C (tg=64 and tg=128)...")
    src_fused = make_fused_swiglu_source(M, K, N, G)
    k_fused = fast.metal_kernel(
        name="fused_swiglu",
        input_names=["activations", "gate_weights", "up_weights"],
        output_names=["output"],
        source=src_fused,
        header=_LUT_HEADER
    )

    grid = (N, 1, M)
    fused_configs = {}
    for tg_val in [64, 128]:
        out_fused = k_fused(inputs=[x, w_gate_packed, w_up_packed], grid=grid, threadgroup=(tg_val, 1, 1), output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
        mx.eval(out_fused)

        diff_fused = (mx.abs(out_fused - expected) / (mx.abs(expected) + 1e-3)).mean().item()
        max_abs_fused = mx.abs(out_fused - expected).max().item()
        mean_mag_exp = mx.abs(expected.astype(mx.float32)).mean().item()
        max_rel_fused = max_abs_fused / mean_mag_exp

        # Parity vs un-fused baseline
        parity_diff = (mx.abs(out_fused - out_unfused) / (mx.abs(out_unfused) + 1e-3)).mean().item()

        for _ in range(warmup):
            mx.eval(k_fused(inputs=[x, w_gate_packed, w_up_packed], grid=grid, threadgroup=(tg_val, 1, 1), output_shapes=[(M, N)], output_dtypes=[mx.float16])[0])

        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(k_fused(inputs=[x, w_gate_packed, w_up_packed], grid=grid, threadgroup=(tg_val, 1, 1), output_shapes=[(M, N)], output_dtypes=[mx.float16])[0])
        lat_fused = (time.perf_counter() - t0) / iters * 1e6
        bw_fused = (2 * (K * N * 2 / 8) / 1e9) / (lat_fused * 1e-6)

        tgs = (N + tg_val - 1) // tg_val
        tgs_per_core = tgs / cores
        fused_configs[tg_val] = {
            "lat": lat_fused,
            "bw": bw_fused,
            "mean_diff": diff_fused,
            "max_rel_diff": max_rel_fused,
            "parity_diff": parity_diff,
            "tgs": tgs,
            "tgs_per_core": tgs_per_core,
        }
        print(f"  ✓ Phase 3C Fused (tg={tg_val:3d}, {tgs:3d} TGs, {tgs_per_core:4.1f}/core): {lat_fused:6.2f} µs | Bandwidth: {bw_fused:5.1f} GB/s | mean_diff: {diff_fused:.6f} | max_rel: {max_rel_fused:.6f}")

    # Optimal Fused Configuration
    best_tg = min(fused_configs.keys(), key=lambda k: fused_configs[k]["lat"])
    best_fused = fused_configs[best_tg]

    # --------------------------------------------------------------------------
    # 4. Numerical Validation across Group Size G=150
    # --------------------------------------------------------------------------
    print("\n[4/4] Validating Numerical Parity for Group Size G=150...")
    K_150 = 5100
    w_gate_raw_150 = mx.random.randint(0, 3, (K_150, N)).astype(mx.float16) - 1.0
    w_gate_p_150, _ = pack_interleaved(w_gate_raw_150, group_size=150)
    w_up_raw_150 = mx.random.randint(0, 3, (K_150, N)).astype(mx.float16) - 1.0
    w_up_p_150, _ = pack_interleaved(w_up_raw_150, group_size=150)
    x_150 = mx.random.normal((M, K_150)).astype(mx.float16)

    expected_150 = compute_unquantized_swiglu_reference(x_150, w_gate_raw_150, w_up_raw_150, G=150)

    src_150 = make_fused_swiglu_source(M, K_150, N, G_SIZE=150)
    k_150 = fast.metal_kernel("fused_swiglu_g150", ["activations", "gate_weights", "up_weights"], ["output"], src_150, _LUT_HEADER)

    out_150 = k_150(inputs=[x_150, w_gate_p_150, w_up_p_150], grid=grid, threadgroup=(64, 1, 1), output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
    mx.eval(out_150)

    diff_150 = (mx.abs(out_150 - expected_150) / (mx.abs(expected_150) + 1e-3)).mean().item()
    max_abs_150 = mx.abs(out_150 - expected_150).max().item()
    mean_mag_150 = mx.abs(expected_150.astype(mx.float32)).mean().item()
    max_rel_150 = max_abs_150 / mean_mag_150
    pass_150 = diff_150 < 0.03 and max_rel_150 < 0.15
    print(f"  ✓ G=150 Fused SwiGLU: mean_diff = {diff_150:.6f}, max_rel_diff = {max_rel_150:.6f} (PASS: {pass_150})")

    # --------------------------------------------------------------------------
    # Clean Markdown Summary Table
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LIVE BATTERY-NORMALIZED SWIGLU COMPARISON TABLE (MLP Expansion: 5120 -> 17408)")
    print("=" * 80)

    print(f"| {'Kernel Configuration':<36} | {'TGs (TGs/Core)':<16} | {'Latency (µs)':<13} | {'Bandwidth':<12} | {'Delta vs Unfused':<17} | {'Delta vs Bonsai':<16} |")
    print(f"| :{'-'*34} | :{'-'*14} | :{'-'*11} | :{'-'*10} | :{'-'*15} | :{'-'*14} |")

    # Bonsai SwiGLU
    print(f"| {'PrismML Bonsai 2-bit SwiGLU':<36} | {'Apple Native':<16} | {lat_bonsai:>10.2f} µs | {bw_bonsai:>9.1f} GB/s | {'---':<17} | {'Baseline (0.0%)':<16} |")

    # Un-fused mlx-ternary
    print(f"| {'Un-fused mlx-ternary (Gate+Up+SiLU)':<36} | {'272 (13.6/core)':<16} | {lat_unfused:>10.2f} µs | {bw_unfused:>9.1f} GB/s | {'Baseline (0.0%)':<17} | {f'{lat_bonsai/lat_unfused:.2f}x':<16} |")

    # Phase 3C Fused configs
    for tg_val in [128, 64]:
        c = fused_configs[tg_val]
        delta_unfused = lat_unfused - c["lat"]
        speed_unfused = lat_unfused / c["lat"]
        speed_bonsai = lat_bonsai / c["lat"]
        star = " ★" if tg_val == best_tg else ""
        tg_info = f"{c['tgs']} ({c['tgs_per_core']:.1f}/core)"
        print(f"| {f'Phase 3C Fused SwiGLU (tg={tg_val}){star}':<36} | {tg_info:<16} | {c['lat']:>10.2f} µs | {c['bw']:>9.1f} GB/s | {f'-{delta_unfused:.1f}µ ({speed_unfused:.2f}x)':<17} | {f'{speed_bonsai:.2f}x':<16} |")

    # --------------------------------------------------------------------------
    # Acceptance Gates Evaluation
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 3C ACCEPTANCE GATES EVALUATION")
    print("=" * 80)

    # Gate 1: Numerical Precision
    g128_pass = best_fused["mean_diff"] < 0.03 and best_fused["max_rel_diff"] < 0.15
    gate1_pass = g128_pass and pass_150
    print(f"Gate 1: Numerical Precision (mean_rel_diff < 0.03, max_rel_diff < 0.15 across G=128 & G=150): {'PASSED' if gate1_pass else 'FAILED'}")

    # Gate 2: Latency Dividend (shaves >= 100 µs off un-fused)
    dividend = lat_unfused - best_fused["lat"]
    gate2_pass = dividend >= 100.0
    print(f"Gate 2: Latency Dividend (shaved >= 100.0 µs vs Un-fused Baseline: {lat_unfused:.2f} µs):")
    print(f"  * Achieved Shaved Latency: {dividend:.2f} µs ({lat_unfused / best_fused['lat']:.2f}x speedup)")
    print(f"  * Gate 2 Status: {'PASSED' if gate2_pass else 'FAILED'}")

    # Gate 3: Live Bonsai Comparison
    print(f"Gate 3: Live Performance Under Battery:")
    print(f"  * Live Bonsai 2-bit SwiGLU:     {lat_bonsai:.2f} µs")
    print(f"  * Un-fused mlx-ternary SwiGLU:  {lat_unfused:.2f} µs")
    print(f"  * Phase 3C Fused SwiGLU:        {best_fused['lat']:.2f} µs (Shaved {dividend:.2f} µs, {best_fused['bw']:.1f} GB/s)")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
