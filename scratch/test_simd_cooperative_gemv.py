"""
32-Lane Cooperative SIMD Reduction GEMV Benchmark (Phase 2A Prototype)
Hardware: Apple M4 Pro (20-core GPU, 24GB Unified Memory, 273 GB/s peak LPDDR5 bandwidth)
Architectural Target: ADR-001 Phase 2A (SIMD Warp Restoration)

This prototype restores intra-warp cooperative K-reduction via hardware `simd_sum()`,
replacing 1-thread-per-column stride dispatch with a 32-lane warp per column.
"""

import sys
import time
from pathlib import Path

# Add src to path so this script can run standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mlx.core as mx
import mlx.core.fast as fast
import mlx_ternary
from mlx_ternary.quantize import pack_interleaved
from mlx_ternary.ops import _LUT_HEADER

# Kernel Cache for compiled cooperative kernels
_COOP_KERNEL_CACHE = {}


def get_cooperative_gemv_source(M: int, K: int, N: int, G_SIZE: int) -> str:
    """
    Metal Shading Language source for 32-lane intra-warp cooperative GEMV.
    - Grid: (32, N, M)
    - Threadgroup: (32, 1, 1) -> 1 SIMDgroup (warp) per output column.
    - Lanes 0..31 cooperatively reduce K_blocks.
    - simd_sum() performs single-cycle warp reduction.
    - Only lane 0 commits to output DRAM.
    """
    num_bytes = 26 if G_SIZE == 128 else 30

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

    for (uint k = start_k; k < end_k; k++) {{
        uint block_offset_uint4 = (k * {N} + col) * 2;
        device const uint4* w_ptr = (device const uint4*)interleaved_weights;
        uint4 vec0 = w_ptr[block_offset_uint4];
        uint4 vec1 = w_ptr[block_offset_uint4 + 1];
        
        half scale = as_type<half>((uint16_t)(vec0.x & 0xFFFF));
        half block_acc = 0;
        uint x_base = k * {G_SIZE};

        struct PackedBlock {{
            uint4 v0;
            uint4 v1;
        }};
        PackedBlock blk = {{vec0, vec1}};
        thread const uchar* blk_bytes = (thread const uchar*)&blk;

        #pragma unroll({num_bytes})
        for (uint i = 0; i < {num_bytes}; i++) {{
            uint8_t b = blk_bytes[2 + i];
            half4 w_low = base3_lut_low[b];
            half w_high = base3_lut_high[b];
            uint curr_x = x_base + i * 5;
    """

    if G_SIZE == 150:
        source += """
            half4 x_low = half4(act_ptr[curr_x], act_ptr[curr_x + 1], act_ptr[curr_x + 2], act_ptr[curr_x + 3]);
            half x_high = act_ptr[curr_x + 4];
            block_acc += dot(x_low, w_low) + x_high * w_high;
        """
    else:
        source += """
            if (i == 25) {
                block_acc += act_ptr[curr_x] * w_low.x + act_ptr[curr_x + 1] * w_low.y + act_ptr[curr_x + 2] * w_low.z;
            } else {
                half4 x_low = half4(act_ptr[curr_x], act_ptr[curr_x + 1], act_ptr[curr_x + 2], act_ptr[curr_x + 3]);
                half x_high = act_ptr[curr_x + 4];
                block_acc += dot(x_low, w_low) + x_high * w_high;
            }
        """

    source += f"""
        }}
        acc += block_acc * scale;
    }}

    half total = simd_sum(acc);

    if (lane_id == 0) {{
        output[row * {N} + col] = total;
    }}
    """
    return source


def get_cooperative_gemv_kernel(M: int, K: int, N: int, G_SIZE: int):
    shape_key = (M, K, N, G_SIZE)
    if shape_key not in _COOP_KERNEL_CACHE:
        source = get_cooperative_gemv_source(M, K, N, G_SIZE)
        kernel = fast.metal_kernel(
            name=f"coop_gemv_{M}_{K}_{N}_{G_SIZE}",
            input_names=["activations", "interleaved_weights"],
            output_names=["output"],
            source=source,
            header=_LUT_HEADER
        )
        _COOP_KERNEL_CACHE[shape_key] = kernel
    return _COOP_KERNEL_CACHE[shape_key]


def cooperative_ternary_gemv(x: mx.array, w_packed: mx.array, group_size: int = 128) -> mx.array:
    """
    Dispatch cooperative 32-lane GEMV kernel.
    """
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])
    if x_flat.dtype != mx.float16:
        x_flat = x_flat.astype(mx.float16)

    M = x_flat.shape[0]
    K = x_flat.shape[1]
    N = w_packed.shape[1]

    kernel = get_cooperative_gemv_kernel(M, K, N, group_size)
    tg_y = 4
    grid_y = ((N + tg_y - 1) // tg_y) * tg_y
    grid = (32, grid_y, M)
    threadgroup = (32, tg_y, 1)

    out = kernel(
        inputs=[x_flat, w_packed],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(M, N)],
        output_dtypes=[mx.float16]
    )

    res = out[0]
    if len(orig_shape) > 2:
        return res.reshape(*orig_shape[:-1], N)
    elif len(orig_shape) == 1:
        return res.reshape(N)
    return res


def verify_mathematical_precision(M: int = 1, K: int = 5120, N: int = 1024, group_size: int = 128):
    """
    Strict validation against reference unpacked matmul.
    Guardrail: relative error threshold < 0.03 for FP16 accumulation.
    """
    print(f"\n--- Verifying Mathematical Precision (M={M}, K={K}, N={N}, G={group_size}) ---")
    x = mx.random.normal((M, K)).astype(mx.float16)
    w_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    w_interleaved, _ = pack_interleaved(w_raw, group_size=group_size)

    K_blocks = K // group_size
    w_grouped = w_raw.reshape(K_blocks, group_size, N)
    scales = mx.abs(w_grouped).max(axis=1).astype(mx.float16)
    safe_scales = mx.where(scales == 0, 1.0, scales)
    w_norm = w_grouped / safe_scales[:, None, :]
    w_ternary = mx.where(w_norm > 0.33, 1, mx.where(w_norm < -0.33, -1, 0))
    w_reconstructed = (w_ternary * scales[:, None, :]).reshape(K, N)

    expected = mx.matmul(x, w_reconstructed)
    actual = cooperative_ternary_gemv(x, w_interleaved, group_size=group_size)
    mx.eval(expected, actual)

    rel_diff = (mx.abs(actual - expected) / (mx.abs(expected) + 1e-3)).mean().item()
    max_diff = mx.abs(actual - expected).max().item()

    print(f"Max Absolute Diff: {max_diff:.4f}")
    print(f"Mean Relative Diff: {rel_diff:.4f} (Threshold: < 0.03)")

    assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"
    assert rel_diff < 0.03, f"Mathematical precision threshold violated! {rel_diff:.4f} >= 0.03"
    print(">> Precision Gate: PASSED")


def benchmark_layer(name: str, M: int, K: int, N: int, group_size: int = 128, iters: int = 50, run001_baseline_us: float = None):
    bytes_ternary = (K // group_size) * N * 32

    x = mx.random.normal((M, K)).astype(mx.float16)
    w_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    w_interleaved, _ = pack_interleaved(w_raw, group_size=group_size)

    # 1. Benchmark Current JIT
    for _ in range(5):
        mx.eval(mlx_ternary.ternary_matmul(x, w_interleaved, group_size=group_size))

    start = time.perf_counter()
    for _ in range(iters):
        mx.eval(mlx_ternary.ternary_matmul(x, w_interleaved, group_size=group_size))
    t_current = (time.perf_counter() - start) / iters

    # 2. Benchmark Cooperative SIMD
    for _ in range(5):
        mx.eval(cooperative_ternary_gemv(x, w_interleaved, group_size=group_size))

    start = time.perf_counter()
    for _ in range(iters):
        mx.eval(cooperative_ternary_gemv(x, w_interleaved, group_size=group_size))
    t_coop = (time.perf_counter() - start) / iters

    bw_current = (bytes_ternary / 1e9) / t_current
    bw_coop = (bytes_ternary / 1e9) / t_coop

    t_coop_us = t_coop * 1e6
    t_curr_us = t_current * 1e6

    print(f"\n[{name}] Shape: M={M}, K={K}, N={N} | Data: {bytes_ternary / 1024 / 1024:.2f} MB")
    if run001_baseline_us is not None:
        speedup_run001 = run001_baseline_us / t_coop_us
        print(f"  * Run 001 Baseline:      {run001_baseline_us:8.2f} us")
        print(f"  * Current JIT:           {t_curr_us:8.2f} us | BW: {bw_current:5.2f} GB/s")
        print(f"  * Cooperative SIMD Warp: {t_coop_us:8.2f} us | BW: {bw_coop:5.2f} GB/s")
        print(f"  * Speedup vs Run 001:    {speedup_run001:5.2f}x ({(speedup_run001 - 1) * 100:+.1f}%)")
    else:
        print(f"  * Current JIT:           {t_curr_us:8.2f} us | BW: {bw_current:5.2f} GB/s")
        print(f"  * Cooperative SIMD Warp: {t_coop_us:8.2f} us | BW: {bw_coop:5.2f} GB/s")

    return t_current, t_coop


def main():
    print("=" * 80)
    print("PHASE 2A: 32-LANE COOPERATIVE SIMD REDUCTION BENCHMARK")
    print("Target Hardware: Apple M4 Pro (20-core GPU, 24GB Unified Memory)")
    print("Testing: 32-lane cooperative simd_sum vs Run 001 Baseline")
    print("=" * 80)

    # 1. Guardrail Verification
    verify_mathematical_precision(M=1, K=256, N=128, group_size=128)
    verify_mathematical_precision(M=1, K=300, N=128, group_size=150)
    verify_mathematical_precision(M=1, K=5120, N=1024, group_size=128)

    # 2. Benchmark Full Bonsai-2-27B Layer Suite
    # Baseline telemetry from ADR-001 (Run 001, commit 70008a9 on M4 Pro)
    run001_baselines = {
        "MLP Gate/Up Proj": 773.70,
        "MLP Down Proj": 305.46,
        "Attention Q Proj": 189.54,
        "Attention KV Proj": 161.33,
        "Attention O Proj": 165.83,
    }

    layers = [
        ("MLP Gate/Up Proj", 1, 5120, 17408, 128),
        ("MLP Down Proj",    1, 17408, 5120, 128),
        ("Attention Q Proj", 1, 5120, 3072,  128),
        ("Attention KV Proj",1, 5120, 512,   128),
        ("Attention O Proj", 1, 3072, 5120,  128),
    ]

    layer_multipliers = {
        "MLP Gate/Up Proj": 2,
        "MLP Down Proj": 1,
        "Attention Q Proj": 1,
        "Attention KV Proj": 2,
        "Attention O Proj": 1,
    }

    total_time_current_layer = 0.0
    total_time_coop_layer = 0.0
    total_time_run001_layer = 0.0

    telemetry_results = {}

    for name, M, K, N, G in layers:
        base_us = run001_baselines.get(name)
        t_curr, t_coop = benchmark_layer(name, M, K, N, group_size=G, iters=50, run001_baseline_us=base_us)
        mult = layer_multipliers[name]
        total_time_current_layer += t_curr * mult
        total_time_coop_layer += t_coop * mult
        if base_us:
            total_time_run001_layer += (base_us * 1e-6) * mult
        telemetry_results[name] = {
            "run001_us": base_us,
            "current_us": t_curr * 1e6,
            "coop_us": t_coop * 1e6,
            "coop_bw": ((K // G) * N * 32 / 1e9) / t_coop
        }

    num_layers = 64
    full_step_current_ms = total_time_current_layer * num_layers * 1000
    full_step_coop_ms = total_time_coop_layer * num_layers * 1000
    full_step_run001_ms = total_time_run001_layer * num_layers * 1000

    tok_sec_current = 1000.0 / full_step_current_ms
    tok_sec_coop = 1000.0 / full_step_coop_ms
    tok_sec_run001 = 1000.0 / full_step_run001_ms

    print("\n" + "=" * 80)
    print("END-TO-END 27B SINGLE-TOKEN PROJECTION (64 LAYERS)")
    print("=" * 80)
    print(f"Run 001 Baseline Step:     {full_step_run001_ms:6.2f} ms | ~{tok_sec_run001:5.1f} tok/s")
    print(f"Current JIT Step:          {full_step_current_ms:6.2f} ms | ~{tok_sec_current:5.1f} tok/s")
    print(f"Cooperative SIMD Warp Step:{full_step_coop_ms:6.2f} ms | ~{tok_sec_coop:5.1f} tok/s")
    print(f"Speedup vs Run 001:        {tok_sec_coop / tok_sec_run001:5.2f}x ({(tok_sec_coop / tok_sec_run001 - 1) * 100:+.1f}%)")

    # Phase 2A Acceptance Gate Check
    gate_target_us = 680.0
    gate_actual_us = telemetry_results["MLP Gate/Up Proj"]["coop_us"]
    print("\n" + "=" * 80)
    print("PHASE 2A ACCEPTANCE GATE EVALUATION")
    print("=" * 80)
    print(f"Target Gate: MLP Gate/Up latency <= {gate_target_us:.1f} us (Run 001: 773.70 us)")
    print(f"Achieved:    MLP Gate/Up latency  = {gate_actual_us:.2f} us")
    if gate_actual_us <= gate_target_us:
        print(f">> GATE PASSED: Latency dropped by {773.70 - gate_actual_us:.2f} us ({((773.70 - gate_actual_us) / 773.70) * 100:.1f}%)!")
    else:
        print(f">> GATE FAILED: Latency {gate_actual_us:.2f} us > {gate_target_us:.1f} us threshold.")
    print("=" * 80)


if __name__ == "__main__":
    main()
