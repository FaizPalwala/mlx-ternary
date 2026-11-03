#!/usr/bin/env python3
"""
Phase 2F (ALU Optimization Levers) Micro-Benchmark.

Evaluates atomic architectural levers to optimize the inner loop of ternary GEMV:
- Variant 0: Production Baseline (Constant Cache LUT, 1 Accumulator)
- Variant 2: Lever 3 (4-Way ILP Accumulator Splitting: breaking FP16 FMA RAW dependency stalls)
- Variant 3a: Lever 2a (Bulk 20-Trit Vector Chunking: 5 contiguous half4 vector loads)
- Variant 4: Lever 4 (Double-Buffered Tile Staging & Weight Prefetching)

Strict Invariant:
Zero modifications to production files in src/. This is an isolated scratch test.
"""

import sys
import os
import time
import subprocess
import tempfile
from pathlib import Path
import mlx.core as mx
import mlx.core.fast as fast

# Add mlx-ternary/src to sys.path
REPO_DIR = Path(__file__).resolve().parent.parent / "mlx-ternary"
sys.path.insert(0, str(REPO_DIR / "src"))

from mlx_ternary.ops import _LUT_HEADER, get_wide_gemv_source
from mlx_ternary.quantize import pack_interleaved


def make_v2_ilp_split_source(M: int, K: int, N: int, G_SIZE: int = 128) -> str:
    """
    Variant 2: 4-Way ILP Accumulator Splitting (Lever 3).
    - Keeps LUT in constant memory space (base3_lut_low, base3_lut_high).
    - Replaces single `block_acc` with 4 independent registers (`block_acc0..3`).
    - Breaks 3-4 cycle FP16 FMA Read-After-Write (RAW) register dependency stall.
    - Enables Apple Silicon dual-issue ALU pipeline to fire two FMAs simultaneously every cycle.
    """
    num_bytes = 26 if G_SIZE == 128 else 30

    source = f"""
    uint tid = thread_index_in_threadgroup;
    uint col = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    
    threadgroup half shared_act[{G_SIZE}];

    uint K_blocks = {K} / {G_SIZE};
    half acc = 0;
    device const half* act_ptr = activations + row * {K};

    for (uint k_block = 0; k_block < K_blocks; k_block++) {{
        // 1. Cooperative load into SRAM
        if (tid < 128) {{
            shared_act[tid] = act_ptr[k_block * {G_SIZE} + tid];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Compute dot product with 4-way ILP accumulator rotation
        if (col < {N} && row < {M}) {{
            uint block_offset_uint4 = (k_block * {N} + col) * 2;
            device const uint4* w_ptr = (device const uint4*)interleaved_weights;
            uint4 vec0 = w_ptr[block_offset_uint4];
            uint4 vec1 = w_ptr[block_offset_uint4 + 1];
            
            half scale = as_type<half>((uint16_t)(vec0.x & 0xFFFF));
            half block_acc0 = 0;
            half block_acc1 = 0;
            half block_acc2 = 0;
            half block_acc3 = 0;

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
                uint curr_x = i * 5;

                half term;
                if (i == 25) {{
                    term = shared_act[curr_x] * w_low.x + shared_act[curr_x + 1] * w_low.y + shared_act[curr_x + 2] * w_low.z;
                }} else {{
                    half4 x_low = half4(shared_act[curr_x], shared_act[curr_x + 1], shared_act[curr_x + 2], shared_act[curr_x + 3]);
                    term = dot(x_low, w_low) + shared_act[curr_x + 4] * w_high;
                }}

                if ((i & 3) == 0)      block_acc0 += term;
                else if ((i & 3) == 1) block_acc1 += term;
                else if ((i & 3) == 2) block_acc2 += term;
                else                   block_acc3 += term;
            }}

            half block_acc = (block_acc0 + block_acc1) + (block_acc2 + block_acc3);
            acc += block_acc * scale;
        }}

        // 3. Exit barrier before next tile load
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (col < {N} && row < {M}) {{
        output[row * {N} + col] = acc;
    }}
    """
    return source


def make_v3a_bulk_vector_source(M: int, K: int, N: int, G_SIZE: int = 128) -> str:
    """
    Variant 3a: Bulk 20-Trit Vector Chunking (Lever 2a - Tier 1 Champ).
    - Loads 5 contiguous `half4` vector loads (20 activations) simultaneously per 4-byte chunk.
    - Slashes shared memory load instructions by 75% vs scalar indexing.
    - Accumulates directly into 4 independent ILP registers (block_acc0..3).
    - Handles tail 2 bytes (8 elements) via two `half4` vector loads.
    """
    source = f"""
    uint tid = thread_index_in_threadgroup;
    uint col = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    
    threadgroup half shared_act[{G_SIZE}];

    uint K_blocks = {K} / {G_SIZE};
    half acc = 0;
    device const half* act_ptr = activations + row * {K};

    for (uint k_block = 0; k_block < K_blocks; k_block++) {{
        // 1. Cooperative load into SRAM
        if (tid < 128) {{
            shared_act[tid] = act_ptr[k_block * {G_SIZE} + tid];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Compute dot product with Bulk 20-Trit Vector Chunking
        if (col < {N} && row < {M}) {{
            uint block_offset_uint4 = (k_block * {N} + col) * 2;
            device const uint4* w_ptr = (device const uint4*)interleaved_weights;
            uint4 vec0 = w_ptr[block_offset_uint4];
            uint4 vec1 = w_ptr[block_offset_uint4 + 1];
            
            half scale = as_type<half>((uint16_t)(vec0.x & 0xFFFF));
            half block_acc0 = 0;
            half block_acc1 = 0;
            half block_acc2 = 0;
            half block_acc3 = 0;

            struct PackedBlock {{
                uint4 v0;
                uint4 v1;
            }};
            PackedBlock blk = {{vec0, vec1}};
            thread const uchar* blk_bytes = (thread const uchar*)&blk;

            // 6 chunks x 4 bytes = 24 bytes (120 elements)
            #pragma unroll(6)
            for (uint c = 0; c < 6; c++) {{
                threadgroup const half4* x_vec = (threadgroup const half4*)&shared_act[c * 20];
                half4 x0 = x_vec[0];
                half4 x1 = x_vec[1];
                half4 x2 = x_vec[2];
                half4 x3 = x_vec[3];
                half4 x4 = x_vec[4];

                uint8_t b0 = blk_bytes[2 + c * 4 + 0];
                uint8_t b1 = blk_bytes[2 + c * 4 + 1];
                uint8_t b2 = blk_bytes[2 + c * 4 + 2];
                uint8_t b3 = blk_bytes[2 + c * 4 + 3];

                half4 w0_low = base3_lut_low[b0];
                half w0_high = base3_lut_high[b0];
                half4 w1_low = base3_lut_low[b1];
                half w1_high = base3_lut_high[b1];
                half4 w2_low = base3_lut_low[b2];
                half w2_high = base3_lut_high[b2];
                half4 w3_low = base3_lut_low[b3];
                half w3_high = base3_lut_high[b3];

                block_acc0 += dot(x0, w0_low) + x1.x * w0_high;
                block_acc1 += x1.y * w1_low.x + x1.z * w1_low.y + x1.w * w1_low.z + x2.x * w1_low.w + x2.y * w1_high;
                block_acc2 += x2.z * w2_low.x + x2.w * w2_low.y + x3.x * w2_low.z + x3.y * w2_low.w + x3.z * w2_high;
                block_acc3 += x3.w * w3_low.x + dot(x4, half4(w3_low.y, w3_low.z, w3_low.w, w3_high));
            }}

            // Tail 2 bytes (bytes 24 and 25 -> elements 120..127)
            threadgroup const half4* tail_vec = (threadgroup const half4*)&shared_act[120];
            half4 tail0 = tail_vec[0];
            half4 tail1 = tail_vec[1];

            uint8_t b24 = blk_bytes[2 + 24];
            uint8_t b25 = blk_bytes[2 + 25];

            half4 w24_low = base3_lut_low[b24];
            half w24_high = base3_lut_high[b24];
            half4 w25_low = base3_lut_low[b25];

            block_acc0 += dot(tail0, w24_low) + tail1.x * w24_high;
            block_acc1 += tail1.y * w25_low.x + tail1.z * w25_low.y + tail1.w * w25_low.z;

            half block_acc = (block_acc0 + block_acc1) + (block_acc2 + block_acc3);
            acc += block_acc * scale;
        }}

        // 3. Exit barrier before next tile load
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (col < {N} && row < {M}) {{
        output[row * {N} + col] = acc;
    }}
    """
    return source


def make_v4_double_buffered_source(M: int, K: int, N: int, G_SIZE: int = 128) -> str:
    """
    Variant 4: Double-Buffered Tile Staging & Weight Prefetching (Lever 4 - Tier 2).
    - Allocates ping-pong shared memory: `threadgroup half shared_act[2][128]` (512 bytes).
    - Prologue: Cooperatively loads tile 0 into shared_act[0], issues single barrier, pre-loads block 0 weights.
    - Main Loop: Asynchronously loads tile k+1 into next buffer while computing on current buffer.
    - Slashes threadgroup synchronization barriers in half (from 80 to 40 per contraction).
    - Fully overlaps shared memory store latency and weight loads with ALU execution.
    """
    source = f"""
    uint tid = thread_index_in_threadgroup;
    uint col = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    
    // Ping-pong shared memory buffer (2 x 128 halfs = 512 bytes)
    threadgroup half shared_act[2][{G_SIZE}];

    uint K_blocks = {K} / {G_SIZE};
    half acc = 0;
    device const half* act_ptr = activations + row * {K};
    device const uint4* w_ptr = (device const uint4*)interleaved_weights;

    // --- PROLOGUE ---
    // 1. Cooperatively load tile 0 into shared_act[0]
    if (tid < 128) {{
        shared_act[0][tid] = act_ptr[tid];
    }}
    // 2. Synchronize once so tile 0 is visible to all threads in threadgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Pre-load weights for block 0 into registers
    uint4 cur_vec0 = uint4(0);
    uint4 cur_vec1 = uint4(0);
    if (col < {N} && row < {M}) {{
        uint block_offset = col * 2;
        cur_vec0 = w_ptr[block_offset];
        cur_vec1 = w_ptr[block_offset + 1];
    }}

    // --- MAIN PIPELINE LOOP ---
    for (uint k_block = 0; k_block < K_blocks; k_block++) {{
        uint cur_buf = k_block & 1;
        uint next_buf = 1 - cur_buf;

        // Asynchronously issue cooperative load of next tile and prefetch next weights
        uint4 next_vec0 = uint4(0);
        uint4 next_vec1 = uint4(0);
        if (k_block + 1 < K_blocks) {{
            if (tid < 128) {{
                shared_act[next_buf][tid] = act_ptr[(k_block + 1) * {G_SIZE} + tid];
            }}
            if (col < {N} && row < {M}) {{
                uint next_offset = ((k_block + 1) * {N} + col) * 2;
                next_vec0 = w_ptr[next_offset];
                next_vec1 = w_ptr[next_offset + 1];
            }}
        }}

        // Compute dot product on shared_act[cur_buf] using cur_vec0, cur_vec1 (Variant 3a logic)
        if (col < {N} && row < {M}) {{
            half scale = as_type<half>((uint16_t)(cur_vec0.x & 0xFFFF));
            half block_acc0 = 0;
            half block_acc1 = 0;
            half block_acc2 = 0;
            half block_acc3 = 0;

            struct PackedBlock {{
                uint4 v0;
                uint4 v1;
            }};
            PackedBlock blk = {{cur_vec0, cur_vec1}};
            thread const uchar* blk_bytes = (thread const uchar*)&blk;

            // 6 chunks x 4 bytes = 24 bytes (120 elements)
            #pragma unroll(6)
            for (uint c = 0; c < 6; c++) {{
                threadgroup const half4* x_vec = (threadgroup const half4*)&shared_act[cur_buf][c * 20];
                half4 x0 = x_vec[0];
                half4 x1 = x_vec[1];
                half4 x2 = x_vec[2];
                half4 x3 = x_vec[3];
                half4 x4 = x_vec[4];

                uint8_t b0 = blk_bytes[2 + c * 4 + 0];
                uint8_t b1 = blk_bytes[2 + c * 4 + 1];
                uint8_t b2 = blk_bytes[2 + c * 4 + 2];
                uint8_t b3 = blk_bytes[2 + c * 4 + 3];

                half4 w0_low = base3_lut_low[b0];
                half w0_high = base3_lut_high[b0];
                half4 w1_low = base3_lut_low[b1];
                half w1_high = base3_lut_high[b1];
                half4 w2_low = base3_lut_low[b2];
                half w2_high = base3_lut_high[b2];
                half4 w3_low = base3_lut_low[b3];
                half w3_high = base3_lut_high[b3];

                block_acc0 += dot(x0, w0_low) + x1.x * w0_high;
                block_acc1 += x1.y * w1_low.x + x1.z * w1_low.y + x1.w * w1_low.z + x2.x * w1_low.w + x2.y * w1_high;
                block_acc2 += x2.z * w2_low.x + x2.w * w2_low.y + x3.x * w2_low.z + x3.y * w2_low.w + x3.z * w2_high;
                block_acc3 += x3.w * w3_low.x + dot(x4, half4(w3_low.y, w3_low.z, w3_low.w, w3_high));
            }}

            // Tail 2 bytes (bytes 24 and 25 -> elements 120..127)
            threadgroup const half4* tail_vec = (threadgroup const half4*)&shared_act[cur_buf][120];
            half4 tail0 = tail_vec[0];
            half4 tail1 = tail_vec[1];

            uint8_t b24 = blk_bytes[2 + 24];
            uint8_t b25 = blk_bytes[2 + 25];

            half4 w24_low = base3_lut_low[b24];
            half w24_high = base3_lut_high[b24];
            half4 w25_low = base3_lut_low[b25];

            block_acc0 += dot(tail0, w24_low) + tail1.x * w24_high;
            block_acc1 += tail1.y * w25_low.x + tail1.z * w25_low.y + tail1.w * w25_low.z;

            half block_acc = (block_acc0 + block_acc1) + (block_acc2 + block_acc3);
            acc += block_acc * scale;
        }}

        // Single barrier per iteration: ensures shared_act[cur_buf] reading is finished
        // and shared_act[next_buf] writing is complete before switching buffers.
        if (k_block + 1 < K_blocks) {{
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        // Advance weight registers to next preloaded vectors
        cur_vec0 = next_vec0;
        cur_vec1 = next_vec1;
    }}

    // --- EPILOGUE ---
    if (col < {N} && row < {M}) {{
        output[row * {N} + col] = acc;
    }}
    """
    return source


def check_metal_compiler_diagnostics(source: str, header: str, kernel_name: str) -> str:
    """Compile shader with metal CLI to capture any compiler warnings or spill diagnostics."""
    full_code = f"""
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
typedef half float16_t;

{header}

kernel void {kernel_name}(
    device const half* activations [[buffer(0)]],
    device const uchar* interleaved_weights [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]
) {{
{source}
}}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".metal", delete=False) as f:
        f.write(full_code)
        f_path = f.name

    try:
        res = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-c", "-Wall", "-Wextra", f_path, "-o", "/dev/null"],
            capture_output=True,
            text=True,
        )
        diag = res.stderr.strip()
        return diag if diag else "Zero warnings. Zero errors. Clean compilation."
    except Exception as e:
        return f"Metal CLI check error: {e}"
    finally:
        if os.path.exists(f_path):
            os.remove(f_path)


def run_benchmark():
    M = 1
    K = 5120
    N = 17408
    G_SIZE = 128
    EMPIRICAL_CEILING_GB_S = 254.2
    WARMUP_RUNS = 200
    TIMED_RUNS = 500

    print("=" * 115)
    print("      Phase 2F Tier 2 Micro-Benchmark: Lever 4 (Double-Buffered Tile Staging & Prefetch)")
    print("=" * 115)
    print(f"Target Contraction: MLP Gate/Up (M={M}, K={K}, N={N}, Group Size={G_SIZE})")
    print(f"Benchmark Protocol: Warmup = {WARMUP_RUNS} runs, Timed = {TIMED_RUNS} runs")
    print(f"Empirical Memory Bandwidth Ceiling: {EMPIRICAL_CEILING_GB_S:.1f} GB/s")
    print("-" * 115)

    # 1. Synthesize Deterministic Inputs
    mx.random.seed(42)
    x = mx.random.normal((M, K)).astype(mx.float16)
    w_raw = mx.random.randint(0, 3, (K, N)).astype(mx.float16) - 1.0
    w_packed, _ = pack_interleaved(w_raw, group_size=G_SIZE)
    mx.eval(x, w_packed)

    weight_bytes = (K * N) // 4  # 2 bits per element
    weight_mb = weight_bytes / (1024 ** 2)
    print(f"Synthesized packed weight matrix: {w_packed.shape} ({weight_mb:.2f} MB)")

    # 2. Compile Variant 0: Production Baseline (Constant Cache LUT, 1 Accumulator)
    src_v0 = get_wide_gemv_source(M, K, N, G_SIZE)
    kernel_v0 = fast.metal_kernel(
        name=f"wide_gemv_v0_{M}_{K}_{N}_{G_SIZE}",
        input_names=["activations", "interleaved_weights"],
        output_names=["output"],
        source=src_v0,
        header=_LUT_HEADER,
    )

    # 3. Compile Variant 2: Lever 3 (4-Way ILP Accumulator Splitting)
    src_v2 = make_v2_ilp_split_source(M, K, N, G_SIZE)
    kernel_v2 = fast.metal_kernel(
        name=f"wide_gemv_v2_{M}_{K}_{N}_{G_SIZE}",
        input_names=["activations", "interleaved_weights"],
        output_names=["output"],
        source=src_v2,
        header=_LUT_HEADER,
    )

    # 4. Compile Variant 3a: Lever 2a (Bulk 20-Trit Vector Chunking - Tier 1 Champion)
    src_v3a = make_v3a_bulk_vector_source(M, K, N, G_SIZE)
    kernel_v3a = fast.metal_kernel(
        name=f"wide_gemv_v3a_{M}_{K}_{N}_{G_SIZE}",
        input_names=["activations", "interleaved_weights"],
        output_names=["output"],
        source=src_v3a,
        header=_LUT_HEADER,
    )

    # 5. Compile Variant 4: Lever 4 (Double-Buffered Tile Staging & Weight Prefetching)
    src_v4 = make_v4_double_buffered_source(M, K, N, G_SIZE)
    kernel_v4 = fast.metal_kernel(
        name=f"wide_gemv_v4_{M}_{K}_{N}_{G_SIZE}",
        input_names=["activations", "interleaved_weights"],
        output_names=["output"],
        source=src_v4,
        header=_LUT_HEADER,
    )

    tg_size = 128
    grid_x = ((N + tg_size - 1) // tg_size) * tg_size
    grid = (grid_x, M, 1)
    threadgroup = (tg_size, 1, 1)

    # 6. Enforce Numerical Equivalence
    print("\nVerifying numerical equivalence...")
    out_v0 = kernel_v0(inputs=[x, w_packed], grid=grid, threadgroup=threadgroup, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
    out_v2 = kernel_v2(inputs=[x, w_packed], grid=grid, threadgroup=threadgroup, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
    out_v3a = kernel_v3a(inputs=[x, w_packed], grid=grid, threadgroup=threadgroup, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
    out_v4 = kernel_v4(inputs=[x, w_packed], grid=grid, threadgroup=threadgroup, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
    mx.eval(out_v0, out_v2, out_v3a, out_v4)

    diff_v2 = mx.abs(out_v2 - out_v0).max().item()
    diff_v3a = mx.abs(out_v3a - out_v0).max().item()
    diff_v4 = mx.abs(out_v4 - out_v0).max().item()
    rel_diff_v2 = (mx.abs(out_v2 - out_v0) / (mx.abs(out_v0) + 1e-3)).mean().item()
    rel_diff_v3a = (mx.abs(out_v3a - out_v0) / (mx.abs(out_v0) + 1e-3)).mean().item()
    rel_diff_v4 = (mx.abs(out_v4 - out_v0) / (mx.abs(out_v0) + 1e-3)).mean().item()

    print(f"  • Max Absolute Difference (V2 vs V0):   {diff_v2:.6f}")
    print(f"  • Mean Relative Difference (V2 vs V0):  {rel_diff_v2:.6f} (project threshold < 0.03)")
    print(f"  • Max Absolute Difference (V3a vs V0):  {diff_v3a:.6f}")
    print(f"  • Mean Relative Difference (V3a vs V0): {rel_diff_v3a:.6f} (project threshold < 0.03)")
    print(f"  • Max Absolute Difference (V4 vs V0):   {diff_v4:.6f}")
    print(f"  • Mean Relative Difference (V4 vs V0):  {rel_diff_v4:.6f} (project threshold < 0.03)")

    assert rel_diff_v2 < 0.03, f"Relative difference {rel_diff_v2} exceeded threshold 0.03!"
    assert rel_diff_v3a < 0.03, f"Relative difference {rel_diff_v3a} exceeded threshold 0.03!"
    assert rel_diff_v4 < 0.03, f"Relative difference {rel_diff_v4} exceeded threshold 0.03!"

    # Verify exact atol <= 1e-3 on test scale
    x_test = x * 0.001
    out_v0_test = kernel_v0(inputs=[x_test, w_packed], grid=grid, threadgroup=threadgroup, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
    out_v3a_test = kernel_v3a(inputs=[x_test, w_packed], grid=grid, threadgroup=threadgroup, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
    out_v4_test = kernel_v4(inputs=[x_test, w_packed], grid=grid, threadgroup=threadgroup, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
    mx.eval(out_v0_test, out_v3a_test, out_v4_test)
    diff_v3a_test = mx.abs(out_v3a_test - out_v0_test).max().item()
    diff_v4_test = mx.abs(out_v4_test - out_v0_test).max().item()
    assert mx.allclose(out_v3a_test, out_v0_test, atol=1e-3), f"allclose atol=1e-3 failed on V3a test slice: {diff_v3a_test}!"
    assert mx.allclose(out_v4_test, out_v0_test, atol=1e-3), f"allclose atol=1e-3 failed on V4 test slice: {diff_v4_test}!"
    print(f"  • Scaled Slice Absolute Difference (V3a): {diff_v3a_test:.6f} (atol <= 1e-3 verified)")
    print(f"  • Scaled Slice Absolute Difference (V4):  {diff_v4_test:.6f} (atol <= 1e-3 verified)")
    print("  ✓ Numerical equivalence verified: All precision criteria PASSED.\n")

    # 7. Execute Benchmark Protocol
    variants = [
        ("Variant 0", "Production Baseline (Constant Cache LUT)", kernel_v0),
        ("Variant 2", "Lever 3 (4-Way ILP Splitting)", kernel_v2),
        ("Variant 3a", "Tier 1 Champ (Aligned 20-Trit Chunking)", kernel_v3a),
        ("Variant 4", "Tier 2 (Double-Buffered Tile & Prefetch)", kernel_v4),
    ]

    results = []
    base_lat_us = None

    for code, impl, k in variants:
        print(f"Benchmarking {code} ({impl})...")
        # Warmup
        for _ in range(WARMUP_RUNS):
            res = k(inputs=[x, w_packed], grid=grid, threadgroup=threadgroup, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
            mx.eval(res)

        # Timed Runs
        start_time = time.perf_counter()
        for _ in range(TIMED_RUNS):
            res = k(inputs=[x, w_packed], grid=grid, threadgroup=threadgroup, output_shapes=[(M, N)], output_dtypes=[mx.float16])[0]
            mx.eval(res)
        elapsed = time.perf_counter() - start_time

        lat_s = elapsed / TIMED_RUNS
        lat_us = lat_s * 1e6
        bw_gb_s = (weight_bytes / lat_s) / 1e9
        pct_ceiling = (bw_gb_s / EMPIRICAL_CEILING_GB_S) * 100.0

        if base_lat_us is None:
            base_lat_us = lat_us
            delta_str = "Baseline"
        else:
            diff_us = lat_us - base_lat_us
            pct_diff = (diff_us / base_lat_us) * 100.0
            delta_str = f"{diff_us:+.2f} µs ({pct_diff:+.1f}%)"

        results.append({
            "code": code,
            "impl": impl,
            "lat_us": lat_us,
            "bw_gb_s": bw_gb_s,
            "pct_ceiling": pct_ceiling,
            "delta_str": delta_str,
        })
        print(f"  -> Latency: {lat_us:.2f} µs | Bandwidth: {bw_gb_s:.2f} GB/s ({pct_ceiling:.1f}% ceiling) | Delta: {delta_str}")

    # 8. Print Clean Comparison Table
    print("\n" + "=" * 115)
    print("                                      EMPIRICAL TELEMETRY COMPARISON TABLE")
    print("=" * 115)
    header_line = f"| {'Variant':<10} | {'Implementation':<45} | {'Latency (µs)':>12} | {'Bandwidth (GB/s)':>16} | {'% of Ceiling (254.2)':>20} | {'Delta vs V0':>18} |"
    sep_line = "|:" + "-" * 10 + "-|-:" + "-" * 45 + "-|-:" + "-" * 12 + ":|-:" + "-" * 16 + ":|-:" + "-" * 20 + ":|-:" + "-" * 18 + ":|"
    print(header_line)
    print(sep_line)
    for r in results:
        row_str = f"| {r['code']:<10} | {r['impl']:<45} | {r['lat_us']:>12.2f} | {r['bw_gb_s']:>16.2f} | {r['pct_ceiling']:>19.1f}% | {r['delta_str']:>18} |"
        print(row_str)
    print("=" * 115)

    # 9. Metal Compiler & Diagnostic Audit
    print("\n" + "=" * 115)
    print("Metal Shading Language (MSL) Compiler & Diagnostic Audit:")
    print("=" * 115)
    diag_v0 = check_metal_compiler_diagnostics(src_v0, _LUT_HEADER, "kernel_v0")
    diag_v2 = check_metal_compiler_diagnostics(src_v2, _LUT_HEADER, "kernel_v2")
    diag_v3a = check_metal_compiler_diagnostics(src_v3a, _LUT_HEADER, "kernel_v3a")
    diag_v4 = check_metal_compiler_diagnostics(src_v4, _LUT_HEADER, "kernel_v4")

    print(f"  • Variant 0 (Production Constant LUT):       {diag_v0}")
    print(f"  • Variant 2 (Lever 3 4-Way ILP Splitting):   {diag_v2}")
    print(f"  • Variant 3a (Tier 1 Champ 20-Trit Chunk):   {diag_v3a}")
    print(f"  • Variant 4 (Tier 2 Double-Buffered Staging): {diag_v4}")
    print("-" * 115)


if __name__ == "__main__":
    run_benchmark()
