import mlx.core as mx
import mlx.core.fast as fast

_KERNEL_CACHE = {}
_GPU_CORE_COUNT = None


def _get_gpu_core_count() -> int:
    """Dynamically query GPU core count on Apple Silicon with fallback."""
    global _GPU_CORE_COUNT
    if _GPU_CORE_COUNT is not None:
        return _GPU_CORE_COUNT
    try:
        import subprocess
        out = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], timeout=2).decode()
        for line in out.splitlines():
            if "Total Number of Cores" in line:
                _GPU_CORE_COUNT = int(line.split(":")[-1].strip())
                return _GPU_CORE_COUNT
    except Exception:
        pass
    dev_name = mx.device_info().get("device_name", "")
    if "Max" in dev_name:
        _GPU_CORE_COUNT = 40
    elif "Ultra" in dev_name:
        _GPU_CORE_COUNT = 80
    elif "Pro" in dev_name:
        _GPU_CORE_COUNT = 20
    else:
        _GPU_CORE_COUNT = 10
    return _GPU_CORE_COUNT

def _build_lut_header():
    lut_entries = []
    low_entries = []
    high_entries = []
    for i in range(243):
        val = i
        row = []
        for _ in range(5):
            rem = val % 3
            if rem == 0: row.append("-1.0")
            elif rem == 1: row.append("0.0")
            elif rem == 2: row.append("1.0")
            val //= 3
        lut_entries.append("{" + ", ".join(row) + "}")
        low_entries.append(f"half4({row[0]}h, {row[1]}h, {row[2]}h, {row[3]}h)")
        high_entries.append(f"{row[4]}h")
    
    lut_str = ",\n".join(lut_entries)
    low_str = ",\n    ".join(low_entries)
    high_str = ",\n    ".join(high_entries)
    
    return f"""
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant float16_t base3_lut[243][5] = {{
{lut_str}
}};

constant half4 base3_lut_low[243] = {{
    {low_str}
}};

constant half base3_lut_high[243] = {{
    {high_str}
}};
"""

_LUT_HEADER = _build_lut_header()

def get_narrow_gemv_source(M: int, K: int, N: int, G_SIZE: int) -> str:
    """
    32-lane cooperative SIMD warp reduction GEMV for narrow layers (N <= 1024, e.g. Attention KV).
    - Grid: (32, ((N + tg_y - 1) // tg_y) * tg_y, M)
    - Threadgroup: (32, tg_y, 1) where tg_y = 4.
    - Intra-warp simd_sum() reduction; 0 threadgroup barriers.
    - Eliminates synchronization tax and saturates all 20 GPU cores.
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


def get_wide_gemv_source(M: int, K: int, N: int, G_SIZE: int) -> str:
    """
    Generate Metal Shading Language source for 1D contiguous column tiling GEMV with
    on-chip SRAM activation caching, aligned 20-trit vector chunking, and 4-way ILP accumulators.
    - Dispatched when N > 1024 (e.g. MLP Gate/Up, MLP Down, Attention Q/O).
    - Grid: (((N + tg_size - 1) // tg_size) * tg_size, M, 1)
    - Threadgroup: (tg_size, 1, 1) where tg_size = 128.
    - Cooperatively loads activation tiles into on-chip SRAM (`threadgroup half shared_act[G_SIZE]`).
    - Aligned 20-Trit Vector Chunking (Lever 2a): Slashes shared memory loads by 75% via contiguous half4 loads.
    - 4-Way ILP Accumulator Splitting (Lever 3): Breaks 3-4 cycle FP16 FMA RAW register dependency stalls.
    - Eliminates redundant DRAM activation reads across columns and fetches 128-byte coalesced weight bursts.
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
    """
    if G_SIZE == 150:
        source += f"""
        if (tid < 22) {{
            shared_act[128 + tid] = act_ptr[k_block * 150 + 128 + tid];
        }}
        """
    source += f"""
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Compute dot product reading from on-chip SRAM
        if (col < {N} && row < {M}) {{
            uint block_offset_uint4 = (k_block * {N} + col) * 2;
            device const uint4* w_ptr = (device const uint4*)interleaved_weights;
            uint4 vec0 = w_ptr[block_offset_uint4];
            uint4 vec1 = w_ptr[block_offset_uint4 + 1];
            
            half scale = as_type<half>((uint16_t)(vec0.x & 0xFFFF));

            struct PackedBlock {{
                uint4 v0;
                uint4 v1;
            }};
            PackedBlock blk = {{vec0, vec1}};
            thread const uchar* blk_bytes = (thread const uchar*)&blk;
    """

    if G_SIZE == 128:
        source += """
            half block_acc0 = 0;
            half block_acc1 = 0;
            half block_acc2 = 0;
            half block_acc3 = 0;

            // 6 chunks x 4 bytes = 24 bytes (120 elements)
            #pragma unroll(6)
            for (uint c = 0; c < 6; c++) {
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
            }

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
        """
    else:
        source += f"""
            half block_acc = 0;
            #pragma unroll({num_bytes})
            for (uint i = 0; i < {num_bytes}; i++) {{
                uint8_t b = blk_bytes[2 + i];
                half4 w_low = base3_lut_low[b];
                half w_high = base3_lut_high[b];
                uint curr_x = i * 5;
                half4 x_low = half4(shared_act[curr_x], shared_act[curr_x + 1], shared_act[curr_x + 2], shared_act[curr_x + 3]);
                half x_high = shared_act[curr_x + 4];
                block_acc += dot(x_low, w_low) + x_high * w_high;
            }}
            acc += block_acc * scale;
        """

    source += f"""
        }}

        // 3. Exit barrier before next tile load
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (col < {N} && row < {M}) {{
        output[row * {N} + col] = acc;
    }}
    """
    return source


def get_elastic_split_k_sources(M: int, K: int, N: int, S: int, group_size: int = 128) -> tuple[str, str]:
    """
    Generate Metal Shading Language sources for hardware-elastic Split-K GEMV reduction.

    Args:
        M: Activation batch/row count.
        K: Contraction dimension.
        N: Output dimension.
        S: Split-K slice count (S >= 1).
        group_size: Quantization group size (128 or 150).

    Returns:
        tuple[str, str]: (stage1_source, stage2_source)
        - stage1_source: Stage 1 partial accumulation kernel across S splits (or direct output if S=1).
        - stage2_source: Stage 2 parallel vector reduction kernel across S splits.
    """
    stage1_source = f"""
    uint col = thread_position_in_grid.x;
    uint split_id = thread_position_in_grid.y;
    uint row = thread_position_in_grid.z;

    if (col >= {N} || split_id >= {S} || row >= {M}) return;

    uint K_blocks = {K} / {group_size};
    uint blocks_per_split = (K_blocks + {S} - 1) / {S};
    uint start_block = split_id * blocks_per_split;
    uint end_block = min(start_block + blocks_per_split, K_blocks);

    if (start_block >= K_blocks) {{
        {"partial_output[row * (" + str(S) + " * " + str(N) + ") + split_id * " + str(N) + " + col] = 0.0f;" if S > 1 else f"output[row * {N} + col] = 0;"}
        return;
    }}

    half acc = 0;
    device const half* act_block = activations + row * {K} + start_block * {group_size};
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

    if group_size == 128:
        stage1_source += f"""
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
        stage1_source += f"""
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

    if S > 1:
        stage1_source += f"""
    }}
    partial_output[row * ({S} * {N}) + split_id * {N} + col] = (float)acc;
    """
    else:
        stage1_source += f"""
    }}
    output[row * {N} + col] = acc;
    """

    stage2_source = f"""
    uint idx = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    if (row >= {M}) return;

    #if {1 if N % 4 == 0 else 0}
    uint N_vec = {N} / 4;
    if (idx >= N_vec) return;

    device const float4* p_base = (device const float4*)(partial_output + row * ({S} * {N}));
    float4 sum = float4(0.0f);

    #pragma unroll({S})
    for (uint s = 0; s < {S}; s++) {{
        sum += p_base[s * N_vec + idx];
    }}

    device half4* out_vec = (device half4*)(output + row * {N});
    out_vec[idx] = half4(sum);
    #else
    if (idx >= {N}) return;
    float sum = 0.0f;
    #pragma unroll({S})
    for (uint s = 0; s < {S}; s++) {{
        sum += partial_output[row * ({S} * {N}) + s * {N} + idx];
    }}
    output[row * {N} + idx] = (half)sum;
    #endif
    """

    return stage1_source, stage2_source


def get_fused_swiglu_source(M: int, K: int, N: int, group_size: int = 128) -> str:
    """
    Generate Metal Shading Language source for single-pass fused Gate + Up + SiLU contraction.
    Uses threadgroup SRAM cooperative activation caching and single-buffered weight streaming
    to strictly enforce <= 24 registers for 100% SIMD warp occupancy.
    """
    num_bytes = 26 if group_size == 128 else 30

    source = f"""
    uint tid = thread_index_in_threadgroup;
    uint col = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;

    threadgroup half shared_act[{group_size}];

    if (col >= {N} || row >= {M}) return;

    uint K_blocks = {K} / {group_size};
    half acc_gate = 0;
    half acc_up = 0;

    device const half* act_block = activations + row * {K};
    device const uint4* w_gate_ptr = ((device const uint4*)gate_weights) + col * 2;
    device const uint4* w_up_ptr = ((device const uint4*)up_weights) + col * 2;
    uint w_stride = {N} * 2;

    for (uint k_block = 0; k_block < K_blocks; k_block++) {{
        // 1. Cooperative load into on-chip SRAM
        if (tid < 128) {{
            shared_act[tid] = act_block[tid];
        }}
    """
    if group_size == 150:
        source += f"""
        if (tid < 22) {{
            shared_act[128 + tid] = act_block[128 + tid];
        }}
        """

    source += f"""
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Direct single-buffered weight streaming (<= 24 registers, 100% warp occupancy)
        uint4 cur_g0 = w_gate_ptr[0];
        uint4 cur_g1 = w_gate_ptr[1];
        w_gate_ptr += w_stride;

        uint4 cur_u0 = w_up_ptr[0];
        uint4 cur_u1 = w_up_ptr[1];
        w_up_ptr += w_stride;

        half scale_g = as_type<half>((uint16_t)(cur_g0.x & 0xFFFF));
        half scale_u = as_type<half>((uint16_t)(cur_u0.x & 0xFFFF));

        struct PackedBlock {{ uint4 v0, v1; }};
        PackedBlock blk_g = {{cur_g0, cur_g1}};
        PackedBlock blk_u = {{cur_u0, cur_u1}};
        thread const uchar* bytes_g = (thread const uchar*)&blk_g;
        thread const uchar* bytes_u = (thread const uchar*)&blk_u;
    """

    if group_size == 128:
        source += f"""
        half g_acc0 = 0, g_acc1 = 0, g_acc2 = 0, g_acc3 = 0;
        half u_acc0 = 0, u_acc1 = 0, u_acc2 = 0, u_acc3 = 0;

        threadgroup const half4* act_vec = (threadgroup const half4*)shared_act;

        #pragma unroll(6)
        for (uint c = 0; c < 6; c++) {{
            half4 x0 = act_vec[c * 5 + 0];
            half4 x1 = act_vec[c * 5 + 1];
            half4 x2 = act_vec[c * 5 + 2];
            half4 x3 = act_vec[c * 5 + 3];
            half4 x4 = act_vec[c * 5 + 4];

            // Gate Chunk FMAs
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

            // Up Chunk FMAs (reuses loaded x0..x4 from SRAM)
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

        threadgroup const half4* tail_vec = (threadgroup const half4*)&shared_act[120];
        half4 tail0 = tail_vec[0];
        half4 tail1 = tail_vec[1];

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
        """
    else:
        source += f"""
        half block_acc_g = 0;
        half block_acc_u = 0;

        #pragma unroll(30)
        for (uint i = 0; i < 30; i++) {{
            uint curr_x = i * 5;
            half4 x_low = half4(shared_act[curr_x], shared_act[curr_x + 1], shared_act[curr_x + 2], shared_act[curr_x + 3]);
            half x_high = shared_act[curr_x + 4];

            uint8_t gb = bytes_g[2 + i];
            half4 gw_low = base3_lut_low[gb]; half gw_high = base3_lut_high[gb];
            block_acc_g += dot(x_low, gw_low) + x_high * gw_high;

            uint8_t ub = bytes_u[2 + i];
            half4 uw_low = base3_lut_low[ub]; half uw_high = base3_lut_high[ub];
            block_acc_u += dot(x_low, uw_low) + x_high * uw_high;
        }}

        acc_gate += block_acc_g * scale_g;
        acc_up   += block_acc_u * scale_u;
        """

    source += f"""
        act_block += {group_size};
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // In-register SiLU(acc_gate) * acc_up evaluation
    float g = (float)acc_gate;
    float silu_g = g / (1.0f + metal::exp(-g));
    half out_val = (half)(silu_g * (float)acc_up);
    output[row * {N} + col] = out_val;
    """
    return source


def get_fused_kv_source(M: int, K: int, N: int, group_size: int = 128) -> str:
    """
    Generate Metal Shading Language source for single-pass fused Key + Value contraction.
    Uses threadgroup SRAM cooperative activation caching and single-buffered weight streaming
    to strictly enforce <= 24 registers for 100% SIMD warp occupancy.

    Args:
        M: Activation batch/row count.
        K: Contraction dimension.
        N: Output dimension for each of K and V.
        group_size: Quantization group size (128 or 150).

    Returns:
        str: Metal Shading Language source for fused KV.
    """
    source = f"""
    uint tid = thread_index_in_threadgroup;
    uint col = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;

    threadgroup half shared_act[{group_size}];

    if (col >= {N} || row >= {M}) return;

    uint K_blocks = {K} / {group_size};
    half acc_k = 0;
    half acc_v = 0;

    device const half* act_block = activations + row * {K};
    device const uint4* w_k_ptr = ((device const uint4*)k_weights) + col * 2;
    device const uint4* w_v_ptr = ((device const uint4*)v_weights) + col * 2;
    uint w_stride = {N} * 2;

    for (uint k_block = 0; k_block < K_blocks; k_block++) {{
        // 1. Cooperative load into on-chip SRAM
        if (tid < 128) {{
            shared_act[tid] = act_block[tid];
        }}
    """
    if group_size == 150:
        source += f"""
        if (tid < 22) {{
            shared_act[128 + tid] = act_block[128 + tid];
        }}
        """

    source += f"""
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Direct single-buffered weight streaming (<= 24 registers, 100% warp occupancy)
        uint4 cur_k0 = w_k_ptr[0];
        uint4 cur_k1 = w_k_ptr[1];
        w_k_ptr += w_stride;

        uint4 cur_v0 = w_v_ptr[0];
        uint4 cur_v1 = w_v_ptr[1];
        w_v_ptr += w_stride;

        half scale_k = as_type<half>((uint16_t)(cur_k0.x & 0xFFFF));
        half scale_v = as_type<half>((uint16_t)(cur_v0.x & 0xFFFF));

        struct PackedBlock {{ uint4 v0, v1; }};
        PackedBlock blk_k = {{cur_k0, cur_k1}};
        PackedBlock blk_v = {{cur_v0, cur_v1}};
        thread const uchar* bytes_k = (thread const uchar*)&blk_k;
        thread const uchar* bytes_v = (thread const uchar*)&blk_v;
    """

    if group_size == 128:
        source += f"""
        half k_acc0 = 0, k_acc1 = 0, k_acc2 = 0, k_acc3 = 0;
        half v_acc0 = 0, v_acc1 = 0, v_acc2 = 0, v_acc3 = 0;

        threadgroup const half4* act_vec = (threadgroup const half4*)shared_act;

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

            // 2. V Chunk FMAs (reuses loaded x0..x4, dual-issue with K)
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

        threadgroup const half4* tail_vec = (threadgroup const half4*)&shared_act[120];
        half4 tail0 = tail_vec[0];
        half4 tail1 = tail_vec[1];

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
        """
    else:
        source += f"""
        half block_acc_k = 0;
        half block_acc_v = 0;

        #pragma unroll(30)
        for (uint i = 0; i < 30; i++) {{
            uint curr_x = i * 5;
            half4 x_low = half4(shared_act[curr_x], shared_act[curr_x + 1], shared_act[curr_x + 2], shared_act[curr_x + 3]);
            half x_high = shared_act[curr_x + 4];

            uint8_t kb = bytes_k[2 + i];
            half4 kw_low = base3_lut_low[kb]; half kw_high = base3_lut_high[kb];
            block_acc_k += dot(x_low, kw_low) + x_high * kw_high;

            uint8_t vb = bytes_v[2 + i];
            half4 vw_low = base3_lut_low[vb]; half vw_high = base3_lut_high[vb];
            block_acc_v += dot(x_low, vw_low) + x_high * vw_high;
        }}

        acc_k += block_acc_k * scale_k;
        acc_v += block_acc_v * scale_v;
        """

    source += f"""
        act_block += {group_size};
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    out_k[row * {N} + col] = acc_k;
    out_v[row * {N} + col] = acc_v;
    """
    return source


def get_fused_qkv_source(M: int, K: int, N: int, group_size: int = 128) -> str:
    """
    Generate Metal Shading Language source for single-pass unified QKV projection.

    Evaluates concatenated [Q, K, V] projections in a single 0-barrier wide GEMV dispatch
    with 4-way ILP accumulators and asynchronous weight prefetching.

    Args:
        M: Activation batch/row count.
        K: Contraction dimension.
        N: Total output dimension (e.g. N_q + N_k + N_v).
        group_size: Quantization group size (128 or 150).

    Returns:
        str: Metal Shading Language source for unified QKV.
    """
    source = f"""
    uint col = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;

    if (col >= {N} || row >= {M}) return;

    uint K_blocks = {K} / {group_size};
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

    if group_size == 128:
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


def get_gemm_source(M, K, N, G_SIZE):
    num_bytes = 26 if G_SIZE == 128 else 30
    k_tile = 128 if G_SIZE == 128 else 152
    
    source = f"""
    uint tid = thread_index_in_threadgroup;
    uint sg_id = tid / 32;
    uint lane_id = tid % 32;
    
    uint m_start = threadgroup_position_in_grid.y * 32;
    uint n_start = threadgroup_position_in_grid.x * 32;
    
    threadgroup half As[32][{k_tile}];
    threadgroup half Bs[{k_tile}][32];
    
    simdgroup_matrix<half, 8, 8> acc[2][4];
    for (int i=0; i<2; i++) {{
        for (int j=0; j<4; j++) {{
            acc[i][j] = simdgroup_matrix<half, 8, 8>(0);
        }}
    }}
    
    uint K_blocks = {K} / {G_SIZE};
    
    for (uint k_block = 0; k_block < K_blocks; k_block++) {{
        for (uint i = tid; i < 32 * {k_tile}; i += 64) {{
            uint row = i / {k_tile};
            uint col = i % {k_tile};
            if (col < {G_SIZE} && (m_start + row) < {M}) {{
                As[row][col] = activations[(m_start + row) * {K} + (k_block * {G_SIZE} + col)];
            }} else {{
                As[row][col] = 0;
            }}
        }}
        
        if (sg_id == 0) {{
            uint col = lane_id; 
            uint global_n = n_start + col;
            
            if (global_n < {N}) {{
                uint block_offset = (k_block * {N} + global_n) * 2; 
                device const uint4* w_ptr = (device const uint4*)interleaved_weights;
                
                uint4 vec0 = w_ptr[block_offset];
                uint4 vec1 = w_ptr[block_offset + 1];
                
                float16_t scale = as_type<half>((uint16_t)(vec0.x & 0xFFFF));
                uint32_t chunks[8] = {{ vec0.x, vec0.y, vec0.z, vec0.w, vec1.x, vec1.y, vec1.z, vec1.w }};
                
                uint byte_idx = 2; 
                #pragma unroll({num_bytes})
                for (uint i = 0; i < {num_bytes}; i++) {{
                    uint chunk_idx = byte_idx / 4;
                    uint byte_offset = (byte_idx % 4) * 8;
                    uint8_t b = (chunks[chunk_idx] >> byte_offset) & 0xFF;
                    
                    Bs[i * 5 + 0][col] = base3_lut[b][0] * scale;
                    Bs[i * 5 + 1][col] = base3_lut[b][1] * scale;
                    Bs[i * 5 + 2][col] = base3_lut[b][2] * scale;
"""
    if G_SIZE == 150:
        source += """
                    Bs[i * 5 + 3][col] = base3_lut[b][3] * scale;
                    Bs[i * 5 + 4][col] = base3_lut[b][4] * scale;
                    byte_idx++;
                }
                Bs[150][col] = 0;
                Bs[151][col] = 0;
"""
    else:
        source += """
                    if (i != 25) {
                        Bs[i * 5 + 3][col] = base3_lut[b][3] * scale;
                        Bs[i * 5 + 4][col] = base3_lut[b][4] * scale;
                    }
                    byte_idx++;
                }
"""
    source += f"""
            }} else {{
                for (uint i = 0; i < {k_tile}; i++) {{
                    Bs[i][col] = 0;
                }}
            }}
        }}
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint k = 0; k < {k_tile}; k += 8) {{
            simdgroup_matrix<half, 8, 8> a[2];
            simdgroup_matrix<half, 8, 8> b[4];
            
            uint sg_row = sg_id * 16;
            
            simdgroup_load(a[0], &As[0][0], {k_tile}, ulong2(k, sg_row + 0));
            simdgroup_load(a[1], &As[0][0], {k_tile}, ulong2(k, sg_row + 8));
            
            simdgroup_load(b[0], &Bs[0][0], 32, ulong2(0, k));
            simdgroup_load(b[1], &Bs[0][0], 32, ulong2(8, k));
            simdgroup_load(b[2], &Bs[0][0], 32, ulong2(16, k));
            simdgroup_load(b[3], &Bs[0][0], 32, ulong2(24, k));
            
            for (int i=0; i<2; i++) {{
                for (int j=0; j<4; j++) {{
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }}
            }}
        }}
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    
    for (int i=0; i<2; i++) {{
        for (int j=0; j<4; j++) {{
            uint out_row = m_start + sg_id * 16 + i * 8;
            uint out_col = n_start + j * 8;
            
            if (out_row < {M} && out_col < {N}) {{
                if (out_row + 8 <= {M} && out_col + 8 <= {N}) {{
                    simdgroup_store(acc[i][j], &output[out_row * {N} + out_col], {N}, ulong2(0,0));
                }} else {{
                    threadgroup half temp_out[8][8];
                    simdgroup_store(acc[i][j], &temp_out[0][0], 8, ulong2(0,0));
                    for (int r=0; r<8; r++) {{
                        for (int c=0; c<8; c++) {{
                            if (out_row + r < {M} && out_col + c < {N}) {{
                                output[(out_row + r) * {N} + out_col + c] = temp_out[r][c];
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}
    """
    return source


def ternary_matmul(x: mx.array, w_packed: mx.array, group_size: int = None) -> mx.array:
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])
    if x_flat.dtype != mx.float16:
        x_flat = x_flat.astype(mx.float16)
        
    M = x_flat.shape[0]
    K = x_flat.shape[1]
    N = w_packed.shape[1]
    K_blocks = w_packed.shape[0]
    
    if group_size is None:
        group_size = K // K_blocks
        
    use_gemm = M > 16
    
    if use_gemm:
        kernel_type = "gemm"
        shape_key = (kernel_type, M, K, N, group_size)
        if shape_key not in _KERNEL_CACHE:
            source = get_gemm_source(M, K, N, group_size)
            kernel = fast.metal_kernel(
                name=f"ternary_{kernel_type}_{M}_{K}_{N}_{group_size}",
                input_names=["activations", "interleaved_weights"],
                output_names=["output"],
                source=source,
                header=_LUT_HEADER
            )
            _KERNEL_CACHE[shape_key] = kernel

        grid_x = ((N + 31) // 32) * 64
        grid_y = (M + 31) // 32
        out = _KERNEL_CACHE[shape_key](
            inputs=[x_flat, w_packed],
            grid=(grid_x, grid_y, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(M, N)],
            output_dtypes=[mx.float16]
        )
        res = out[0]
    else:
        # GEMV regime (M <= 16): Shape-aware hybrid dispatch
        if N <= 1024:
            # Narrow layers (e.g. Attention KV): SIMD Cooperative Warp reduction (0 barriers)
            kernel_type = "narrow_gemv"
            shape_key = (kernel_type, M, K, N, group_size)
            if shape_key not in _KERNEL_CACHE:
                source = get_narrow_gemv_source(M, K, N, group_size)
                kernel = fast.metal_kernel(
                    name=f"ternary_{kernel_type}_{M}_{K}_{N}_{group_size}",
                    input_names=["activations", "interleaved_weights"],
                    output_names=["output"],
                    source=source,
                    header=_LUT_HEADER
                )
                _KERNEL_CACHE[shape_key] = kernel

            tg_y = 4
            grid_y = ((N + tg_y - 1) // tg_y) * tg_y
            out = _KERNEL_CACHE[shape_key](
                inputs=[x_flat, w_packed],
                grid=(32, grid_y, M),
                threadgroup=(32, tg_y, 1),
                output_shapes=[(M, N)],
                output_dtypes=[mx.float16]
            )
            res = out[0]
        else:
            # Wide layers (e.g. MLP Gate/Up, MLP Down, Attention Q/O)
            if K >= 8192:
                # Deep contractions: hardware-elastic split-K reduction
                cores = _get_gpu_core_count()
                base_tgs = max(1, (N + 127) // 128)
                tgs_per_core = 8
                S = max(1, round((cores * tgs_per_core) / base_tgs))
                K_blocks = K // group_size
                S = min(S, K_blocks)
            else:
                S = 1

            if S > 1:
                s1_key = ("split_k_stage1", M, K, N, S, group_size)
                if s1_key not in _KERNEL_CACHE:
                    s1_src, _ = get_elastic_split_k_sources(M, K, N, S, group_size)
                    _KERNEL_CACHE[s1_key] = fast.metal_kernel(
                        name=f"ternary_split_k_stage1_{M}_{K}_{N}_{S}_{group_size}",
                        input_names=["activations", "interleaved_weights"],
                        output_names=["partial_output"],
                        source=s1_src,
                        header=_LUT_HEADER
                    )
                s2_key = ("split_k_stage2", M, N, S)
                if s2_key not in _KERNEL_CACHE:
                    _, s2_src = get_elastic_split_k_sources(M, K, N, S, group_size)
                    _KERNEL_CACHE[s2_key] = fast.metal_kernel(
                        name=f"ternary_split_k_stage2_{M}_{N}_{S}",
                        input_names=["partial_output"],
                        output_names=["output"],
                        source=s2_src
                    )

                grid_x = ((N + 127) // 128) * 128
                p = _KERNEL_CACHE[s1_key](
                    inputs=[x_flat, w_packed],
                    grid=(grid_x, S, M),
                    threadgroup=(128, 1, 1),
                    output_shapes=[(M, S, N)],
                    output_dtypes=[mx.float32]
                )[0]

                if N % 4 == 0:
                    tg_s2 = min(256, max(32, N // 4))
                    grid_s2 = (((N // 4) + tg_s2 - 1) // tg_s2) * tg_s2
                else:
                    tg_s2 = min(256, max(32, N))
                    grid_s2 = ((N + tg_s2 - 1) // tg_s2) * tg_s2

                out = _KERNEL_CACHE[s2_key](
                    inputs=[p],
                    grid=(grid_s2, M, 1),
                    threadgroup=(tg_s2, 1, 1),
                    output_shapes=[(M, N)],
                    output_dtypes=[mx.float16]
                )
                res = out[0]
            else:
                # 0-barrier SIMD-warp grid (S=1)
                kernel_type = "simd_gemv"
                shape_key = (kernel_type, M, K, N, group_size)
                if shape_key not in _KERNEL_CACHE:
                    s1_src, _ = get_elastic_split_k_sources(M, K, N, 1, group_size)
                    _KERNEL_CACHE[shape_key] = fast.metal_kernel(
                        name=f"ternary_{kernel_type}_{M}_{K}_{N}_{group_size}",
                        input_names=["activations", "interleaved_weights"],
                        output_names=["output"],
                        source=s1_src,
                        header=_LUT_HEADER
                    )
                grid_x = ((N + 127) // 128) * 128
                out = _KERNEL_CACHE[shape_key](
                    inputs=[x_flat, w_packed],
                    grid=(grid_x, 1, M),
                    threadgroup=(128, 1, 1),
                    output_shapes=[(M, N)],
                    output_dtypes=[mx.float16]
                )
                res = out[0]

    if len(orig_shape) > 2:
        return res.reshape(*orig_shape[:-1], N)
    elif len(orig_shape) == 1:
        return res.reshape(N)
    return res


def ternary_swiglu(
    x: mx.array,
    w_gate_packed: mx.array,
    w_up_packed: mx.array,
    group_size: int = None
) -> mx.array:
    """
    Fused Ternary SwiGLU activation and projection: SiLU(x @ w_gate) * (x @ w_up).

    Traverses both Gate and Up packed ternary weights in a single kernel pass,
    reusing loaded activation chunks across dual-issue ALU pipelines with an in-register
    SiLU product epilogue.

    Args:
        x: Input activations of shape (..., K).
        w_gate_packed: Gate projection packed weights of shape (K_blocks, N, 32).
        w_up_packed: Up projection packed weights of shape (K_blocks, N, 32).
        group_size: Scale group size (128 or 150). Inferred if None.

    Returns:
        mx.array: SwiGLU output activations of shape (..., N).
    """
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])
    if x_flat.dtype != mx.float16:
        x_flat = x_flat.astype(mx.float16)

    M = x_flat.shape[0]
    K = x_flat.shape[1]
    N = w_gate_packed.shape[1]
    K_blocks = w_gate_packed.shape[0]

    if group_size is None:
        group_size = K // K_blocks

    if M > 1:
        # Prefill regime (M > 1): Bifurcate to 2D tiled GEMM / AMX matrix multiplications
        import mlx.nn as nn
        gate = ternary_matmul(x_flat, w_gate_packed, group_size=group_size)
        up = ternary_matmul(x_flat, w_up_packed, group_size=group_size)
        res = (nn.silu(gate.astype(mx.float32)) * up.astype(mx.float32)).astype(mx.float16)
        if len(orig_shape) > 2:
            return res.reshape(*orig_shape[:-1], N)
        elif len(orig_shape) == 1:
            return res.reshape(N)
        return res

    shape_key = ("fused_swiglu", M, K, N, group_size)
    if shape_key not in _KERNEL_CACHE:
        source = get_fused_swiglu_source(M, K, N, group_size)
        kernel = fast.metal_kernel(
            name=f"ternary_fused_swiglu_{M}_{K}_{N}_{group_size}",
            input_names=["activations", "gate_weights", "up_weights"],
            output_names=["output"],
            source=source,
            header=_LUT_HEADER
        )
        _KERNEL_CACHE[shape_key] = kernel

    tg_x = 128
    grid_x = ((N + tg_x - 1) // tg_x) * tg_x
    out = _KERNEL_CACHE[shape_key](
        inputs=[x_flat, w_gate_packed, w_up_packed],
        grid=(grid_x, M, 1),
        threadgroup=(tg_x, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[mx.float16]
    )
    res = out[0]

    if len(orig_shape) > 2:
        return res.reshape(*orig_shape[:-1], N)
    elif len(orig_shape) == 1:
        return res.reshape(N)
    return res


def ternary_fused_kv(
    x: mx.array,
    w_k_packed: mx.array,
    w_v_packed: mx.array,
    group_size: int = None
) -> tuple[mx.array, mx.array]:
    """
    Fused Key and Value projections: x @ w_k and x @ w_v.

    Reuses loaded activation vector chunks in registers across dual-issue ALU pipelines,
    evaluating both projections concurrently in a single kernel dispatch.

    Args:
        x: Input activations of shape (..., K).
        w_k_packed: Key projection packed weights of shape (K_blocks, N, 32).
        w_v_packed: Value projection packed weights of shape (K_blocks, N, 32).
        group_size: Scale group size (128 or 150). Inferred if None.

    Returns:
        tuple[mx.array, mx.array]: (out_k, out_v) activations each of shape (..., N).
    """
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])
    if x_flat.dtype != mx.float16:
        x_flat = x_flat.astype(mx.float16)

    M = x_flat.shape[0]
    K = x_flat.shape[1]
    N = w_k_packed.shape[1]
    K_blocks = w_k_packed.shape[0]

    if group_size is None:
        group_size = K // K_blocks

    if M > 1:
        # Prefill regime (M > 1): Bifurcate to 2D tiled GEMM / AMX matrix multiplications
        out_k = ternary_matmul(x_flat, w_k_packed, group_size=group_size)
        out_v = ternary_matmul(x_flat, w_v_packed, group_size=group_size)
        if len(orig_shape) > 2:
            return out_k.reshape(*orig_shape[:-1], N), out_v.reshape(*orig_shape[:-1], N)
        elif len(orig_shape) == 1:
            return out_k.reshape(N), out_v.reshape(N)
        return out_k, out_v

    shape_key = ("fused_kv", M, K, N, group_size)
    if shape_key not in _KERNEL_CACHE:
        source = get_fused_kv_source(M, K, N, group_size)
        kernel = fast.metal_kernel(
            name=f"ternary_fused_kv_{M}_{K}_{N}_{group_size}",
            input_names=["activations", "k_weights", "v_weights"],
            output_names=["out_k", "out_v"],
            source=source,
            header=_LUT_HEADER
        )
        _KERNEL_CACHE[shape_key] = kernel

    tg_x = 128
    grid_x = ((N + tg_x - 1) // tg_x) * tg_x
    out_k, out_v = _KERNEL_CACHE[shape_key](
        inputs=[x_flat, w_k_packed, w_v_packed],
        grid=(grid_x, M, 1),
        threadgroup=(tg_x, 1, 1),
        output_shapes=[(M, N), (M, N)],
        output_dtypes=[mx.float16, mx.float16]
    )

    if len(orig_shape) > 2:
        return out_k.reshape(*orig_shape[:-1], N), out_v.reshape(*orig_shape[:-1], N)
    elif len(orig_shape) == 1:
        return out_k.reshape(N), out_v.reshape(N)
    return out_k, out_v


def ternary_fused_qkv(
    x: mx.array,
    w_qkv_packed: mx.array,
    group_size: int = None
) -> mx.array:
    """
    Unified Query, Key, and Value projection: x @ w_qkv.

    Executes a single 0-barrier wide GEMV dispatch over concatenated [Q, K, V]
    interleaved weights, maximizing GPU occupancy and memory reuse.

    Args:
        x: Input activations of shape (..., K).
        w_qkv_packed: Concatenated QKV packed weights of shape (K_blocks, N_qkv, 32).
        group_size: Scale group size (128 or 150). Inferred if None.

    Returns:
        mx.array: Contiguous output tensor of shape (..., N_qkv).
    """
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])
    if x_flat.dtype != mx.float16:
        x_flat = x_flat.astype(mx.float16)

    M = x_flat.shape[0]
    K = x_flat.shape[1]
    N = w_qkv_packed.shape[1]
    K_blocks = w_qkv_packed.shape[0]

    if group_size is None:
        group_size = K // K_blocks

    if M > 1:
        # Prefill regime (M > 1): Bifurcate to 2D tiled GEMM / AMX matrix multiplications
        res = ternary_matmul(x_flat, w_qkv_packed, group_size=group_size)
        if len(orig_shape) > 2:
            return res.reshape(*orig_shape[:-1], N)
        elif len(orig_shape) == 1:
            return res.reshape(N)
        return res

    shape_key = ("fused_qkv", M, K, N, group_size)
    if shape_key not in _KERNEL_CACHE:
        source = get_fused_qkv_source(M, K, N, group_size)
        kernel = fast.metal_kernel(
            name=f"ternary_fused_qkv_{M}_{K}_{N}_{group_size}",
            input_names=["activations", "interleaved_weights"],
            output_names=["output"],
            source=source,
            header=_LUT_HEADER
        )
        _KERNEL_CACHE[shape_key] = kernel

    tg_x = 64
    grid_x = ((N + tg_x - 1) // tg_x) * tg_x
    out = _KERNEL_CACHE[shape_key](
        inputs=[x_flat, w_qkv_packed],
        grid=(grid_x, M, 1),
        threadgroup=(tg_x, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[mx.float16]
    )
    res = out[0]

    if len(orig_shape) > 2:
        return res.reshape(*orig_shape[:-1], N)
    elif len(orig_shape) == 1:
        return res.reshape(N)
    return res
