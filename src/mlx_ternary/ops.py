import mlx.core as mx
import mlx.core.fast as fast

_KERNEL_CACHE = {}

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
        grid = (grid_x, grid_y, 1)
        threadgroup = (64, 1, 1)
        out_shapes = [(M, N)]
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
            grid = (32, grid_y, M)
            threadgroup = (32, tg_y, 1)
            out_shapes = [(M, N)]
        else:
            # Wide layers (e.g. MLP Gate/Up, MLP Down, Attention Q/O): Shared SRAM activation caching
            kernel_type = "wide_gemv"
            shape_key = (kernel_type, M, K, N, group_size)
            if shape_key not in _KERNEL_CACHE:
                source = get_wide_gemv_source(M, K, N, group_size)
                kernel = fast.metal_kernel(
                    name=f"ternary_{kernel_type}_{M}_{K}_{N}_{group_size}",
                    input_names=["activations", "interleaved_weights"],
                    output_names=["output"],
                    source=source,
                    header=_LUT_HEADER
                )
                _KERNEL_CACHE[shape_key] = kernel

            tg_size = 128
            grid_x = ((N + tg_size - 1) // tg_size) * tg_size
            grid = (grid_x, M, 1)
            threadgroup = (tg_size, 1, 1)
            out_shapes = [(M, N)]

    out = _KERNEL_CACHE[shape_key](
        inputs=[x_flat, w_packed],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=out_shapes,
        output_dtypes=[mx.float16]
    )
    
    res = out[0]
    if len(orig_shape) > 2:
        return res.reshape(*orig_shape[:-1], N)
    elif len(orig_shape) == 1:
        return res.reshape(N)
    return res
