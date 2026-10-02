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

def get_gemv_source(M, K, N, G_SIZE, S=1):
    num_bytes = 26 if G_SIZE == 128 else 30
    
    source = f"""
    uint tid = thread_position_in_grid.x;
    uint s_idx = thread_position_in_grid.y;
    uint row = thread_position_in_grid.z;
    
    if (row >= {M} || s_idx >= {S} || tid >= {N}) {{
        return;
    }}

    uint col = tid;
    uint K_blocks = {K} / {G_SIZE};
    uint blocks_per_slice = (K_blocks + {S} - 1) / {S};
    uint k_start = s_idx * blocks_per_slice;
    uint k_end = min(k_start + blocks_per_slice, K_blocks);

    half acc = 0;

    device const uchar* w_base = (device const uchar*)interleaved_weights;
    device const half* act_ptr = activations + row * {K};

    for (uint k = k_start; k < k_end; k++) {{
        device const uchar* block_ptr = w_base + (k * {N} + col) * 32;
        half scale = *((device const half*)block_ptr);
        
        half block_acc = 0;
        uint x_base = k * {G_SIZE};

        #pragma unroll({num_bytes})
        for (uint i = 0; i < {num_bytes}; i++) {{
            uint8_t b = block_ptr[2 + i];
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

    output[row * ({S} * {N}) + s_idx * {N} + col] = acc;
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
    kernel_type = "gemm" if use_gemm else "gemv"
    
    split_k = 1
    if not use_gemm and M <= 2 and N <= 1024 and K_blocks >= 8:
        split_k = min(8, K_blocks)
        
    shape_key = (kernel_type, M, K, N, group_size, split_k)
    
    if shape_key not in _KERNEL_CACHE:
        if use_gemm:
            source = get_gemm_source(M, K, N, group_size)
        else:
            source = get_gemv_source(M, K, N, group_size, S=split_k)
            
        kernel = fast.metal_kernel(
            name=f"ternary_{kernel_type}_{M}_{K}_{N}_{group_size}_s{split_k}",
            input_names=["activations", "interleaved_weights"],
            output_names=["output"],
            source=source,
            header=_LUT_HEADER
        )
        _KERNEL_CACHE[shape_key] = kernel

    if use_gemm:
        grid_x = ((N + 31) // 32) * 64
        grid_y = (M + 31) // 32
        grid = (grid_x, grid_y, 1)
        threadgroup = (64, 1, 1)
        out_shapes = [(M, N)]
    else:
        grid_x = ((N + 63) // 64) * 64
        grid = (grid_x, split_k, M)
        threadgroup = (64, 1, 1)
        out_shapes = [(M, split_k, N)] if split_k > 1 else [(M, N)]

    out = _KERNEL_CACHE[shape_key](
        inputs=[x_flat, w_packed],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=out_shapes,
        output_dtypes=[mx.float16]
    )
    
    res = out[0].sum(axis=1) if split_k > 1 else out[0]
    if len(orig_shape) > 2:
        return res.reshape(*orig_shape[:-1], N)
    elif len(orig_shape) == 1:
        return res.reshape(N)
    return res
