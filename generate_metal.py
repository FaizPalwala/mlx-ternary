import os

metal_code = """
#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

constant float16_t base3_lut[243][5] = {
#include "lut.h"
};

"""

def gen_kernel(G_SIZE):
    if G_SIZE == 150:
        k_padded = 152
        num_bytes = 30
    else:
        k_padded = 128
        num_bytes = 26
        
    code = f"""
kernel void ternary_matmul_kernel_{G_SIZE}(
    device const float16_t* activations [[buffer(0)]], 
    device const uint8_t* interleaved_weights [[buffer(1)]], 
    device float16_t* output [[buffer(2)]], 
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {{
    uint lane_id = gid.x;      
    uint col = gid.y;          
    uint row = gid.z;          
    
    if (row >= M || col >= N || lane_id >= 32) {{
        return;
    }}

    uint K_blocks = K / {G_SIZE};
    uint blocks_per_thread = (K_blocks + 31) / 32;
    uint start_k = lane_id * blocks_per_thread;
    uint end_k = min(start_k + blocks_per_thread, K_blocks);

    float16_t acc = 0;

    for (uint k = start_k; k < end_k; k++) {{
        uint block_offset_uint4 = (k * N + col) * 2;
        
        device const uint4* w_ptr = (device const uint4*)interleaved_weights;
        uint4 vec0 = w_ptr[block_offset_uint4];
        uint4 vec1 = w_ptr[block_offset_uint4 + 1];
        
        float16_t scale = as_type<half>((uint16_t)(vec0.x & 0xFFFF));
        
        float16_t block_acc = 0;
        uint x_idx = row * K + k * {G_SIZE};
        
        uint32_t chunks[8] = {{ vec0.x, vec0.y, vec0.z, vec0.w, vec1.x, vec1.y, vec1.z, vec1.w }};
        
        uint byte_idx = 2; 
        
        #pragma unroll({num_bytes})
        for (uint i = 0; i < {num_bytes}; i++) {{
            uint chunk_idx = byte_idx / 4;
            uint byte_offset = (byte_idx % 4) * 8;
            uint8_t b = (chunks[chunk_idx] >> byte_offset) & 0xFF;
            
            float16_t w0 = base3_lut[b][0];
            float16_t w1 = base3_lut[b][1];
            float16_t w2 = base3_lut[b][2];
            float16_t w3 = base3_lut[b][3];
            float16_t w4 = base3_lut[b][4];
            
            float16_t x0 = activations[x_idx + i * 5 + 0];
            float16_t x1 = activations[x_idx + i * 5 + 1];
            float16_t x2 = activations[x_idx + i * 5 + 2];
"""
    if G_SIZE == 150:
        code += """
            float16_t x3 = activations[x_idx + i * 5 + 3];
            float16_t x4 = activations[x_idx + i * 5 + 4];
            block_acc += x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3 + x4 * w4;
"""
    else: # 128
        code += """
            if (i == 25) {
                block_acc += x0 * w0 + x1 * w1 + x2 * w2; // 25*5 = 125, +3 = 128
            } else {
                float16_t x3 = activations[x_idx + i * 5 + 3];
                float16_t x4 = activations[x_idx + i * 5 + 4];
                block_acc += x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3 + x4 * w4;
            }
"""
    code += """
            byte_idx++;
        }
        
        acc += block_acc * scale;
    }

    float16_t total = simd_sum(acc);
    
    if (lane_id == 0) {
        output[row * N + col] = total;
    }
}

kernel void ternary_matmul_gemm_kernel_""" + str(G_SIZE) + """(
    device const float16_t* activations [[buffer(0)]], 
    device const uint8_t* interleaved_weights [[buffer(1)]], 
    device float16_t* output [[buffer(2)]], 
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint sg_id = tid / 32;
    uint lane_id = tid % 32;
    
    uint m_start = gid.y * 32;
    uint n_start = gid.x * 32;
    
    threadgroup half As[32][""" + str(k_padded) + """];
    threadgroup half Bs[""" + str(k_padded) + """][32];
    
    simdgroup_matrix<float, 8, 8> acc[2][4];
    for (int i=0; i<2; i++) {
        for (int j=0; j<4; j++) {
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0);
        }
    }
    
    uint K_blocks = K / """ + str(G_SIZE) + """;
    
    for (uint k_block = 0; k_block < K_blocks; k_block++) {
        for(uint i = tid; i < 32 * """ + str(k_padded) + """; i += 64) {
            uint row = i / """ + str(k_padded) + """;
            uint col = i % """ + str(k_padded) + """;
            if (col < """ + str(G_SIZE) + """ && (m_start + row) < M) {
                As[row][col] = activations[(m_start + row) * K + (k_block * """ + str(G_SIZE) + """ + col)];
            } else {
                As[row][col] = 0;
            }
        }
        
        if (sg_id == 0) {
            uint col = lane_id; 
            uint global_n = n_start + col;
            
            if (global_n < N) {
                uint block_offset = (k_block * N + global_n) * 2; 
                device const uint4* w_ptr = (device const uint4*)interleaved_weights;
                
                uint4 vec0 = w_ptr[block_offset];
                uint4 vec1 = w_ptr[block_offset + 1];
                
                float16_t scale = as_type<half>((uint16_t)(vec0.x & 0xFFFF));
                uint32_t chunks[8] = { vec0.x, vec0.y, vec0.z, vec0.w, vec1.x, vec1.y, vec1.z, vec1.w };
                
                uint byte_idx = 2; 
                #pragma unroll(""" + str(num_bytes) + """)
                for (uint i = 0; i < """ + str(num_bytes) + """; i++) {
                    uint chunk_idx = byte_idx / 4;
                    uint byte_offset = (byte_idx % 4) * 8;
                    uint8_t b = (chunks[chunk_idx] >> byte_offset) & 0xFF;
                    
                    Bs[i * 5 + 0][col] = base3_lut[b][0] * scale;
                    Bs[i * 5 + 1][col] = base3_lut[b][1] * scale;
                    Bs[i * 5 + 2][col] = base3_lut[b][2] * scale;
"""
    if G_SIZE == 150:
        code += """
                    Bs[i * 5 + 3][col] = base3_lut[b][3] * scale;
                    Bs[i * 5 + 4][col] = base3_lut[b][4] * scale;
"""
    else: # 128
        code += """
                    if (i != 25) {
                        Bs[i * 5 + 3][col] = base3_lut[b][3] * scale;
                        Bs[i * 5 + 4][col] = base3_lut[b][4] * scale;
                    }
"""
    code += """
                    byte_idx++;
                }
"""
    if G_SIZE == 150:
        code += """
                Bs[150][col] = 0;
                Bs[151][col] = 0;
"""
    code += """
            } else {
                for (uint i = 0; i < """ + str(k_padded) + """; i++) {
                    Bs[i][col] = 0;
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint k = 0; k < """ + str(k_padded) + """; k += 8) {
            simdgroup_matrix<half, 8, 8> a[2];
            simdgroup_matrix<half, 8, 8> b[4];
            
            uint sg_row = sg_id * 16;
            
            simdgroup_load(a[0], &As[0][0], """ + str(k_padded) + """, ulong2(k, sg_row + 0));
            simdgroup_load(a[1], &As[0][0], """ + str(k_padded) + """, ulong2(k, sg_row + 8));
            
            simdgroup_load(b[0], &Bs[0][0], 32, ulong2(0, k));
            simdgroup_load(b[1], &Bs[0][0], 32, ulong2(8, k));
            simdgroup_load(b[2], &Bs[0][0], 32, ulong2(16, k));
            simdgroup_load(b[3], &Bs[0][0], 32, ulong2(24, k));
            
            for (int i=0; i<2; i++) {
                for (int j=0; j<4; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    for (int i=0; i<2; i++) {
        for (int j=0; j<4; j++) {
            uint out_row = m_start + sg_id * 16 + i * 8;
            uint out_col = n_start + j * 8;
            
            if (out_row < M && out_col < N) {
                simdgroup_matrix<half, 8, 8> acc_half(acc[i][j]);
                
                if (out_row + 8 <= M && out_col + 8 <= N) {
                    simdgroup_store(acc_half, &output[out_row * N + out_col], N, ulong2(0,0));
                } else {
                    threadgroup half temp_out[8][8];
                    simdgroup_store(acc_half, &temp_out[0][0], 8, ulong2(0,0));
                    for (int r=0; r<8; r++) {
                        for (int c=0; c<8; c++) {
                            if (out_row + r < M && out_col + c < N) {
                                output[(out_row + r) * N + out_col + c] = temp_out[r][c];
                            }
                        }
                    }
                }
            }
        }
    }
}
"""
    return code

with open("src/mlx_ternary/cpp/ternary.metal", "w") as f:
    f.write(metal_code)
    f.write(gen_kernel(150))
    f.write(gen_kernel(128))

print("Generated ternary.metal with 150 and 128 kernels!")
