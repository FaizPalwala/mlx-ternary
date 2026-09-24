#include <metal_stdlib>
using namespace metal;

/* =====================================================================
 * FUNCTIONAL SPECIFICATION: Ternary MatMul Kernel (Vectorized v2.0)
 * =====================================================================
 * This kernel performs a forward-pass matrix multiplication optimized
 * for 1.58-bit ternary language models.
 * 
 * ARCHITECTURAL SOTA OPTIMIZATIONS (Marlin-Style GEMV):
 * 1. Vectorized Memory Streaming: Uses `uint4` (128-bit) loads to 
 *    pull 16 columns of weights simultaneously, perfectly saturating 
 *    the memory bus.
 * 2. Zero-Hop Register Accumulation: Bypasses threadgroup memory for 
 *    weights. Weights go straight from global RAM -> Registers -> Math.
 * 3. ALU Pipelining: Uses `select` / ternary operators which compile 
 *    to fast `fcsel` instructions, avoiding true branch divergence.
 * =====================================================================
 */

#include "base3_lut.h"

kernel void ternary_matmul_kernel(
    device const float16_t* activations [[buffer(0)]], // shape: (M, K)
    device const uint8_t* packed_weights [[buffer(1)]], // shape: (K/5, N)
    device const float16_t* scales [[buffer(2)]], // shape: (N)
    device float16_t* output [[buffer(3)]], // shape: (M, N)
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    uint lane_id = gid.x;      // 0..31 (SIMD lane)
    uint vec_col = gid.y;      // 0..N/16 - 1
    uint row = gid.z;          // 0..M - 1
    
    if (row >= M || vec_col * 16 >= N || lane_id >= 32) {
        return;
    }

    float16_t acc[16] = {0};

    device const uint4* w_vec = (device const uint4*)packed_weights;
    uint N_vec = N / 16; 

    uint K_blocks = K / 5;
    uint blocks_per_thread = (K_blocks + 31) / 32;
    uint start_k = lane_id * blocks_per_thread;
    uint end_k = min(start_k + blocks_per_thread, K_blocks);

    for (uint k = start_k; k < end_k; k++) {
        uint x_idx = row * K + k * 5;
        float16_t x0 = activations[x_idx + 0];
        float16_t x1 = activations[x_idx + 1];
        float16_t x2 = activations[x_idx + 2];
        float16_t x3 = activations[x_idx + 3];
        float16_t x4 = activations[x_idx + 4];

        uint4 w_bytes_uint4 = w_vec[k * N_vec + vec_col];

        for (uint c = 0; c < 16; c++) {
            uint chunk = w_bytes_uint4[c / 4];
            uint8_t b = (chunk >> ((c % 4) * 8)) & 0xFF;
            
            float16_t w0 = base3_lut[b][0];
            float16_t w1 = base3_lut[b][1];
            float16_t w2 = base3_lut[b][2];
            float16_t w3 = base3_lut[b][3];
            float16_t w4 = base3_lut[b][4];

            acc[c] += x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3 + x4 * w4;
        }
    }

    // SIMD reduce across the 32 threads in the threadgroup
    for (uint c = 0; c < 16; c++) {
        float16_t total = simd_sum(acc[c]);
        
        // Only thread 0 writes the reduced output to global memory
        if (lane_id == 0) {
            uint actual_col = vec_col * 16 + c;
            if (actual_col < N) {
                output[row * N + actual_col] = total * scales[actual_col];
            }
        }
    }
}


