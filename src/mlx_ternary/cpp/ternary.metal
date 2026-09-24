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

kernel void ternary_matmul_kernel(
    device const float16_t* activations [[buffer(0)]], // shape: (M, K)
    device const uint8_t* packed_weights [[buffer(1)]], // shape: (K/4, N) - 4 weights per byte
    device const float16_t* scales [[buffer(2)]], // shape: (N) - scaling factor for each column
    device float16_t* output [[buffer(3)]], // shape: (M, N)
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y; // M dimension
    
    // We vectorize across N. Each thread processes 16 columns.
    // gid.x is the thread index in the grid.
    uint vec_col = gid.x; 
    
    if (row >= M || vec_col * 16 >= N) {
        return;
    }

    // Accumulators for 16 columns
    float16_t acc[16] = {0};

    // Cast weights to uint4 (128-bit) for max bus saturation
    device const uint4* w_vec = (device const uint4*)packed_weights;
    uint N_vec = N / 16; 

    // Loop over K in blocks of 4 (since 1 byte = 4 weights down the K dimension)
    uint K_blocks = K / 4;
    for (uint k = 0; k < K_blocks; k++) {
        // Read 4 activations (shared across all columns)
        // The L1 cache will broadcast this to all threads in the SIMD group
        uint x_idx = row * K + k * 4;
        float16_t x0 = activations[x_idx + 0];
        float16_t x1 = activations[x_idx + 1];
        float16_t x2 = activations[x_idx + 2];
        float16_t x3 = activations[x_idx + 3];

        // Read 16 bytes of weights (1 byte per column)
        uint4 w_bytes_uint4 = w_vec[k * N_vec + vec_col];
        thread uint8_t* bytes = (thread uint8_t*)&w_bytes_uint4;

        // Process all 16 columns
        for (uint c = 0; c < 16; c++) {
            uint8_t b = bytes[c];
            
            // Extract 2-bit values
            uint8_t v0 = b & 0x3;
            uint8_t v1 = (b >> 2) & 0x3;
            uint8_t v2 = (b >> 4) & 0x3;
            uint8_t v3 = (b >> 6) & 0x3;

            // Map to float16: 0->0.0, 1->1.0, 2->-1.0
            float16_t w0 = (v0 == 0) ? 0.0h : ((v0 == 1) ? 1.0h : -1.0h);
            float16_t w1 = (v1 == 0) ? 0.0h : ((v1 == 1) ? 1.0h : -1.0h);
            float16_t w2 = (v2 == 0) ? 0.0h : ((v2 == 1) ? 1.0h : -1.0h);
            float16_t w3 = (v3 == 0) ? 0.0h : ((v3 == 1) ? 1.0h : -1.0h);

            acc[c] += x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3;
        }
    }

    // Apply scaling and write out 16 elements
    for (uint c = 0; c < 16; c++) {
        uint actual_col = vec_col * 16 + c;
        if (actual_col < N) {
            output[row * N + actual_col] = acc[c] * scales[actual_col];
        }
    }
}


