#include <metal_stdlib>
using namespace metal;

/* =====================================================================
 * FUNCTIONAL SPECIFICATION: Ternary MatMul Kernel (Un-Fused v1.0)
 * =====================================================================
 * This kernel performs a forward-pass matrix multiplication optimized
 * for 1.58-bit ternary language models.
 * 
 * DESIGN DECISIONS:
 * 1. 2-bit Packing: We assume weights are packed 4-to-a-byte (uint8_t).
 *    Mapping: 00 -> 0.0, 01 -> 1.0, 10 -> -1.0.
 * 2. Un-Fused Scaling: This kernel ONLY computes the integer accumulation
 *    of the ternary weights. It DOES NOT apply the fp16 scaling factor.
 *    Scaling must be handled upstream in the MLX computation graph.
 * 3. ALU Starvation: By branching on the weight value (-1, 0, 1), we
 *    execute pure addition/subtraction, bypassing FP16 multiplication.
 * =====================================================================
 */

inline float16_t unpack_ternary(uint8_t packed, uint index) {
    uint8_t val = (packed >> (index * 2)) & 0x3;
    if (val == 1) return 1.0h;
    if (val == 2) return -1.0h;
    return 0.0h;
}

// Bypasses FP16 multipliers using control flow / addition
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
    uint col = gid.x; // N dimension

    if (row >= M || col >= N) {
        return;
    }

    float16_t acc = 0.0h;

    // Dot product loop
    for (uint k = 0; k < K; k += 4) {
        uint weight_idx = (k / 4) * N + col; // Assuming column-major or similar packing
        uint8_t packed_w = packed_weights[weight_idx];

        for (uint i = 0; i < 4; i++) {
            if (k + i < K) {
                float16_t w_val = unpack_ternary(packed_w, i);
                float16_t x_val = activations[row * K + (k + i)];
                
                // Pure addition/subtraction branch
                // (In a heavily optimized kernel, this would use bit-tricks to avoid branching)
                if (w_val > 0.5h) {
                    acc += x_val;
                } else if (w_val < -0.5h) {
                    acc -= x_val;
                }
            }
        }
    }

    // Apply scale and write out
    output[row * N + col] = acc * scales[col];
}
