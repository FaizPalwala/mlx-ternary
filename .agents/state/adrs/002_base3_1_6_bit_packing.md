# Architecture Decision Record: 1.6-bit Base-3 Ternary Encoding

## Context
PrismML achieved extreme memory-bandwidth efficiency for Bonsai 2 by using 1.76-bit packing. To surpass this, we need to minimize the memory footprint of ternary weights $\{-1, 0, 1\}$ to the absolute physical minimum.

## Decision
We implemented a **1.6-bit Base-3 Packing** architecture with a **Constant-Memory Decompression LUT**.

1. **Information Theory Limit**: 
   A ternary weight contains $\log_2(3) \approx 1.58$ bits of information. 
   By packing 5 ternary weights into a single 8-bit byte ($3^5 = 243 \le 256$), we achieve $8/5 = 1.6$ bits per weight. This is practically at the theoretical Shannon limit and beats PrismML's 1.76 bits by 10%.

2. **Zero-Compute Decompression**:
   Instead of using integer modulo and division in the GPU thread (`val % 3`), which consumes ALUs and slows down memory streaming, we generated a 256-element look-up table (LUT).
   The LUT is placed in the Metal `constant` address space, meaning it is permanently pinned in the GPU's L1/L2 cache and broadcast instantly to all threads.

3. **Vectorized uint4 Streaming**:
   The Metal kernel loads 16 bytes (`uint4`) simultaneously per thread, instantly querying the LUT 16 times to decompress 80 weights in just a few clock cycles.

## Results
- **Theoretical Footprint**: 20% smaller than 2-bit quantization, and 10% smaller than PrismML's 1.76-bit format.
- **Micro-Benchmark**: Down to `0.052 ms` per layer (M=1, K=4096, N=4096) on the M4 Pro.
- **Next Steps**: Multi-threaded block accumulation to fully saturate the 273 GB/s memory bandwidth limit of the M4 Pro chip.
