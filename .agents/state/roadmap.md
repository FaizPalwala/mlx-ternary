# Product Roadmap

## Near-Term (v1.0.0)
- [ ] Implement foundational C++ Metal kernels for ternary addition (bypassing FP16 multiplication).
- [ ] Expose `load()` and `generate()` via Python bindings.
- [ ] Validate outputs against CPU-based `llama.cpp` (PrismML fork) implementations to ensure accuracy.

## Mid-Term (v1.1.0)
- [ ] Add support for mixed-precision caching (FP16 KV Cache alongside ternary weights).
- [ ] Implement structured JSON generation endpoints for agentic workflows.
- [ ] Optimize memory mapping to support 70B+ class ternary models in 16GB-24GB unified memory.

## Long-Term / Moonshots
- [ ] Publish academic or engineering blog post detailing the M4 Pro TPS benchmarks.
- [ ] Official integration push upstream to MLX core repository.
