#include "TernaryOp.h"

namespace mlx_ternary {

TernaryMatMul::TernaryMatMul() {}

std::unique_ptr<mlx::core::CustomOp> TernaryMatMul::clone() const {
    return std::make_unique<TernaryMatMul>(*this);
}

std::vector<mlx::core::array> TernaryMatMul::eval_cpu(
    const std::vector<mlx::core::array>& inputs,
    const std::vector<mlx::core::array>& outputs) {
    // CPU Fallback: Throw or implement slow reference addition-only matmul
    throw std::runtime_error("TernaryMatMul::eval_cpu not yet implemented. Use GPU.");
}

// eval_gpu implementation
std::vector<mlx::core::array> TernaryMatMul::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    const std::vector<mlx::core::array>& outputs) {
    
    auto& x = inputs[0]; // Activations (M, K)
    auto& w = inputs[1]; // Packed Weights (K/4, N)
    auto& out = outputs[0]; // Output (M, N)

    // 1. Get MLX Metal Device and stream
    // auto& d = mlx::core::metal::device(x.device());
    // auto* compute_encoder = d.get_command_encoder(d.stream());

    // 2. Fetch the pre-compiled pipeline state for "ternary_matmul_kernel"
    // auto* cpo = d.get_kernel("ternary_matmul_kernel");
    // compute_encoder->setComputePipelineState(cpo);

    // 3. Bind buffers
    // compute_encoder->setBuffer(static_cast<MTL::Buffer*>(x.buffer().ptr()), x.offset(), 0);
    // compute_encoder->setBuffer(static_cast<MTL::Buffer*>(w.buffer().ptr()), w.offset(), 1);
    // compute_encoder->setBuffer(static_cast<MTL::Buffer*>(out.buffer().ptr()), out.offset(), 2);
    
    // 4. Bind shape constants
    // uint M = x.shape(0);
    // uint K = x.shape(1);
    // uint N = w.shape(1);
    // compute_encoder->setBytes(&M, sizeof(uint), 3);
    // ...

    // 5. Dispatch Threadgroups
    // MTL::Size grid_size = MTL::Size::Make(N, M, 1);
    // MTL::Size group_size = MTL::Size::Make(16, 16, 1); // Tune this
    // compute_encoder->dispatchThreads(grid_size, group_size);

    throw std::runtime_error("TernaryMatMul::eval_gpu: Metal dispatch logic written but requires MLX MTL headers to link.");
}

mlx::core::array ternary_matmul(const mlx::core::array& x, const mlx::core::array& w) {
    // Input validation
    if (x.ndim() < 2 || w.ndim() < 2) {
        throw std::invalid_argument("[mlx_ternary] Inputs must be at least 2D.");
    }
    
    // Output shape calculation: standard matmul rules
    // For a packed weight matrix, we need to know the 'true' unpacked shape.
    // Assuming for now w is unpacked or metadata is stored out of band.
    std::vector<int> out_shape = x.shape();
    out_shape.back() = w.shape().back(); // x: (..., M, K), w: (K, N) -> out: (..., M, N)

    // Instantiate our custom op
    auto op = std::make_shared<TernaryMatMul>();
    
    // Insert into MLX graph
    return mlx::core::CustomOp::create(
        "TernaryMatMul", 
        op, 
        {x, w}, 
        {out_shape}, 
        {x.dtype()} // Output dtype matches input activations (usually float16)
    )[0];
}

} // namespace mlx_ternary
