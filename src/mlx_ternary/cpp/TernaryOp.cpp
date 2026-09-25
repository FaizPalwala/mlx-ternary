#include "TernaryOp.h"
#include "ternary_metal.h" 

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace mlx_ternary {

void TernaryMatMul::eval_cpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {
    throw std::runtime_error("TernaryMatMul::eval_cpu not yet implemented. Use GPU.");
}

void TernaryMatMul::eval_gpu(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {
    
#ifdef _METAL_
    auto& x = inputs[0];
    auto& w_interleaved = inputs[1]; 
    auto& out = outputs[0];

    out.set_data(mlx::core::allocator::malloc(out.nbytes()));

    auto& s = stream();
    auto& d = mlx::core::metal::device(s.device);

    auto lib = d.get_library("mlx_ternary", [&]() {
        return std::string(ternary_metal_source);
    });

    uint M = 1;
    for (int i = 0; i < x.ndim() - 1; ++i) M *= x.shape(i);
    uint K = x.shape().back();
    uint N = w_interleaved.shape(1);

    std::string dec_name = "ternary_matmul_kernel_" + std::to_string(group_size_);
    std::string amx_name = "ternary_matmul_gemm_kernel_" + std::to_string(group_size_);

    if (M <= 16) {
        auto kernel = d.get_kernel(dec_name, lib);
        auto& compute_encoder = mlx::core::metal::get_command_encoder(s);
        compute_encoder.set_compute_pipeline_state(kernel);

        compute_encoder.set_input_array(x, 0);
        compute_encoder.set_input_array(w_interleaved, 1);
        compute_encoder.set_output_array(out, 2);

        compute_encoder.set_bytes(M, 3);
        compute_encoder.set_bytes(K, 4);
        compute_encoder.set_bytes(N, 5);

        MTL::Size grid_dims = MTL::Size(32, N, M);
        MTL::Size group_dims = MTL::Size(32, 1, 1);

        compute_encoder.dispatch_threads(grid_dims, group_dims);
    } else {
        auto kernel = d.get_kernel(amx_name, lib);
        auto& compute_encoder = mlx::core::metal::get_command_encoder(s);
        compute_encoder.set_compute_pipeline_state(kernel);

        compute_encoder.set_input_array(x, 0);
        compute_encoder.set_input_array(w_interleaved, 1);
        compute_encoder.set_output_array(out, 2);

        compute_encoder.set_bytes(M, 3);
        compute_encoder.set_bytes(K, 4);
        compute_encoder.set_bytes(N, 5);

        uint BM = 32;
        uint BN = 32;
        MTL::Size grid_dims = MTL::Size((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
        MTL::Size group_dims = MTL::Size(64, 1, 1);

        compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    }
#else
    throw std::runtime_error("TernaryMatMul::eval_gpu: compiled without Metal support.");
#endif
}

mlx::core::array ternary_matmul(
    const mlx::core::array& x,
    const mlx::core::array& w_interleaved,
    uint32_t group_size
) {
    if (group_size != 128 && group_size != 150) {
        throw std::invalid_argument("[mlx_ternary] Only group_size 128 and 150 are natively supported.");
    }

    auto s = mlx::core::default_stream(mlx::core::default_device());
    if (x.ndim() < 2 || w_interleaved.ndim() < 3) {
        throw std::invalid_argument("[mlx_ternary] Inputs must be appropriately sized.");
    }

    uint K = x.shape().back();
    uint pad_K = (group_size - (K % group_size)) % group_size;
    
    mlx::core::array x_padded = x;
    if (pad_K > 0) {
        std::vector<std::pair<int, int>> pad_widths(x.ndim(), {0, 0});
        pad_widths.back() = {0, pad_K};
        x_padded = mlx::core::pad(x, pad_widths, mlx::core::array(0.0f, x.dtype()), "constant", s);
    }

    auto x_c = mlx::core::contiguous(x_padded, false, s);
    auto w_c = mlx::core::contiguous(w_interleaved, false, s);
    
    auto out_shape = x.shape();
    out_shape.back() = w_interleaved.shape(1); // N

    auto stream = mlx::core::to_stream(s);
    auto op = std::make_shared<TernaryMatMul>(stream, group_size);
    
    return mlx::core::array(
        out_shape, 
        x.dtype(),
        op,
        {x_c, w_c}
    );
}

} // namespace mlx_ternary
