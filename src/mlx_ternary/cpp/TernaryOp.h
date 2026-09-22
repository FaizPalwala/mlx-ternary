#pragma once

#include <mlx/mlx.h>

namespace mlx_ternary {

class TernaryMatMul : public mlx::core::CustomOp {
public:
    // Constructor. For now, we take no extra parameters, but we might need
    // group_size, bits_per_weight, etc. later.
    TernaryMatMul();

    // Standard MLX CustomOp interface overrides
    std::unique_ptr<mlx::core::CustomOp> clone() const override;
    
    std::vector<mlx::core::array> eval_cpu(
        const std::vector<mlx::core::array>& inputs,
        const std::vector<mlx::core::array>& outputs) override;

    std::vector<mlx::core::array> eval_gpu(
        const std::vector<mlx::core::array>& inputs,
        const std::vector<mlx::core::array>& outputs) override;

    std::string name() const override { return "TernaryMatMul"; }
};

// Expose a factory function to easily create the op in Python bindings
mlx::core::array ternary_matmul(const mlx::core::array& x, const mlx::core::array& w);

} // namespace mlx_ternary
