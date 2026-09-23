#pragma once

#include <mlx/mlx.h>
#include <mlx/primitives.h>

namespace mlx_ternary {

class TernaryMatMul : public mlx::core::Primitive {
public:
    explicit TernaryMatMul(mlx::core::Stream stream)
        : mlx::core::Primitive(stream) {}

    void eval_cpu(
        const std::vector<mlx::core::array>& inputs,
        std::vector<mlx::core::array>& outputs) override;

    void eval_gpu(
        const std::vector<mlx::core::array>& inputs,
        std::vector<mlx::core::array>& outputs) override;

    void print(std::ostream& os) override { os << "TernaryMatMul"; }
    bool is_equivalent(const mlx::core::Primitive& other) const override {
        return typeid(*this) == typeid(other);
    }
};

// Expose a factory function to easily create the op in Python bindings
mlx::core::array ternary_matmul(const mlx::core::array& x, const mlx::core::array& w_packed, const mlx::core::array& scales, mlx::core::StreamOrDevice s = {});

} // namespace mlx_ternary
