#pragma once

#include <mlx/mlx.h>
#include <mlx/primitives.h>

namespace mlx_ternary {

class TernaryMatMul : public mlx::core::Primitive {
public:
    uint32_t group_size_;
    
    explicit TernaryMatMul(mlx::core::Stream stream, uint32_t group_size)
        : mlx::core::Primitive(stream), group_size_(group_size) {}

    void eval_cpu(
        const std::vector<mlx::core::array>& inputs,
        std::vector<mlx::core::array>& outputs) override;

    void eval_gpu(
        const std::vector<mlx::core::array>& inputs,
        std::vector<mlx::core::array>& outputs) override;

    const char* name() const override { return "TernaryMatMul"; }
    bool is_equivalent(const mlx::core::Primitive& other) const override {
        const TernaryMatMul* o = dynamic_cast<const TernaryMatMul*>(&other);
        return o && (group_size_ == o->group_size_);
    }
};

mlx::core::array ternary_matmul(const mlx::core::array& x, const mlx::core::array& w_interleaved, uint32_t group_size);

} // namespace mlx_ternary
