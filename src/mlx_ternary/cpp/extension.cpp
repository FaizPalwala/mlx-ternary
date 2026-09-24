#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include "TernaryOp.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(mlx_ternary_ext, m) {
    m.doc() = "MLX Ternary native C++ extension";
    m.def("ternary_matmul", &mlx_ternary::ternary_matmul, "Ternary addition-only matrix multiplication",
          "x"_a, "w_packed"_a, "scales"_a);
}
