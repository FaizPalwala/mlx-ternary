#include <pybind11/pybind11.h>
#include "TernaryOp.h"

namespace py = pybind11;

PYBIND11_MODULE(mlx_ternary_ext, m) {
    m.doc() = "MLX Ternary native C++ extension";
    m.def("ternary_matmul", &mlx_ternary::ternary_matmul, "Ternary addition-only matrix multiplication",
          py::arg("x"), py::arg("w_packed"), py::arg("scales"), py::arg("stream") = mlx::core::StreamOrDevice{});
}
