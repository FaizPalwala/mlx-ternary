#include <pybind11/pybind11.h>

namespace py = pybind11;

int add_stub(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(mlx_ternary_ext, m) {
    m.doc() = "MLX Ternary native C++ extension";
    m.def("add_stub", &add_stub, "A stub function to verify bindings");
}
