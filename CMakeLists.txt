cmake_minimum_required(VERSION 3.24)
project(mlx_ternary LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

# Compile the extension
pybind11_add_module(mlx_ternary_ext MODULE src/mlx_ternary/cpp/extension.cpp)

# We will need to link against MLX and compile Metal shaders.
# For now, this is a basic stub.
target_compile_definitions(mlx_ternary_ext PRIVATE VERSION_INFO=${PROJECT_VERSION})
