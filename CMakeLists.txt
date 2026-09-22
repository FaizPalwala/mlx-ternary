cmake_minimum_required(VERSION 3.24)
project(mlx_ternary LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# 1. Fetch MLX from source
# We lock to a specific tag to prevent upstream breaks.
FetchContent_Declare(
  mlx
  GIT_REPOSITORY https://github.com/ml-explore/mlx.git
  GIT_TAG        v0.16.0
)
# This will pull down MLX and make its targets available (e.g., `mlx`)
FetchContent_MakeAvailable(mlx)

find_package(pybind11 CONFIG REQUIRED)

# 2. Compile our Python extension
pybind11_add_module(mlx_ternary_ext 
    src/mlx_ternary/cpp/extension.cpp
    src/mlx_ternary/cpp/TernaryOp.cpp
)

# 3. Link our extension against Apple's MLX C++ library
target_link_libraries(mlx_ternary_ext PRIVATE mlx)

target_compile_definitions(mlx_ternary_ext PRIVATE VERSION_INFO=${PROJECT_VERSION})
