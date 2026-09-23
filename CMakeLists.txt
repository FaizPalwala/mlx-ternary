cmake_minimum_required(VERSION 3.24)
project(mlx_ternary LANGUAGES CXX)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

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

# Disable unnecessary MLX targets (tests, examples, bindings) to speed up build
# and fix old CMake compatibility issues with doctest
set(BUILD_TESTS OFF CACHE BOOL "Disable MLX tests" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable MLX examples" FORCE)
set(BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Disable MLX python" FORCE)
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Disable MLX python" FORCE)

# This will pull down MLX and make its targets available (e.g., `mlx`)
FetchContent_MakeAvailable(mlx)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# 2. Convert Metal Shader to C++ Header for JIT
set(METAL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/mlx_ternary/cpp/ternary.metal")
set(METAL_HEADER "${CMAKE_CURRENT_BINARY_DIR}/ternary_metal.h")

add_custom_command(
    OUTPUT ${METAL_HEADER}
    COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/metal_to_h.py" ${METAL_SOURCE} ${METAL_HEADER}
    DEPENDS ${METAL_SOURCE} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/metal_to_h.py"
    COMMENT "Generating C++ header from Metal shader..."
)

# 3. Compile our Python extension
pybind11_add_module(mlx_ternary_ext 
    src/mlx_ternary/cpp/extension.cpp
    src/mlx_ternary/cpp/TernaryOp.cpp
    ${METAL_HEADER} # Depend on the generated header
)
target_include_directories(mlx_ternary_ext PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# 3. Link our extension against Apple's MLX C++ library
target_link_libraries(mlx_ternary_ext PRIVATE mlx)

target_compile_definitions(mlx_ternary_ext PRIVATE VERSION_INFO=${PROJECT_VERSION})

# 4. Install the extension into the python package
install(TARGETS mlx_ternary_ext DESTINATION mlx_ternary)
