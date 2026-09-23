cmake_minimum_required(VERSION 3.24)
project(mlx_ternary LANGUAGES CXX)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Find MLX from the pip installation
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

# Fetch nanobind instead of pybind11
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG v3.0.1
  GIT_SHALLOW TRUE
  EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(nanobind)

# 2. Convert Metal Shader to C++ Header for JIT
set(METAL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/mlx_ternary/cpp/ternary.metal")
set(METAL_HEADER "${CMAKE_CURRENT_BINARY_DIR}/ternary_metal.h")

add_custom_command(
    OUTPUT ${METAL_HEADER}
    COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/metal_to_h.py" ${METAL_SOURCE} ${METAL_HEADER}
    DEPENDS ${METAL_SOURCE} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/metal_to_h.py"
    COMMENT "Generating C++ header from Metal shader..."
)

# 3. Compile our Python extension using nanobind
nanobind_add_module(mlx_ternary_ext 
    NB_STATIC
    STABLE_ABI
    NB_DOMAIN mlx
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
