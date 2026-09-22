# Building from Source

## Prerequisites

- **macOS** (Apple Silicon M1/M2/M3/M4 required for Metal acceleration)
- **CMake** `>= 3.24`
- **Python** `>= 3.9`

## Build Instructions

`mlx_ternary` relies on `scikit-build-core` and fetches Apple's `mlx` library directly from source to ensure stable C++ linkage.

```bash
# Clone the repository
git clone https://github.com/your-repo/mlx-ternary.git
cd mlx-ternary

# Install via pip (this will trigger CMake and compile the C++ extension)
pip install .
```

### Metal Shader Compilation

We utilize **Just-In-Time (JIT)** compilation for our Metal shaders. 
At build time, the CMake pipeline converts `ternary.metal` into a C++ string header. At runtime, MLX injects dimensional constants into the shader string and compiles it on the fly. This enables the Metal driver to aggressively unroll loops based on exact inference matrix shapes.
