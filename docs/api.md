# API Reference

The `mlx_ternary` Python package provides a drop-in replacement for standard `mlx-lm` operations, specifically optimized for 1.58b ternary models.

## `mlx_ternary.load(model_path: str)`

Loads a safetensors model using the custom ternary MLX core backend.

**Arguments:**
- `model_path` (str): The local path or HuggingFace repo ID containing the model weights.

**Returns:**
- `model`: The initialized MLX model.
- `tokenizer`: The associated tokenizer.

## `mlx_ternary.ternary_matmul(x: mlx.core.array, w: mlx.core.array)`

Performs a forward pass using the custom Metal addition-only kernels.

**Arguments:**
- `x`: Activation tensor (float16)
- `w`: Packed ternary weights (uint8)

**Returns:**
- `out`: Output activation tensor (float16)
