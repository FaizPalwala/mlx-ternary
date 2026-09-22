import mlx.core as mx
from typing import Tuple, Dict

def load_ternary_model(model_path: str) -> Tuple[Dict[str, mx.array], Dict[str, mx.array]]:
    """
    Loads a ternary safetensors model.
    
    In a standard 1.58b model export, the weights are often stored in two parts:
    - `*.weight`: The packed uint8 ternary values.
    - `*.weight_scale`: The fp16 scaling factor.

    This function loads them via MLX's native safetensors parser.
    
    Args:
        model_path: Path to the safetensors file or directory.
        
    Returns:
        weights: A dictionary of the packed uint8 weight tensors.
        scales: A dictionary of the fp16 scaling tensors.
    """
    
    # MLX natively supports reading .safetensors files
    raw_state_dict = mx.load(model_path)
    
    weights = {}
    scales = {}
    
    for key, tensor in raw_state_dict.items():
        if key.endswith(".weight_scale"):
            # Strip suffix to match the base layer name
            base_name = key.replace(".weight_scale", "")
            scales[base_name] = tensor
        elif key.endswith(".weight"):
            base_name = key.replace(".weight", "")
            weights[base_name] = tensor
        else:
            # Handle other params like biases or layernorms
            weights[key] = tensor
            
    return weights, scales
