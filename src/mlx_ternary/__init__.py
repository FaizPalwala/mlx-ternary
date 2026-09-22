from .mlx_ternary_ext import ternary_matmul
from .loader import load_ternary_model

def load(model_path: str):
    """
    Load a ternary model from safetensors.
    """
    weights, scales = load_ternary_model(model_path)
    # Placeholder for instantiating the actual NN architecture
    # model = TernaryLlama(weights, scales)
    return weights, scales

def generate(model, tokenizer, prompt: str):
    """
    Generate tokens from a loaded ternary model.
    """
    raise NotImplementedError("Ternary text generation not yet implemented.")

__all__ = ["load", "generate", "ternary_matmul"]
