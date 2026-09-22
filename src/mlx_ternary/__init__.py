from .mlx_ternary_ext import add_stub

def load(model_path: str):
    """
    Load a ternary model from safetensors.
    """
    raise NotImplementedError("Ternary model loading not yet implemented.")

def generate(model, tokenizer, prompt: str):
    """
    Generate tokens from a loaded ternary model.
    """
    raise NotImplementedError("Ternary text generation not yet implemented.")

__all__ = ["load", "generate", "add_stub"]
