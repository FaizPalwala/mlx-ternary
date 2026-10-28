"""
mlx_ternary.speculative: High-performance speculative decoding engines for ternary models on Apple Silicon.
"""

from .pld import (
    find_candidate_tokens,
    save_cache_checkpoint,
    restore_cache_checkpoint,
    pld_generate,
)

__all__ = [
    "find_candidate_tokens",
    "save_cache_checkpoint",
    "restore_cache_checkpoint",
    "pld_generate",
]
