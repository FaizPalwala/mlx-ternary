"""
mlx_ternary.speculative: High-performance speculative decoding engines for ternary models on Apple Silicon.
"""

from .pld import (
    find_candidate_tokens,
    save_cache_checkpoint,
    restore_cache_checkpoint,
    pld_generate,
    generate_pld_speculative,
)
from .draft_model import (
    VocabularyMapper,
    load_draft_model,
    draft_speculative_generate,
    generate_draft_speculative,
)
from .self_speculative import (
    LayerPrunedDraftModel,
    self_speculative_generate,
    generate_self_speculative,
)
from .router import (
    select_speculative_mode,
    generate_speculative,
    native_generate,
    get_device_memory_gb,
)

__all__ = [
    "find_candidate_tokens",
    "save_cache_checkpoint",
    "restore_cache_checkpoint",
    "pld_generate",
    "generate_pld_speculative",
    "VocabularyMapper",
    "load_draft_model",
    "draft_speculative_generate",
    "generate_draft_speculative",
    "LayerPrunedDraftModel",
    "self_speculative_generate",
    "generate_self_speculative",
    "select_speculative_mode",
    "generate_speculative",
    "native_generate",
    "get_device_memory_gb",
]
