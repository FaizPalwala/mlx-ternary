"""
Unit tests for Self-Speculative Layer-Pruned Draft Engine.
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
from mlx_ternary.speculative.self_speculative import (
    LayerPrunedDraftModel,
    EarlyExitAdapter,
    self_speculative_generate,
)
from mlx_ternary.speculative.pld import save_cache_checkpoint, restore_cache_checkpoint


class MockLayer(nn.Module):
    def __init__(self, is_linear=False):
        super().__init__()
        self.is_linear = is_linear

    def __call__(self, x, mask=None, cache=None):
        if cache is not None:
            cache.offset += x.shape[1]
        return x + 1.0


class MockInnerModel(nn.Module):
    def __init__(self, num_layers=8):
        super().__init__()
        self.fa_idx = 1
        self.ssm_idx = 0
        self.layers = [MockLayer(is_linear=(i % 2 == 0)) for i in range(num_layers)]

    def embed_tokens(self, x):
        B, S = x.shape
        # return dummy hidden representation
        return mx.ones((B, S, 16))

    def norm(self, h):
        return h


class MockCache:
    def __init__(self):
        self.offset = 0

    def trim(self, n):
        self.offset -= n


class MockModel(nn.Module):
    def __init__(self, num_layers=8, vocab_size=50):
        super().__init__()
        self.model = MockInnerModel(num_layers=num_layers)
        self.vocab_size = vocab_size

    def make_cache(self):
        return [MockCache() for _ in range(len(self.model.layers))]

    def lm_head(self, h):
        B, S = h.shape[:2]
        return mx.zeros((B, S, self.vocab_size))


def test_layer_pruned_draft_model_init():
    full_model = MockModel(num_layers=8)
    draft_model = LayerPrunedDraftModel(full_model, num_layers=4)

    assert len(draft_model.layers) == 4
    assert len(draft_model.make_cache()) == 4
    assert draft_model.fa_idx == 1
    assert draft_model.ssm_idx == 0


def test_layer_pruned_draft_model_default_dynamic():
    # Model-agnostic dynamic layer depth: max(4, total_layers // 4)
    # 8-layer model -> max(4, 8 // 4) = 4
    m8 = MockModel(num_layers=8)
    d8 = LayerPrunedDraftModel(m8)
    assert d8.num_layers == 4
    assert len(d8.layers) == 4

    # 64-layer model (e.g. 27B) -> max(4, 64 // 4) = 16
    m64 = MockModel(num_layers=64)
    d64 = LayerPrunedDraftModel(m64)
    assert d64.num_layers == 16
    assert len(d64.layers) == 16

    # 32-layer model (e.g. 7B) -> max(4, 32 // 4) = 8
    m32 = MockModel(num_layers=32)
    d32 = LayerPrunedDraftModel(m32)
    assert d32.num_layers == 8
    assert len(d32.layers) == 8

    # 12-layer model -> max(4, 12 // 4) = 4
    m12 = MockModel(num_layers=12)
    d12 = LayerPrunedDraftModel(m12)
    assert d12.num_layers == 4
    assert len(d12.layers) == 4


def test_layer_pruned_draft_forward():
    full_model = MockModel(num_layers=8)
    draft_model = LayerPrunedDraftModel(full_model, num_layers=4)
    cache = draft_model.make_cache()

    x = mx.array([[1, 2, 3]], dtype=mx.int32)
    logits = draft_model(x, cache=cache)

    assert logits.shape == (1, 3, 50)
    # Each of the 4 layers should have advanced cache by 3 tokens
    assert all(c.offset == 3 for c in cache)


def test_self_speculative_cache_rollback():
    full_model = MockModel(num_layers=8)
    draft_model = LayerPrunedDraftModel(full_model, num_layers=4)

    target_cache = full_model.make_cache()
    draft_cache = draft_model.make_cache()

    # Step 1: Checkpoint
    t_chk = save_cache_checkpoint(target_cache)
    d_chk = save_cache_checkpoint(draft_cache)

    # Step 2: Mutate / Advance
    for c in target_cache:
        c.offset = 25
    for c in draft_cache:
        c.offset = 15

    # Step 3: Restore
    restore_cache_checkpoint(target_cache, t_chk)
    restore_cache_checkpoint(draft_cache, d_chk)

    assert all(c.offset == 0 for c in target_cache)
    assert all(c.offset == 0 for c in draft_cache)


def test_early_exit_adapter():
    hidden_dim = 16
    rank = 4
    adapter = EarlyExitAdapter(hidden_dim=hidden_dim, rank=rank)

    x = mx.random.normal((2, 5, hidden_dim))
    out = adapter(x)
    assert out.shape == (2, 5, hidden_dim)

    # Test integration into LayerPrunedDraftModel
    full_model = MockModel(num_layers=8)
    draft_model = LayerPrunedDraftModel(full_model, num_layers=4, exit_head=adapter)
    logits = draft_model(mx.array([[1, 2, 3]], dtype=mx.int32))
    assert logits.shape == (1, 3, 50)
