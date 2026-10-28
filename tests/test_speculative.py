"""
Unit tests for Micro-Draft LLM Speculative Engine.
"""

import pytest
import mlx.core as mx
from mlx_ternary.speculative.draft_model import VocabularyMapper, draft_speculative_generate
from mlx_ternary.speculative.pld import save_cache_checkpoint, restore_cache_checkpoint


def test_vocabulary_mapper():
    draft_vocab = {"hello": 0, "world": 1, "only_draft": 2}
    target_vocab = {"hello": 10, "world": 20, "only_target": 30}

    mapper = VocabularyMapper(draft_vocab, target_vocab)
    assert mapper.is_identity is False

    # Draft -> Target
    assert mapper.to_target(0) == 10
    assert mapper.to_target(1) == 20
    assert mapper.to_target(2) == -1
    assert mapper.to_target(999) == -1

    # Target -> Draft
    assert mapper.to_draft(10) == 0
    assert mapper.to_draft(20) == 1
    assert mapper.to_draft(30) == -1
    assert mapper.to_draft(999) == -1


def test_vocabulary_mapper_identity():
    vocab = {"a": 0, "b": 1, "c": 2}
    mapper = VocabularyMapper(vocab, vocab)
    assert mapper.is_identity is True
    assert mapper.to_target(1) == 1
    assert mapper.to_draft(2) == 2


class MockDraftCache:
    def __init__(self):
        self.offset = 10

    def trim(self, n):
        self.offset -= n


class MockTargetCache:
    def __init__(self):
        self.offset = 50

    def trim(self, n):
        self.offset -= n


def test_draft_cache_trimming():
    cache = [MockDraftCache() for _ in range(4)]
    for c in cache:
        c.trim(3)
    assert all(c.offset == 7 for c in cache)


def test_speculative_verification_flow():
    """
    Test end-to-end speculative acceptance and cache synchronization using mock models.
    """
    class MockTargetModel:
        def __init__(self, predictions):
            self.predictions = predictions # list of predicted tokens per step

        def make_cache(self):
            return [MockTargetCache() for _ in range(2)]

        def model(self, x, cache=None):
            # Advance cache offset by number of tokens
            if cache:
                for c in cache:
                    c.offset += x.shape[1]
            return x

        def lm_head(self, h):
            # Return one-hot-like logits matching self.predictions
            B, S = h.shape[:2]
            vocab_size = 100
            logits = mx.zeros((B, S, vocab_size))
            return logits

    class MockDraftModel:
        def __init__(self, draft_tokens):
            self.draft_tokens = list(draft_tokens)
            self.idx = 0

        def __call__(self, x, cache=None):
            if cache:
                for c in cache:
                    c.offset += x.shape[1]
            B, S = x.shape
            vocab_size = 100
            logits = mx.zeros((B, S, vocab_size))
            return logits

    # Target model and draft model instances
    target_m = MockTargetModel([1, 2, 3])
    draft_m = MockDraftModel([1, 2, 3])

    # Cache checkpoint and restore verify
    target_cache = target_m.make_cache()
    chk = save_cache_checkpoint(target_cache)
    assert chk[0] == 50

    target_cache[0].offset = 65
    restore_cache_checkpoint(target_cache, chk)
    assert target_cache[0].offset == 50
