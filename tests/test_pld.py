"""
Unit tests for Zero-RAM Prompt Lookup Decoding (PLD).
"""

import pytest
from mlx_ternary.speculative.pld import (
    find_candidate_tokens,
    save_cache_checkpoint,
    restore_cache_checkpoint,
)


def test_find_candidate_tokens_matching():
    # Sequence with a repeated trigram [10, 20, 30]
    tokens = [1, 2, 10, 20, 30, 40, 50, 60, 7, 8, 10, 20, 30]
    # Query is [10, 20, 30] at the end
    draft = find_candidate_tokens(tokens, max_ngram_size=3, min_ngram_size=2, num_draft=3)
    assert draft == [40, 50, 60]


def test_find_candidate_tokens_fallback_bigram():
    # Only bigram [20, 30] matches, not trigram [99, 20, 30]
    tokens = [1, 2, 20, 30, 40, 50, 60, 7, 8, 99, 20, 30]
    draft = find_candidate_tokens(tokens, max_ngram_size=3, min_ngram_size=2, num_draft=2)
    assert draft == [40, 50]


def test_find_candidate_tokens_no_match():
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    draft = find_candidate_tokens(tokens, max_ngram_size=3, min_ngram_size=2, num_draft=4)
    assert draft == []


def test_cache_checkpoint_restore():
    class DummyKVCache:
        def __init__(self):
            self.offset = 42

    class DummyArraysCache:
        def __init__(self):
            self.cache = ["arr0_v1", "arr1_v1"]

        def __getitem__(self, idx):
            return self.cache[idx]

        def __setitem__(self, idx, val):
            self.cache[idx] = val

    kv = DummyKVCache()
    arr = DummyArraysCache()
    cache_list = [kv, arr]

    checkpoint = save_cache_checkpoint(cache_list)
    assert checkpoint[0] == 42
    assert checkpoint[1] == ("arr0_v1", "arr1_v1")

    # Mutate cache
    kv.offset = 100
    arr[0] = "arr0_v2"
    arr[1] = "arr1_v2"

    # Restore
    restore_cache_checkpoint(cache_list, checkpoint)
    assert kv.offset == 42
    assert arr[0] == "arr0_v1"
    assert arr[1] == "arr1_v1"


def test_find_candidate_tokens_anti_repetition():
    # Candidate immediately repeating trailing token: tokens[-1] == 30, following tokens in match: [30, 40]
    tokens = [1, 2, 10, 20, 30, 30, 40, 7, 8, 10, 20, 30]
    draft = find_candidate_tokens(tokens, max_ngram_size=3, min_ngram_size=2, num_draft=2)
    # Draft should reject [30, 40] because draft[0] == tokens[-1]
    assert draft == []

    # Candidate with cycle of length <= 3: [10, 20, 10, 20]
    tokens_cycle = [1, 2, 80, 90, 5, 6, 5, 6, 7, 8, 80, 90]
    draft_cycle = find_candidate_tokens(tokens_cycle, max_ngram_size=2, min_ngram_size=2, num_draft=4)
    assert draft_cycle == []
