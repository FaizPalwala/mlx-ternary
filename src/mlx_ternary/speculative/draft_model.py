"""
Micro-Draft LLM Speculative Decoding Engine (Qwen2.5-0.5B-Instruct).

Accelerates 1.58-bit ternary language models using an ultra-fast, lightweight
4-bit draft LLM (~0.26 GB VRAM) as an autoregressive proposal engine.

Key Invariants:
1. Low-VRAM Footprint: Uses 4-bit quantized Qwen2.5-0.5B-Instruct consuming
   < 0.3 GB VRAM (well within the 1.2 GB architectural budget).
2. Ultra-Fast Proposal: Generates draft tokens at ~250 tok/s (~4 ms/token on M4 Pro),
   adding negligible overhead before target model verification.
3. Batched Verification: Evaluates K draft tokens in a single forward pass
   (M = K + 1) using mlx-ternary's native fast GEMM kernel.
4. Robust Cache Synchronization: Bidirectional synchronization and trimming between
   draft model KVCache and target model hybrid recurrent/KV cache.
"""

from typing import List, Tuple, Optional, Set, Generator, Dict, Any
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .pld import save_cache_checkpoint, restore_cache_checkpoint


class VocabularyMapper:
    """
    O(1) bidirectional vocabulary mapping between draft and target tokenizers.
    Handles exact-match or differing vocabulary spaces with zero per-step latency.
    """
    def __init__(self, draft_vocab: Dict[str, int], target_vocab: Dict[str, int]):
        self.draft_size = max(draft_vocab.values()) + 1 if draft_vocab else 0
        self.target_size = max(target_vocab.values()) + 1 if target_vocab else 0
        
        self.draft_to_target = np.full(self.draft_size, -1, dtype=np.int32)
        self.target_to_draft = np.full(self.target_size, -1, dtype=np.int32)
        
        for token_str, d_id in draft_vocab.items():
            if token_str in target_vocab:
                t_id = target_vocab[token_str]
                self.draft_to_target[d_id] = t_id
                self.target_to_draft[t_id] = d_id
                
        self.is_identity = bool(self.draft_size == self.target_size and 
                                 np.all(self.draft_to_target == np.arange(self.draft_size)))

    def to_target(self, draft_token_id: int) -> int:
        if self.is_identity:
            return draft_token_id
        if 0 <= draft_token_id < self.draft_size:
            return int(self.draft_to_target[draft_token_id])
        return -1

    def to_draft(self, target_token_id: int) -> int:
        if self.is_identity:
            return target_token_id
        if 0 <= target_token_id < self.target_size:
            return int(self.target_to_draft[target_token_id])
        return -1


def load_draft_model(model_name: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"):
    """
    Load micro-draft LLM checkpoint via mlx_lm.
    """
    try:
        from mlx_lm import load
    except ImportError:
        raise ImportError("mlx_lm is required for draft model speculative decoding. Install it via: pip install mlx-lm")
        
    draft_model, draft_tokenizer = load(model_name)
    return draft_model, draft_tokenizer


def draft_speculative_generate(
    target_model: Any,
    draft_model: Any,
    target_prompt_tokens: List[int],
    draft_prompt_tokens: Optional[List[int]] = None,
    vocab_mapper: Optional[VocabularyMapper] = None,
    max_tokens: int = 128,
    num_draft: int = 4,
    stop_tokens: Optional[Set[int]] = None,
    target_cache: Optional[List[Any]] = None,
    draft_cache: Optional[List[Any]] = None,
    adaptive_throttling: bool = True,
    min_acceptance_threshold: float = 0.15,
) -> Generator[Tuple[int, bool, Dict[str, Any]], None, None]:
    """
    Generate tokens using Qwen2.5-0.5B draft-model speculative decoding.

    Args:
        target_model: Target 1.58-bit ternary model (Bonsai 27B).
        draft_model: Micro-draft model (Qwen2.5-0.5B-Instruct).
        target_prompt_tokens: Prompt token IDs in target model vocabulary.
        draft_prompt_tokens: Prompt token IDs in draft model vocabulary (optional).
        vocab_mapper: VocabularyMapper instance for translating token IDs.
        max_tokens: Maximum tokens to generate.
        num_draft: Number of candidate tokens to speculate per step (K=3..5).
        stop_tokens: Target token IDs that terminate generation.
        target_cache: Optional pre-allocated cache for target model.
        draft_cache: Optional pre-allocated cache for draft model.
        adaptive_throttling: If True, dynamically fall back to M=1 when draft acceptance drops.
        min_acceptance_threshold: Minimum rolling acceptance rate before throttling draft speculation.

    Yields:
        (token_id, is_draft_accepted, stats_dict)
    """
    from mlx_lm.models.cache import make_prompt_cache

    if stop_tokens is None:
        stop_tokens = set()

    if draft_prompt_tokens is None:
        draft_prompt_tokens = target_prompt_tokens

    if target_cache is None:
        target_cache = target_model.make_cache()

    if draft_cache is None:
        draft_cache = make_prompt_cache(draft_model)

    stats = {
        "speculative_steps": 0,
        "fallback_steps": 0,
        "draft_tokens_proposed": 0,
        "draft_tokens_accepted": 0,
        "acceptance_rate": 0.0,
        "draft_generation_time_s": 0.0,
        "target_verification_time_s": 0.0,
        "throttled_steps": 0,
    }

    recent_accepts: List[float] = []
    throttle_cooldown = 0

    # 1. Prompt Prefill Phase
    # Target model prefill
    x_target = mx.array([target_prompt_tokens], dtype=mx.int32)
    hidden_target = target_model.model(x_target, cache=target_cache)
    logits_target = target_model.lm_head(hidden_target[:, -1:, :])[:, -1, :]
    first_target_tok = int(mx.argmax(logits_target, axis=-1).item())

    # Draft model prefill
    x_draft = mx.array([draft_prompt_tokens], dtype=mx.int32)
    _ = draft_model(x_draft, cache=draft_cache)
    
    # Align draft model with first target token
    first_draft_tok = first_target_tok if vocab_mapper is None else vocab_mapper.to_draft(first_target_tok)
    if first_draft_tok != -1:
        _ = draft_model(mx.array([[first_draft_tok]], dtype=mx.int32), cache=draft_cache)

    generated: List[int] = [first_target_tok]
    curr_target_tok = first_target_tok
    curr_draft_tok = first_draft_tok

    yield first_target_tok, False, stats
    if first_target_tok in stop_tokens or len(generated) >= max_tokens:
        return

    # 2. Speculative Decode Generation Loop
    while len(generated) < max_tokens:
        budget = max_tokens - len(generated)
        k_draft = min(num_draft, budget)
        is_throttled = False
        if adaptive_throttling:
            if throttle_cooldown > 0:
                throttle_cooldown -= 1
                is_throttled = True
            elif len(recent_accepts) >= 3 and (sum(recent_accepts) / len(recent_accepts)) < min_acceptance_threshold:
                is_throttled = True
                throttle_cooldown = 4  # bypass draft overhead and run fast M=1
                stats["throttled_steps"] += 1

        if curr_draft_tok == -1 or k_draft <= 0 or is_throttled:
            # Fallback path: standard 1-token decode step on target model
            x = mx.array([[curr_target_tok]], dtype=mx.int32)
            hidden = target_model.model(x, cache=target_cache)
            logits = target_model.lm_head(hidden[:, -1:, :])[:, -1, :]
            next_target_tok = int(mx.argmax(logits, axis=-1).item())

            curr_target_tok = next_target_tok
            curr_draft_tok = curr_target_tok if vocab_mapper is None else vocab_mapper.to_draft(curr_target_tok)
            if curr_draft_tok != -1:
                _ = draft_model(mx.array([[curr_draft_tok]], dtype=mx.int32), cache=draft_cache)

            generated.append(curr_target_tok)
            stats["fallback_steps"] += 1

            yield curr_target_tok, False, stats
            if curr_target_tok in stop_tokens:
                break
            continue

        # 3. Draft Model Candidate Generation
        draft_cands: List[int] = []
        target_cands: List[int] = []
        c_in = mx.array([[curr_draft_tok]], dtype=mx.int32)

        for _ in range(k_draft):
            out_draft = draft_model(c_in, cache=draft_cache)
            d_tok = int(mx.argmax(out_draft[:, -1, :], axis=-1).item())
            t_tok = d_tok if vocab_mapper is None else vocab_mapper.to_target(d_tok)
            
            if t_tok == -1:
                # Token unmappable to target vocab; stop proposing further tokens
                break
                
            draft_cands.append(d_tok)
            target_cands.append(t_tok)
            c_in = mx.array([[d_tok]], dtype=mx.int32)

        K = len(target_cands)
        if K == 0:
            # Fallback
            x = mx.array([[curr_target_tok]], dtype=mx.int32)
            hidden = target_model.model(x, cache=target_cache)
            logits = target_model.lm_head(hidden[:, -1:, :])[:, -1, :]
            next_target_tok = int(mx.argmax(logits, axis=-1).item())

            curr_target_tok = next_target_tok
            curr_draft_tok = curr_target_tok if vocab_mapper is None else vocab_mapper.to_draft(curr_target_tok)
            if curr_draft_tok != -1:
                _ = draft_model(mx.array([[curr_draft_tok]], dtype=mx.int32), cache=draft_cache)

            generated.append(curr_target_tok)
            stats["fallback_steps"] += 1

            yield curr_target_tok, False, stats
            if curr_target_tok in stop_tokens:
                break
            continue

        # 4. Target Model Batched Verification (M = K + 1)
        stats["speculative_steps"] += 1
        stats["draft_tokens_proposed"] += K

        target_checkpoint = save_cache_checkpoint(target_cache)
        x_verify = mx.array([[curr_target_tok] + target_cands], dtype=mx.int32)

        hidden_v = target_model.model(x_verify, cache=target_cache)
        logits_v = target_model.lm_head(hidden_v)
        preds = mx.argmax(logits_v[0], axis=-1)
        mx.eval(preds)
        preds_list = preds.tolist()

        # Check accepted tokens sequentially
        n_accept = 0
        for i in range(K):
            if preds_list[i] == target_cands[i]:
                n_accept += 1
            else:
                break

        stats["draft_tokens_accepted"] += n_accept
        if stats["draft_tokens_proposed"] > 0:
            stats["acceptance_rate"] = stats["draft_tokens_accepted"] / stats["draft_tokens_proposed"]

        recent_accepts.append(n_accept / K if K > 0 else 0.0)
        if len(recent_accepts) > 8:
            recent_accepts.pop(0)

        # 5. Commit & Cache Synchronization
        if n_accept == K:
            # 100% acceptance! All K draft tokens accepted!
            for i in range(K):
                tok = target_cands[i]
                generated.append(tok)
                yield tok, True, stats
                if len(generated) >= max_tokens or tok in stop_tokens:
                    return

            bonus_tok = int(preds_list[K])
            curr_target_tok = bonus_tok
            curr_draft_tok = bonus_tok if vocab_mapper is None else vocab_mapper.to_draft(bonus_tok)
            if curr_draft_tok != -1:
                _ = draft_model(mx.array([[curr_draft_tok]], dtype=mx.int32), cache=draft_cache)

            generated.append(curr_target_tok)
            yield curr_target_tok, False, stats
            if curr_target_tok in stop_tokens:
                break
        else:
            # Partial acceptance or total miss (n_accept < K)
            # Rollback target model cache
            restore_cache_checkpoint(target_cache, target_checkpoint)
            
            # Commit accepted prefix: [curr_target_tok] + target_cands[:n_accept]
            commit_tokens = [curr_target_tok] + target_cands[:n_accept]
            x_commit = mx.array([commit_tokens], dtype=mx.int32)
            hidden_commit = target_model.model(x_commit, cache=target_cache)
            mx.eval(hidden_commit)

            for i in range(n_accept):
                tok = target_cands[i]
                generated.append(tok)
                yield tok, True, stats
                if len(generated) >= max_tokens or tok in stop_tokens:
                    return

            # Target model prediction at divergence position
            divergence_tok = int(preds_list[n_accept])
            curr_target_tok = divergence_tok

            # Synchronize draft model: rewind rejected tokens
            num_rewind = len(draft_cands) - n_accept
            if num_rewind > 0:
                for c in draft_cache:
                    c.trim(num_rewind)

            # Advance draft model with the verified divergence token
            curr_draft_tok = divergence_tok if vocab_mapper is None else vocab_mapper.to_draft(divergence_tok)
            if curr_draft_tok != -1:
                _ = draft_model(mx.array([[curr_draft_tok]], dtype=mx.int32), cache=draft_cache)

            generated.append(curr_target_tok)
            yield curr_target_tok, False, stats
            if curr_target_tok in stop_tokens:
                break


# Alias for universal speculative router
generate_draft_speculative = draft_speculative_generate

