"""
Zero-RAM Prompt Lookup Decoding (PLD) Engine for mlx-ternary.

Prompt Lookup Decoding accelerates autoregressive inference by exploiting the
observation that many generation tasks (source code, JSON/structured data, and
document QA/RAG) exhibit strong n-gram overlap with the prompt or recently generated text.

Key Invariants:
1. Zero Additional VRAM: Candidate generation is performed via zero-allocation
   n-gram lookup over token history in host memory.
2. Bit-Identical Rollback: Non-accepted draft tokens are cleanly rewound in the
   hybrid KV/recurrent cache with zero state corruption.
3. Zero-Penalty Fallback: When no n-gram matches are found, execution falls back
   directly to standard autoregressive decode with zero latency overhead.
"""

from typing import List, Tuple, Optional, Set, Generator, Dict, Any
import mlx.core as mx


def find_candidate_tokens(
    tokens: List[int],
    max_ngram_size: int = 3,
    min_ngram_size: int = 2,
    num_draft: int = 4,
) -> List[int]:
    """
    Search past token context for matching n-grams and propose candidate continuations.

    Args:
        tokens: Full token sequence (prompt + generated tokens so far).
        max_ngram_size: Maximum n-gram window to match (default 3).
        min_ngram_size: Minimum n-gram window to match (default 2).
        num_draft: Number of continuation tokens to propose (K=3..5).

    Returns:
        List of draft token IDs, or empty list if no n-gram match found.
    """
    seq_len = len(tokens)
    if seq_len < min_ngram_size + 1 or num_draft <= 0:
        return []

    # Search for n-grams from largest to smallest
    for ngram_size in range(max_ngram_size, min_ngram_size - 1, -1):
        if seq_len < ngram_size + 1:
            continue

        query = tokens[-ngram_size:]
        # Scan backwards through history (excluding the current tail) for the most recent match
        for i in range(seq_len - ngram_size - 1, -1, -1):
            if tokens[i : i + ngram_size] == query:
                # Found matching n-gram; return following tokens as draft candidates
                draft = tokens[i + ngram_size : i + ngram_size + num_draft]
                if not draft:
                    continue

                # Anti-repetition check:
                # 1. Reject if candidate immediately repeats the last token
                if draft[0] == tokens[-1]:
                    continue

                # 2. Reject if candidate contains a cycle of length <= 3
                has_cycle = False
                for cycle_len in range(1, min(4, len(draft))):
                    if len(draft) >= 2 * cycle_len and draft[:cycle_len] == draft[cycle_len : 2 * cycle_len]:
                        has_cycle = True
                        break
                if has_cycle:
                    continue

                return draft

    return []


def save_cache_checkpoint(cache: List[Any]) -> List[Any]:
    """
    Capture a zero-copy snapshot of the hybrid KV / recurrent cache state.
    - KVCache: stores integer offset.
    - ArraysCache (recurrent GatedDeltaNet): stores references to current conv and recurrent state arrays.
    """
    checkpoint = []
    for c in cache:
        if hasattr(c, "offset"):
            checkpoint.append(c.offset)
        elif hasattr(c, "cache"):
            checkpoint.append((c[0], c[1]))
        else:
            checkpoint.append(None)
    return checkpoint


def restore_cache_checkpoint(cache: List[Any], checkpoint: List[Any]) -> None:
    """
    Restore the hybrid cache state to a previous snapshot with bit-identical precision.
    Handles KVCache (.offset) and GatedDeltaNet ArraysCache (c[0], c[1]).
    """
    for c, saved in zip(cache, checkpoint):
        if saved is None:
            continue
        if hasattr(c, "offset"):
            c.offset = saved
        elif hasattr(c, "cache"):
            c[0] = saved[0]
            c[1] = saved[1]


def pld_generate(
    model: Any,
    prompt_tokens: List[int],
    max_tokens: int = 128,
    num_draft: int = 4,
    max_ngram_size: int = 3,
    min_ngram_size: int = 2,
    stop_tokens: Optional[Set[int]] = None,
    cache: Optional[List[Any]] = None,
) -> Generator[Tuple[int, bool, Dict[str, Any]], None, None]:
    """
    Autoregressively generate tokens using Prompt Lookup Decoding (PLD).

    Args:
        model: Loaded ternary language model.
        prompt_tokens: Tokenized input prompt.
        max_tokens: Maximum number of tokens to generate.
        num_draft: Number of candidate draft tokens to speculate per step (K=3..5).
        max_ngram_size: Maximum n-gram window to match.
        min_ngram_size: Minimum n-gram window to match.
        stop_tokens: Set of token IDs that terminate generation.
        cache: Optional pre-allocated model KV cache.

    Yields:
        (token_id, is_draft_accepted, stats_dict)
    """
    if stop_tokens is None:
        stop_tokens = set()

    if cache is None:
        cache = model.make_cache()

    stats = {
        "speculative_steps": 0,
        "fallback_steps": 0,
        "draft_tokens_proposed": 0,
        "draft_tokens_accepted": 0,
        "acceptance_rate": 0.0,
    }

    # 1. Prompt Prefill Phase
    x_prompt = mx.array([prompt_tokens], dtype=mx.int32)
    hidden = model.model(x_prompt, cache=cache)
    logits = model.lm_head(hidden[:, -1:, :])[:, -1, :]
    first_token = int(mx.argmax(logits, axis=-1).item())

    generated: List[int] = [first_token]
    all_tokens: List[int] = list(prompt_tokens) + [first_token]
    current_token = first_token

    yield first_token, False, stats
    if first_token in stop_tokens or len(generated) >= max_tokens:
        return

    # 2. Speculative Decode Generation Loop
    throttle_draft = False
    while len(generated) < max_tokens:
        budget = max_tokens - len(generated)
        k_draft = min(num_draft, budget)

        # Anti-repetition / anti-loop guard on consecutive identical tokens
        if len(all_tokens) >= 3 and all_tokens[-1] == all_tokens[-2] == all_tokens[-3]:
            throttle_draft = True

        if throttle_draft:
            draft_tokens = []
            throttle_draft = False
        else:
            # Look up candidate draft tokens from prior context
            draft_tokens = find_candidate_tokens(
                all_tokens,
                max_ngram_size=max_ngram_size,
                min_ngram_size=min_ngram_size,
                num_draft=k_draft,
            )

        if not draft_tokens:
            # Fallback path: Standard single-token autoregressive step
            x = mx.array([[current_token]], dtype=mx.int32)
            hidden = model.model(x, cache=cache)
            logits = model.lm_head(hidden[:, -1:, :])[:, -1, :]
            next_token = int(mx.argmax(logits, axis=-1).item())

            current_token = next_token
            generated.append(current_token)
            all_tokens.append(current_token)
            stats["fallback_steps"] += 1

            yield current_token, False, stats
            if current_token in stop_tokens:
                break
        else:
            # Speculative verification path
            K = len(draft_tokens)
            stats["speculative_steps"] += 1
            stats["draft_tokens_proposed"] += K

            checkpoint = save_cache_checkpoint(cache)
            candidates = [current_token] + draft_tokens
            x_cand = mx.array([candidates], dtype=mx.int32)

            hidden = model.model(x_cand, cache=cache)
            logits = model.lm_head(hidden)
            preds = mx.argmax(logits[0], axis=-1)
            mx.eval(preds)
            preds_list = preds.tolist()

            # Verify draft candidates against target model predictions
            n_accept = 0
            for i in range(K):
                if preds_list[i] == draft_tokens[i]:
                    n_accept += 1
                else:
                    break

            stats["draft_tokens_accepted"] += n_accept
            if stats["draft_tokens_proposed"] > 0:
                stats["acceptance_rate"] = stats["draft_tokens_accepted"] / stats["draft_tokens_proposed"]

            if n_accept == K:
                # Full draft match: all K draft tokens accepted!
                # Cache already contains [current_token, d_0, ..., d_{K-1}]
                for i in range(K):
                    tok = draft_tokens[i]
                    generated.append(tok)
                    all_tokens.append(tok)
                    yield tok, True, stats
                    if len(generated) >= max_tokens or tok in stop_tokens:
                        return

                # Bonus token generated from final position
                bonus_token = preds_list[K]
                current_token = bonus_token
                generated.append(current_token)
                all_tokens.append(current_token)
                yield current_token, False, stats
                if current_token in stop_tokens:
                    break
            else:
                # Partial match or total miss (n_accept < K)
                # 1. Restore hybrid cache to checkpoint prior to draft speculative step
                restore_cache_checkpoint(cache, checkpoint)

                # 2. Advance cache with accepted prefix: [current_token] + draft_tokens[:n_accept]
                commit_tokens = [current_token] + draft_tokens[:n_accept]
                _ = model.model(mx.array([commit_tokens], dtype=mx.int32), cache=cache)

                # 3. Yield verified draft tokens
                for i in range(n_accept):
                    tok = draft_tokens[i]
                    generated.append(tok)
                    all_tokens.append(tok)
                    yield tok, True, stats
                    if len(generated) >= max_tokens or tok in stop_tokens:
                        return

                # 4. Yield the divergence token (already predicted by target model at position n_accept)
                divergence_token = preds_list[n_accept]
                current_token = divergence_token
                generated.append(current_token)
                all_tokens.append(current_token)
                yield current_token, False, stats
                if current_token in stop_tokens:
                    break


# Alias for universal speculative router
generate_pld_speculative = pld_generate

