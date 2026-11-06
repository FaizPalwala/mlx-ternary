#!/usr/bin/env python3
"""
Train EarlyExitAdapter (RFC-002) for Layer-Pruned Self-Speculative Drafting.

Learns a low-rank mapping from draft layer hidden state (e.g. layer 16)
to target final layer hidden state (e.g. layer 64) using representation MSE loss.
Parameters: ~5.2M (FP16: ~10.5 MB).
Backbone weights remain frozen.
"""

import argparse
import time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_ternary.speculative.self_speculative import EarlyExitAdapter


def train(args):
    from mlx_lm import load

    print(f"Loading backbone model from {args.model_path}...")
    model, tokenizer = load(args.model_path)
    model.freeze()

    total_layers = len(model.model.layers)
    draft_layer = args.draft_layer if args.draft_layer is not None else max(4, total_layers // 4)
    print(f"Model loaded: {total_layers} total layers. Draft exit layer: {draft_layer}.")

    hidden_dim = getattr(model.config, "hidden_size", 5120)
    adapter = EarlyExitAdapter(hidden_dim=hidden_dim, rank=args.rank)
    mx.eval(adapter.parameters())

    optimizer = optim.AdamW(learning_rate=args.lr)

    # Synthetic / prompt calibration dataset for representation alignment
    calibration_prompts = [
        "Explain the theory of general relativity and its implications for spacetime curvature.",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n",
        "Write a comprehensive summary of quantum entanglement and quantum teleportation protocols.",
        "class TrieNode:\n    def __init__(self):\n        self.children = {}\n        self.is_end_of_word = False\n",
        "Apple Silicon unified memory architecture enables zero-copy tensor sharing between CPU and GPU.",
        "SELECT department, AVG(salary) FROM employees WHERE hire_date > '2023-01-01' GROUP BY department;",
        "The quick brown fox jumps over the lazy dog. Multilingual tokenization requires robust vocabulary.",
        "import numpy as np\nimport mlx.core as mx\n\ndef gelu_fast(x):\n    return 0.5 * x * (1.0 + mx.tanh(0.79788456 * (x + 0.044715 * mx.power(x, 3))))\n",
    ]

    tokens_list = [tokenizer.encode(p) for p in calibration_prompts]

    def loss_fn(adapter_module, x_tokens):
        # Extract draft representation h_draft and target representation h_final
        h = model.model.embed_tokens(x_tokens)
        
        # Forward through layers up to draft_layer
        for layer in model.model.layers[:draft_layer]:
            h = layer(h)
        h_draft = h

        # Forward remaining layers up to final layer
        for layer in model.model.layers[draft_layer:]:
            h = layer(h)
        h_target = h

        # Map h_draft through adapter and compute MSE loss against h_target
        h_predicted = adapter_module(h_draft)
        return mx.mean((h_predicted - h_target) ** 2)

    loss_and_grad_fn = nn.value_and_grad(adapter, loss_fn)

    print(f"Starting training for {args.steps} steps (rank={args.rank}, lr={args.lr})...")
    start_time = time.time()

    for step in range(1, args.steps + 1):
        prompt_idx = (step - 1) % len(tokens_list)
        toks = mx.array([tokens_list[prompt_idx]], dtype=mx.int32)

        loss, grads = loss_and_grad_fn(adapter, toks)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        if step % args.log_interval == 0 or step == args.steps:
            elapsed = time.time() - start_time
            print(f"Step {step:4d}/{args.steps} | MSE Loss: {loss.item():.6f} | Elapsed: {elapsed:.2f}s")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adapter.save_weights(str(output_path))
    print(f"Saved EarlyExitAdapter weights to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train RFC-002 EarlyExitAdapter for Self-Speculative Drafting")
    parser.add_argument("--model-path", type=str, required=True, help="Path to base model checkpoint")
    parser.add_argument("--output", "-o", type=str, default="early_exit_adapter.safetensors", help="Output .safetensors path")
    parser.add_argument("--draft-layer", type=int, default=16, help="Draft layer index to exit from (default: 16)")
    parser.add_argument("--rank", type=int, default=1024, help="Low-rank bottleneck dimension (default: 1024)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps (default: 100)")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval steps (default: 10)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
