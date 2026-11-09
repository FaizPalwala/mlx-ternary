#!/usr/bin/env python3
"""
Train EarlyExitAdapter (RFC-002) for Layer-Pruned Self-Speculative Drafting.

Learns a low-rank mapping from draft layer hidden state (e.g. layer 16)
to target final layer hidden state (e.g. layer 64) using representation MSE loss.
Parameters: ~10.5M (FP16: ~10.5 MB).
Backbone weights remain frozen.
"""

import argparse
import time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tokenizers import Tokenizer

from mlx_ternary.loader import load_model
from mlx_ternary.speculative.self_speculative import EarlyExitAdapter
from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask


def train(args):
    print(f"Loading backbone model from {args.model_path}...")
    model, config = load_model(args.model_path, fuse=False)
    tokenizer = Tokenizer.from_file(str(Path(args.model_path) / "tokenizer.json"))
    model.freeze()

    total_layers = len(model.model.layers)
    draft_layer = args.draft_layer if args.draft_layer is not None else max(4, total_layers // 4)
    print(f"Model loaded: {total_layers} total layers. Draft exit layer: {draft_layer}.")

    hidden_dim = getattr(config, "hidden_size", 5120) if hasattr(config, "hidden_size") else 5120
    adapter = EarlyExitAdapter(hidden_dim=hidden_dim, rank=args.rank)
    mx.eval(adapter.parameters())

    optimizer = optim.AdamW(learning_rate=args.lr)

    # 16 Diverse calibration prompts covering JSON, Python code, and technical prose
    calibration_prompts = [
        # JSON structured data & schema keys
        '{\n  "name": "scale_k8s_deployment",\n  "description": "Scale Kubernetes deployment replicas in namespace",\n  "parameters": {\n    "type": "object",\n    "properties": {\n      "deployment": {"type": "string"},\n      "replicas": {"type": "integer"},\n      "namespace": {"type": "string", "default": "default"}\n    },\n    "required": ["deployment", "replicas"]\n  }\n}',
        '{\n  "query": "SELECT user_id, count(*), sum(amount) FROM transactions WHERE status = \'completed\' GROUP BY user_id ORDER BY sum(amount) DESC LIMIT 50;",\n  "timeout_seconds": 30,\n  "cache_results": true\n}',
        '{\n  "apiVersion": "apps/v1",\n  "kind": "Deployment",\n  "metadata": {"name": "checkout-service", "labels": {"app": "checkout"}},\n  "spec": {"replicas": 5, "template": {"spec": {"containers": [{"name": "web", "image": "checkout:v2.1"}]}}}\n}',
        '{\n  "error": null,\n  "status": 200,\n  "data": {\n    "cluster": "us-east1-production",\n    "nodes_healthy": 128,\n    "active_pods": 1024,\n    "cpu_utilization_pct": 68.4\n  }\n}',
        # Python code & async/await definitions
        'import asyncio\nfrom typing import Optional, List\n\nclass AsyncConnectionPool:\n    def __init__(self, host: str, port: int, max_connections: int = 10):\n        self.host = host\n        self.port = port\n        self.max_connections = max_connections\n        self._pool = asyncio.Queue(max_size=max_connections)\n        self._lock = asyncio.Lock()\n\n    async def acquire(self):\n        async with self._lock:\n            return await self._pool.get()\n',
        'import mlx.core as mx\nimport mlx.nn as nn\n\nclass TernaryLinear(nn.Module):\n    def __init__(self, in_features: int, out_features: int):\n        super().__init__()\n        self.weight = mx.zeros((out_features, in_features))\n        self.scale = mx.ones((out_features, 1))\n\n    def __call__(self, x: mx.array) -> mx.array:\n        return mx.matmul(x, self.weight.T) * self.scale\n',
        'async def fetch_all_pages(session, urls: List[str], concurrency_limit: int = 5) -> List[dict]:\n    semaphore = asyncio.Semaphore(concurrency_limit)\n    async def fetch_one(url):\n        async with semaphore:\n            async with session.get(url) as response:\n                return await response.json()\n    return await asyncio.gather(*(fetch_one(u) for u in urls))\n',
        'def topological_sort(graph: Dict[str, List[str]]) -> List[str]:\n    visited = set()\n    order = []\n    def dfs(node):\n        if node in visited:\n            return\n        visited.add(node)\n        for neighbor in graph.get(node, []):\n            dfs(neighbor)\n        order.append(node)\n    for n in graph:\n        dfs(n)\n    return order[::-1]\n',
        # Technical prose & systems architecture
        'Apple Silicon unified memory architecture (UMA) allows the CPU, GPU, and Neural Engine to access high-bandwidth LPDDR5 memory without copying data across PCIe buses, achieving bandwidths exceeding 300 GB/s.',
        'Raft state machine replication requires that log entries are committed only after being written to durable disk storage and acknowledged by a quorum majority of cluster nodes across independent failure domains.',
        'Speculative decoding accelerates language model inference by executing an inexpensive draft model to propose K candidate tokens, followed by a single batched verification pass on the full target model.',
        'Prompt Lookup Decoding (PLD) exploits n-gram repetitions in prompt context to propose candidate tokens with zero additional VRAM overhead, ideal for document summarization and schema-driven generation.',
        # Mathematical reasoning & scientific computing
        'The singular value decomposition of matrix A in R^{m x n} is A = U Sigma V^T, where U is an m x m orthogonal matrix, Sigma contains singular values in descending order, and V is an n x n orthogonal matrix.',
        'In quantum electrodynamics, the Dirac equation combines quantum mechanics with special relativity: (i gamma^mu partial_mu - m) psi = 0, predicting antimatter and describing spin-1/2 fermions.',
        'Gradient descent with momentum accelerates convergence along directions of low curvature: v_{t+1} = beta v_t + (1 - beta) grad_w L(w), followed by parameter update w_{t+1} = w_t - alpha v_{t+1}.',
        'BitNet b1.58 quantizes linear layer weights into {-1, 0, +1}, turning float multiplications into integer additions and dramatically reducing memory bandwidth requirements on modern neural accelerators.',
    ]

    tokens_list = [tokenizer.encode(p).ids for p in calibration_prompts]

    print(f"Precomputing representations for {len(tokens_list)} calibration sequences...")
    cached_data = []
    for toks_raw in tokens_list:
        toks = mx.array([toks_raw], dtype=mx.int32)
        h = model.model.embed_tokens(toks)
        fa_mask = create_attention_mask(h)
        ssm_mask = create_ssm_mask(h)

        for layer in model.model.layers[:draft_layer]:
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, mask=mask)
        h_draft = mx.stop_gradient(h)

        for layer in model.model.layers[draft_layer:]:
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, mask=mask)
        h_target = mx.stop_gradient(h)

        mx.eval(h_draft, h_target)
        cached_data.append((h_draft, h_target, toks))

    print("Precomputation complete. Training adapter with Cosine + Cross-Entropy Loss...")

    def loss_fn(adapter_module, h_draft, h_target, x_tokens):
        h_pred = adapter_module(h_draft)
        
        # Cosine distance loss for representation alignment
        norm_pred = h_pred / mx.linalg.norm(h_pred, axis=-1, keepdims=True)
        norm_target = h_target / mx.linalg.norm(h_target, axis=-1, keepdims=True)
        cos_loss = 1.0 - mx.mean(mx.sum(norm_pred * norm_target, axis=-1))

        # Cross-Entropy loss for next-token prediction
        norm_h = model.model.norm(h_pred)
        logits = model.lm_head(norm_h)
        ce_loss = mx.mean(nn.losses.cross_entropy(logits[:, :-1, :], x_tokens[:, 1:]))

        return cos_loss * 10.0 + ce_loss

    loss_and_grad_fn = nn.value_and_grad(adapter, loss_fn)

    print(f"Starting training for {args.steps} steps (rank={args.rank}, lr={args.lr})...")
    start_time = time.time()

    for step in range(1, args.steps + 1):
        idx = (step - 1) % len(cached_data)
        h_d, h_t, toks = cached_data[idx]

        loss, grads = loss_and_grad_fn(adapter, h_d, h_t, toks)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        if step % args.log_interval == 0 or step == args.steps:
            elapsed = time.time() - start_time
            print(f"Step {step:4d}/{args.steps} | Loss: {loss.item():.4f} | Elapsed: {elapsed:.2f}s")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import mlx.utils as utils
    adapter.update(utils.tree_map(lambda p: p.astype(mx.float16), adapter.parameters()))
    adapter.save_weights(str(output_path))
    print(f"Saved EarlyExitAdapter FP16 weights to {output_path}")


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
