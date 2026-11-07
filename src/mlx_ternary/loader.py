import sys
import json
from pathlib import Path
from typing import Tuple, Optional, Any, Dict
import mlx.core as mx
import mlx.nn as nn

from .layers import TernarySwiGLU, TernaryQKV, TernaryLinear


def auto_fuse_model_layers(model: Any) -> int:
    """
    Scans a loaded model and in-memory replaces:
    1. layer.mlp (gate_proj + up_proj) with TernarySwiGLU
    2. layer.self_attn (q_proj + k_proj + v_proj) with TernaryQKV (if full_attention)
    Returns total number of fused modules created.
    """
    fused_count = 0
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers

    if layers is None:
        return 0

    for layer in layers:
        # 1. Fuse MLP SwiGLU (Gate + Up)
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate_proj") and hasattr(layer.mlp, "up_proj"):
            gate = layer.mlp.gate_proj
            up = layer.mlp.up_proj

            # Extract block & signs if present (Hadamard-aware)
            block = getattr(gate, "block", 0)
            signs = getattr(gate, "signs", None)
            group_size = getattr(gate, "group_size", 128)

            in_features = getattr(gate, "input_dims", None)
            if in_features is None:
                if hasattr(gate.weight, "shape"):
                    in_features = gate.weight.shape[0] * group_size if gate.weight.ndim == 3 else gate.weight.shape[0]
                else:
                    in_features = 5120

            out_features = getattr(gate, "output_dims", None)
            if out_features is None:
                out_features = gate.weight.shape[1] if hasattr(gate.weight, "shape") else 17408

            pad_k = getattr(gate, "pad_k", 0)

            swiglu = TernarySwiGLU(
                in_features=in_features,
                out_features=out_features,
                group_size=group_size,
                pad_k=pad_k,
                block=block,
                signs=signs,
            )
            swiglu.w_gate = gate.weight
            swiglu.w_up = up.weight

            class FusedMLPWrapper(nn.Module):
                def __init__(self, swiglu_mod, down_proj):
                    super().__init__()
                    self.swiglu = swiglu_mod
                    self.down_proj = down_proj

                def __call__(self, x):
                    return self.down_proj(self.swiglu(x))

            layer.mlp = FusedMLPWrapper(swiglu, layer.mlp.down_proj)
            fused_count += 1

        # 2. Fuse Attention QKV (for full_attention layers)
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj") and hasattr(layer.self_attn, "k_proj") and hasattr(layer.self_attn, "v_proj"):
            qp = layer.self_attn.q_proj
            kp = layer.self_attn.k_proj
            vp = layer.self_attn.v_proj

            block = getattr(qp, "block", 0)
            signs = getattr(qp, "signs", None)
            group_size = getattr(qp, "group_size", 128)

            in_features = getattr(qp, "input_dims", None)
            if in_features is None:
                if hasattr(qp.weight, "shape"):
                    in_features = qp.weight.shape[0] * group_size if qp.weight.ndim == 3 else qp.weight.shape[0]
                else:
                    in_features = 5120

            pad_k = getattr(qp, "pad_k", 0)

            q_dim = qp.weight.shape[1] if hasattr(qp.weight, "shape") else 3072
            k_dim = kp.weight.shape[1] if hasattr(kp.weight, "shape") else 512
            v_dim = vp.weight.shape[1] if hasattr(vp.weight, "shape") else 512

            qkv = TernaryQKV(
                in_features=in_features,
                q_dim=q_dim,
                k_dim=k_dim,
                v_dim=v_dim,
                group_size=group_size,
                pad_k=pad_k,
                block=block,
                signs=signs,
            )
            # Concat weights across output dimension
            qkv.w_qkv = mx.concatenate([qp.weight, kp.weight, vp.weight], axis=1) if hasattr(qp.weight, "shape") else None

            # Wrap attention to use single-pass QKV dispatch
            orig_self_attn = layer.self_attn

            class FusedAttentionWrapper(nn.Module):
                def __init__(self, qkv_mod, orig_attn):
                    super().__init__()
                    self.qkv = qkv_mod
                    self.orig = orig_attn

                def __call__(
                    self,
                    x: mx.array,
                    mask: Optional[mx.array] = None,
                    cache: Optional[Any] = None,
                ) -> mx.array:
                    is_2d = x.ndim == 2
                    if is_2d:
                        x = x[:, None, :]
                    B, L, D = x.shape

                    # 1. Single-pass fused QKV dispatch (1 FWHT pass)
                    q_proj_out, keys, values = self.qkv(x)

                    # Fallback for mock/non-standard attention modules
                    if not hasattr(self.orig, "q_norm") or not hasattr(self.orig, "o_proj"):
                        if hasattr(self.orig, "_forward_with_qkv"):
                            out = self.orig._forward_with_qkv(x, q_proj_out, keys, values, mask=mask, cache=cache)
                        else:
                            out = self.orig(x, mask=mask, cache=cache)
                        return out.squeeze(1) if is_2d else out

                    # 2. Handle Qwen output gating (split queries and gate if head dimension is doubled)
                    if hasattr(self.orig, "num_attention_heads") and hasattr(self.orig, "head_dim"):
                        expected_q_dim = self.orig.num_attention_heads * self.orig.head_dim * 2
                        if q_proj_out.shape[-1] == expected_q_dim:
                            queries, gate = mx.split(
                                q_proj_out.reshape(B, L, self.orig.num_attention_heads, -1), 2, axis=-1
                            )
                            gate = gate.reshape(B, L, -1)
                        else:
                            queries = q_proj_out.reshape(B, L, self.orig.num_attention_heads, -1)
                            gate = None
                    else:
                        queries = q_proj_out
                        gate = None

                    # 3. Apply RMSNorms and transposes
                    queries = self.orig.q_norm(queries).transpose(0, 2, 1, 3)
                    keys = self.orig.k_norm(keys.reshape(B, L, self.orig.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
                    values = values.reshape(B, L, self.orig.num_key_value_heads, -1).transpose(0, 2, 1, 3)

                    # 4. RoPE & KV Cache updates
                    if cache is not None:
                        queries = self.orig.rope(queries, offset=cache.offset)
                        keys = self.orig.rope(keys, offset=cache.offset)
                        keys, values = cache.update_and_fetch(keys, values)
                    else:
                        queries = self.orig.rope(queries)
                        keys = self.orig.rope(keys)

                    # 5. Scaled Dot-Product Attention
                    try:
                        output = mx.fast.scaled_dot_product_attention(
                            queries, keys, values, cache=cache, scale=self.orig.scale, mask=mask
                        )
                    except TypeError:
                        try:
                            from mlx_lm.models.base import scaled_dot_product_attention
                            output = scaled_dot_product_attention(
                                queries, keys, values, cache=cache, scale=self.orig.scale, mask=mask
                            )
                        except Exception:
                            output = mx.fast.scaled_dot_product_attention(
                                queries, keys, values, scale=self.orig.scale, mask=mask
                            )

                    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

                    # 6. Apply attention output gate (if present) and project out
                    if gate is not None:
                        output = output * mx.sigmoid(gate)

                    out = self.orig.o_proj(output)
                    return out.squeeze(1) if is_2d else out

            layer.self_attn = FusedAttentionWrapper(qkv, orig_self_attn)
            fused_count += 1

    return fused_count


def load_model(model_path: str | Path, fuse: bool = True, **kwargs) -> Tuple[Any, Any]:
    """
    Loads a model and tokenizer, automatically applying kernel fusion (SwiGLU and QKV).
    Supports:
    1. Custom artifact loaders (e.g. PrismML / Bonsai packs with runtime/artifact.py)
    2. Standard mlx-lm models
    """
    model_path = Path(model_path)
    runtime_dir = model_path / "runtime"

    if runtime_dir.exists():
        sys.path.insert(0, str(runtime_dir))
        try:
            from artifact import load_model as artifact_load
            from transformers import PreTrainedTokenizerFast
            from mlx_lm.tokenizer_utils import TokenizerWrapper

            model, config = artifact_load(model_path)
            raw_tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(model_path / "tokenizer.json"))
            chat_template_file = model_path / "chat_template.jinja"
            if chat_template_file.exists():
                raw_tokenizer.chat_template = chat_template_file.read_text()
            tokenizer = TokenizerWrapper(raw_tokenizer)
            if fuse:
                fused = auto_fuse_model_layers(model)
                if fused > 0:
                    print(f"[mlx_ternary] Auto-fused {fused} layers (SwiGLU & QKV).")
            return model, tokenizer
        except Exception:
            pass

    import mlx_lm
    model, tokenizer = mlx_lm.load(str(model_path), **kwargs)
    if fuse:
        fused = auto_fuse_model_layers(model)
        if fused > 0:
            print(f"[mlx_ternary] Auto-fused {fused} layers (SwiGLU & QKV).")
    return model, tokenizer


def load_ternary_model(model_path: str) -> Tuple[Dict[str, mx.array], Dict[str, mx.array]]:
    """
    Loads a ternary safetensors model.
    """
    raw_state_dict = mx.load(model_path)
    weights = {}
    scales = {}
    for key, tensor in raw_state_dict.items():
        if key.endswith(".weight_scale"):
            base_name = key.replace(".weight_scale", "")
            scales[base_name] = tensor
        elif key.endswith(".weight"):
            base_name = key.replace(".weight", "")
            weights[base_name] = tensor
        else:
            weights[key] = tensor
    return weights, scales
