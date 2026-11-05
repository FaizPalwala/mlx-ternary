import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from .ops import ternary_matmul, ternary_swiglu, ternary_fused_kv, ternary_fused_qkv


def fwht(x: mx.array, block: int, signs: Optional[mx.array] = None, inverse: bool = False) -> mx.array:
    """
    Normalized Sylvester-Walsh-Hadamard Transform on Apple Silicon.
    Scales by 1/sqrt(block) using native mx.hadamard_transform.
    """
    if block <= 0:
        return x

    shape, dtype = x.shape, x.dtype
    if shape[-1] % block != 0:
        raise ValueError(f"Hadamard block {block} does not divide activation width {shape[-1]}")

    x_f32 = x.astype(mx.float32)
    if signs is not None and not inverse:
        x_f32 = x_f32 * signs

    x_rot = mx.hadamard_transform(x_f32.reshape(-1, block), scale=1.0 / math.sqrt(block)).reshape(shape)

    if signs is not None and inverse:
        x_rot = x_rot * signs

    return x_rot.astype(dtype)

from .quantize import pack_interleaved

class TernaryLinear(nn.Module):
    """
    Model-agnostic Neural Network layer for 1.6-bit Ternary Matrix Multiplication.
    Acts as a drop-in replacement for standard `nn.Linear` in MLX models,
    or initialized directly from pre-packed interleaved weights.
    """
    def __init__(
        self,
        input_dims_or_weight,
        output_dims: int = None,
        bias: bool = False,
        group_size: int = 128,
        pad_k: int = 0,
        block: int = 0,
        signs: Optional[mx.array] = None,
    ):
        super().__init__()
        self.group_size = group_size
        self.pad_k = pad_k
        self.bias = None
        self.block = block
        self.signs = signs

        if isinstance(input_dims_or_weight, mx.array):
            self.weight_packed = input_dims_or_weight
            self.input_dims = None
            self.output_dims = self.weight_packed.shape[1]
        else:
            self.input_dims = input_dims_or_weight
            self.output_dims = output_dims
            self.pad_k = (group_size - (self.input_dims % group_size)) % group_size
            padded_k = self.input_dims + self.pad_k
            blocks = padded_k // group_size
            self.weight_packed = mx.zeros((blocks, output_dims, 32), dtype=mx.uint8)
            if bias:
                self.bias = mx.zeros((output_dims,), dtype=mx.float16)

    def quantize_from_raw(self, w_raw: mx.array):
        """
        Quantize FP16 weights into 32-byte cache-line interleaved ternary blocks.
        Accepts standard PyTorch/MLX weight shape (output_dims, input_dims) or transposed (input_dims, output_dims).
        """
        if self.output_dims is not None and self.input_dims is not None:
            if w_raw.shape == (self.output_dims, self.input_dims):
                w_raw = w_raw.T
        self.weight_packed, self.pad_k = pack_interleaved(w_raw, group_size=self.group_size)

    @classmethod
    def from_raw(
        cls,
        w_raw: mx.array,
        bias: bool = False,
        group_size: int = 128,
        block: int = 0,
        signs: Optional[mx.array] = None,
    ) -> "TernaryLinear":
        """
        Construct TernaryLinear directly from unquantized FP16 weights.
        Accepts weights shaped as (K, N) or PyTorch standard (N, K).
        """
        if w_raw.ndim != 2:
            raise ValueError(f"Expected 2D weight tensor, got shape {w_raw.shape}")
        w_packed, pad_k = pack_interleaved(w_raw, group_size=group_size)
        layer = cls(w_packed, bias=bias, group_size=group_size, pad_k=pad_k, block=block, signs=signs)
        layer.input_dims = w_raw.shape[0]
        layer.output_dims = w_raw.shape[1]
        return layer

    def __call__(self, x: mx.array) -> mx.array:
        if self.block > 0 and self.signs is not None:
            x = fwht(x, self.block, self.signs)
        elif self.block > 0:
            x = fwht(x, self.block)
        if self.pad_k > 0:
            *batch, k = x.shape
            x = mx.concatenate([x, mx.zeros((*batch, self.pad_k), dtype=x.dtype)], axis=-1)
        out = ternary_matmul(x, self.weight_packed, group_size=self.group_size)
        if self.bias is not None:
            out = out + self.bias
        return out

class BonsaiTernaryLinear(TernaryLinear):
    """
    PrismML Bonsai style Linear layer with Fast Walsh-Hadamard Transform (FWHT) preprocessing.
    """
    def __init__(
        self,
        weight_packed,
        output_dims: int = None,
        bias: bool = False,
        group_size: int = 128,
        pad_k: int = 0,
        block: int = 0,
        signs: Optional[mx.array] = None,
    ):
        super().__init__(
            weight_packed,
            output_dims=output_dims,
            bias=bias,
            group_size=group_size,
            pad_k=pad_k,
            block=block,
            signs=signs,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x)


class TernarySwiGLU(nn.Module):
    """
    Fused Ternary SwiGLU projection layer: SiLU(x @ w_gate) * (x @ w_up).

    Executes single-pass memory traversal across interleaved ternary weights
    with in-register activation evaluation.
    """
    def __init__(
        self,
        input_dims_or_gate=None,
        output_dims_or_up=None,
        output_dims: int = None,
        group_size: int = 128,
        pad_k: int = 0,
        block: int = 0,
        signs: Optional[mx.array] = None,
        in_features: int = None,
        out_features: int = None,
    ):
        super().__init__()
        self.group_size = group_size
        self.pad_k = pad_k
        self.block = block
        self.signs = signs

        if in_features is not None:
            input_dims_or_gate = in_features
        if out_features is not None:
            output_dims_or_up = out_features

        if isinstance(input_dims_or_gate, mx.array) and isinstance(output_dims_or_up, mx.array):
            self.gate_packed = input_dims_or_gate
            self.up_packed = output_dims_or_up
            self.input_dims = None
            self.output_dims = self.gate_packed.shape[1]
        else:
            self.input_dims = input_dims_or_gate
            self.output_dims = output_dims if output_dims is not None else output_dims_or_up
            if self.input_dims is not None and self.output_dims is not None:
                self.pad_k = (group_size - (self.input_dims % group_size)) % group_size
                padded_k = self.input_dims + self.pad_k
                blocks = padded_k // group_size
                self.gate_packed = mx.zeros((blocks, self.output_dims, 32), dtype=mx.uint8)
                self.up_packed = mx.zeros((blocks, self.output_dims, 32), dtype=mx.uint8)
            else:
                self.gate_packed = None
                self.up_packed = None

    def __setattr__(self, key: str, val):
        if key == "w_gate":
            super().__setattr__("gate_packed", val)
            return
        elif key == "w_up":
            super().__setattr__("up_packed", val)
            return
        super().__setattr__(key, val)

    def quantize_from_raw(self, w_gate_raw: mx.array, w_up_raw: mx.array):
        """
        Quantize FP16 Gate and Up weights into 32-byte cache-line interleaved ternary blocks.
        Accepts standard PyTorch/MLX weight shape (output_dims, input_dims) or transposed (input_dims, output_dims).
        """
        if self.output_dims is not None and self.input_dims is not None:
            if w_gate_raw.shape == (self.output_dims, self.input_dims):
                w_gate_raw = w_gate_raw.T
            if w_up_raw.shape == (self.output_dims, self.input_dims):
                w_up_raw = w_up_raw.T
        self.gate_packed, self.pad_k = pack_interleaved(w_gate_raw, group_size=self.group_size)
        self.up_packed, _ = pack_interleaved(w_up_raw, group_size=self.group_size)

    @classmethod
    def from_raw(
        cls,
        w_gate: mx.array,
        w_up: mx.array,
        group_size: int = 128,
        block: int = 0,
        signs: Optional[mx.array] = None,
    ) -> "TernarySwiGLU":
        """
        Construct TernarySwiGLU directly from unquantized FP16 weights.
        Accepts weights shaped as (K, N) or PyTorch standard (N, K).
        """
        if w_gate.ndim == 2 and w_up.ndim == 2:
            if w_gate.shape[0] == w_up.shape[0]:
                pass
            elif w_gate.shape[1] == w_up.shape[1]:
                w_gate = w_gate.T
                w_up = w_up.T
            else:
                raise ValueError(
                    f"Incompatible weight shapes for SwiGLU: w_gate={w_gate.shape}, w_up={w_up.shape}"
                )
        gate_packed, pad_k = pack_interleaved(w_gate, group_size=group_size)
        up_packed, _ = pack_interleaved(w_up, group_size=group_size)
        layer = cls(gate_packed, up_packed, group_size=group_size, pad_k=pad_k, block=block, signs=signs)
        layer.input_dims = w_gate.shape[0]
        layer.output_dims = w_gate.shape[1]
        return layer

    def __call__(self, x: mx.array) -> mx.array:
        if self.block > 0 and self.signs is not None:
            x = fwht(x, self.block, self.signs)
        elif self.block > 0:
            x = fwht(x, self.block)
        if self.pad_k > 0:
            *batch, k = x.shape
            x = mx.concatenate([x, mx.zeros((*batch, self.pad_k), dtype=x.dtype)], axis=-1)
        gate = self.gate_packed if self.gate_packed is not None else getattr(self, "w_gate", None)
        up = self.up_packed if self.up_packed is not None else getattr(self, "w_up", None)
        return ternary_swiglu(x, gate, up, group_size=self.group_size)


class TernaryFusedKV(nn.Module):
    """
    Fused Key and Value projection layer: x @ w_k and x @ w_v.

    Reuses loaded activation vector chunks in registers across dual-issue ALU pipelines,
    evaluating both projections concurrently in a single kernel dispatch.
    """
    def __init__(
        self,
        input_dims_or_k,
        kv_dims_or_v=None,
        kv_dims: int = None,
        group_size: int = 128,
        pad_k: int = 0,
        block: int = 0,
        signs: Optional[mx.array] = None,
    ):
        super().__init__()
        self.group_size = group_size
        self.pad_k = pad_k
        self.block = block
        self.signs = signs

        if isinstance(input_dims_or_k, mx.array) and isinstance(kv_dims_or_v, mx.array):
            self.k_packed = input_dims_or_k
            self.v_packed = kv_dims_or_v
            self.input_dims = None
            self.kv_dims = self.k_packed.shape[1]
        else:
            self.input_dims = input_dims_or_k
            self.kv_dims = kv_dims if kv_dims is not None else kv_dims_or_v
            self.pad_k = (group_size - (self.input_dims % group_size)) % group_size
            padded_k = self.input_dims + self.pad_k
            blocks = padded_k // group_size
            self.k_packed = mx.zeros((blocks, self.kv_dims, 32), dtype=mx.uint8)
            self.v_packed = mx.zeros((blocks, self.kv_dims, 32), dtype=mx.uint8)

    @classmethod
    def from_raw(
        cls,
        w_k: mx.array,
        w_v: mx.array,
        group_size: int = 128,
        block: int = 0,
        signs: Optional[mx.array] = None,
    ) -> "TernaryFusedKV":
        """
        Construct TernaryFusedKV directly from unquantized FP16 weights.
        Accepts weights shaped as (K, N) or PyTorch standard (N, K).
        """
        if w_k.ndim == 2 and w_v.ndim == 2:
            if w_k.shape[0] == w_v.shape[0]:
                pass  # Contraction axis is already axis 0
            elif w_k.shape[1] == w_v.shape[1]:
                w_k = w_k.T
                w_v = w_v.T
            else:
                raise ValueError(
                    f"Incompatible weight shapes for KV fusion: w_k={w_k.shape}, w_v={w_v.shape}"
                )
        k_packed, pad_k = pack_interleaved(w_k, group_size=group_size)
        v_packed, _ = pack_interleaved(w_v, group_size=group_size)
        layer = cls(k_packed, v_packed, group_size=group_size, pad_k=pad_k, block=block, signs=signs)
        layer.input_dims = w_k.shape[0]
        layer.kv_dims = w_k.shape[1]
        return layer

    def quantize_from_raw(self, w_k_raw: mx.array, w_v_raw: mx.array):
        """
        Quantize FP16 K and V weights into 32-byte cache-line interleaved ternary blocks.
        """
        if self.kv_dims is not None and self.input_dims is not None:
            if w_k_raw.shape == (self.kv_dims, self.input_dims):
                w_k_raw = w_k_raw.T
            if w_v_raw.shape == (self.kv_dims, self.input_dims):
                w_v_raw = w_v_raw.T
        self.k_packed, self.pad_k = pack_interleaved(w_k_raw, group_size=self.group_size)
        self.v_packed, _ = pack_interleaved(w_v_raw, group_size=self.group_size)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        if self.block > 0 and self.signs is not None:
            x = fwht(x, self.block, self.signs)
        elif self.block > 0:
            x = fwht(x, self.block)
        if self.pad_k > 0:
            *batch, k = x.shape
            x = mx.concatenate([x, mx.zeros((*batch, self.pad_k), dtype=x.dtype)], axis=-1)
        return ternary_fused_kv(x, self.k_packed, self.v_packed, group_size=self.group_size)


class TernaryQKV(nn.Module):
    """
    Unified Query, Key, and Value projection layer.

    Evaluates concatenated [Q, K, V] projections in a single 0-barrier wide GEMV dispatch
    and slices the output into query, key, and value activation tensors.
    """
    def __init__(
        self,
        input_dims_or_qkv=None,
        q_dim: int = None,
        k_dim: int = None,
        v_dim: int = None,
        split: bool = True,
        group_size: int = 128,
        pad_k: int = 0,
        block: int = 0,
        signs: Optional[mx.array] = None,
        in_features: int = None,
    ):
        super().__init__()
        self.group_size = group_size
        self.pad_k = pad_k
        self.split = split
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.block = block
        self.signs = signs

        if in_features is not None:
            input_dims_or_qkv = in_features

        if isinstance(input_dims_or_qkv, mx.array):
            self.qkv_packed = input_dims_or_qkv
            self.input_dims = None
            total_n = self.qkv_packed.shape[1]
            self.output_dims = total_n
        else:
            self.input_dims = input_dims_or_qkv
            if q_dim is not None and k_dim is not None and v_dim is not None:
                self.output_dims = q_dim + k_dim + v_dim
            else:
                self.output_dims = None
            if self.input_dims is not None and self.output_dims is not None:
                self.pad_k = (group_size - (self.input_dims % group_size)) % group_size
                padded_k = self.input_dims + self.pad_k
                blocks = padded_k // group_size
                self.qkv_packed = mx.zeros((blocks, self.output_dims, 32), dtype=mx.uint8)
            else:
                self.qkv_packed = None

    def __setattr__(self, key: str, val):
        if key == "w_qkv":
            super().__setattr__("qkv_packed", val)
            return
        super().__setattr__(key, val)

    @classmethod
    def from_raw(
        cls,
        w_q: mx.array,
        w_k: mx.array,
        w_v: mx.array,
        split: bool = True,
        group_size: int = 128,
        block: int = 0,
        signs: Optional[mx.array] = None,
    ) -> "TernaryQKV":
        """
        Construct TernaryQKV directly from unquantized FP16 weights.
        Accepts weights shaped as (K, N) or PyTorch standard (N, K).
        """
        if w_q.ndim == 2 and w_k.ndim == 2 and w_v.ndim == 2:
            if w_q.shape[0] == w_k.shape[0] == w_v.shape[0]:
                pass  # Contraction axis is already axis 0
            elif w_q.shape[1] == w_k.shape[1] == w_v.shape[1]:
                w_q = w_q.T
                w_k = w_k.T
                w_v = w_v.T
            else:
                raise ValueError(
                    f"Incompatible weight shapes for QKV fusion: w_q={w_q.shape}, w_k={w_k.shape}, w_v={w_v.shape}"
                )

        q_dim = w_q.shape[1]
        k_dim = w_k.shape[1]
        v_dim = w_v.shape[1]

        w_qkv = mx.concatenate([w_q, w_k, w_v], axis=1)
        qkv_packed, pad_k = pack_interleaved(w_qkv, group_size=group_size)
        layer = cls(
            qkv_packed,
            q_dim=q_dim,
            k_dim=k_dim,
            v_dim=v_dim,
            split=split,
            group_size=group_size,
            pad_k=pad_k,
            block=block,
            signs=signs,
        )
        layer.input_dims = w_q.shape[0]
        return layer

    def quantize_from_raw(self, w_q_raw: mx.array, w_k_raw: mx.array, w_v_raw: mx.array):
        """
        Quantize FP16 Q, K, and V weights into 32-byte cache-line interleaved ternary blocks.
        """
        if self.q_dim is not None and self.input_dims is not None:
            if w_q_raw.shape == (self.q_dim, self.input_dims):
                w_q_raw = w_q_raw.T
            if w_k_raw.shape == (self.k_dim, self.input_dims):
                w_k_raw = w_k_raw.T
            if w_v_raw.shape == (self.v_dim, self.input_dims):
                w_v_raw = w_v_raw.T
        w_qkv = mx.concatenate([w_q_raw, w_k_raw, w_v_raw], axis=1)
        self.qkv_packed, self.pad_k = pack_interleaved(w_qkv, group_size=self.group_size)

    def __call__(self, x: mx.array):
        if self.block > 0 and self.signs is not None:
            x = fwht(x, self.block, self.signs)
        elif self.block > 0:
            x = fwht(x, self.block)
        if self.pad_k > 0:
            *batch, k = x.shape
            x = mx.concatenate([x, mx.zeros((*batch, self.pad_k), dtype=x.dtype)], axis=-1)
        packed = self.qkv_packed if self.qkv_packed is not None else getattr(self, "w_qkv", None)
        out = ternary_fused_qkv(x, packed, group_size=self.group_size)
        if self.split and self.q_dim is not None and self.k_dim is not None:
            q = out[..., :self.q_dim]
            k = out[..., self.q_dim:self.q_dim + self.k_dim]
            v = out[..., self.q_dim + self.k_dim:]
            return q, k, v
        return out


