import unittest
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn

from mlx_ternary.loader import auto_fuse_model_layers, load_model
from mlx_ternary.layers import TernarySwiGLU, TernaryQKV


class MockProj:
    def __init__(self, in_features: int, out_features: int, block: int = 0, signs: mx.array = None):
        blocks = in_features // 128
        self.weight = mx.zeros((blocks, out_features, 32), dtype=mx.uint8)
        self.block = block
        self.signs = signs
        self.group_size = 128


class MockMLP(nn.Module):
    def __init__(self, in_features: int = 512, out_features: int = 1024):
        super().__init__()
        self.gate_proj = MockProj(in_features, out_features, block=128, signs=mx.ones((in_features,)))
        self.up_proj = MockProj(in_features, out_features, block=128, signs=mx.ones((in_features,)))
        self.down_proj = nn.Identity()

    def __call__(self, x):
        return self.down_proj(x)


class MockSelfAttn(nn.Module):
    def __init__(self, in_features: int = 512, q_dim: int = 256, k_dim: int = 64, v_dim: int = 64):
        super().__init__()
        self.q_proj = MockProj(in_features, q_dim, block=128, signs=mx.ones((in_features,)))
        self.k_proj = MockProj(in_features, k_dim, block=128, signs=mx.ones((in_features,)))
        self.v_proj = MockProj(in_features, v_dim, block=128, signs=mx.ones((in_features,)))

    def __call__(self, x, mask=None, cache=None):
        return x


class MockLayer(nn.Module):
    def __init__(self, with_attn: bool = False):
        super().__init__()
        self.mlp = MockMLP()
        if with_attn:
            self.self_attn = MockSelfAttn()


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = [MockLayer(with_attn=False), MockLayer(with_attn=True)]


class TestLoader(unittest.TestCase):
    def test_auto_fuse_mock_model(self):
        """Verify auto_fuse_model_layers correctly traverses and binds SwiGLU and QKV."""
        model = MockModel()
        fused_count = auto_fuse_model_layers(model)

        # Layer 0 has MLP (1), Layer 1 has MLP (1) and Attention (1) -> 3 total
        self.assertEqual(fused_count, 3)

        # Verify Layer 0 MLP fusion
        layer0 = model.model.layers[0]
        self.assertTrue(hasattr(layer0.mlp, "swiglu"))
        self.assertIsInstance(layer0.mlp.swiglu, TernarySwiGLU)

        # Verify Layer 1 Attention fusion
        layer1 = model.model.layers[1]
        self.assertTrue(hasattr(layer1.mlp, "swiglu"))
        self.assertIsInstance(layer1.mlp.swiglu, TernarySwiGLU)
        self.assertTrue(hasattr(layer1.self_attn, "qkv"))
        self.assertIsInstance(layer1.self_attn.qkv, TernaryQKV)

        # Verify forward execution runs without error
        x = mx.random.normal((1, 512)).astype(mx.float16)
        out_mlp = layer0.mlp(x)
        mx.eval(out_mlp)
        self.assertEqual(out_mlp.shape, (1, 1024))

        out_attn = layer1.self_attn(x)
        mx.eval(out_attn)
        self.assertEqual(out_attn.shape, (1, 512))

    def test_auto_fuse_graceful_fallback(self):
        """Verify auto_fuse_model_layers returns 0 on non-conforming or empty inputs."""
        self.assertEqual(auto_fuse_model_layers(None), 0)
        self.assertEqual(auto_fuse_model_layers(object()), 0)

        empty_model = nn.Module()
        self.assertEqual(auto_fuse_model_layers(empty_model), 0)


if __name__ == "__main__":
    unittest.main()
