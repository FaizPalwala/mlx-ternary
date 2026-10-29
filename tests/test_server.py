import unittest
from pathlib import Path
import mlx.core as mx

from mlx_ternary.server import setup_model_loader

class TestServerLoader(unittest.TestCase):
    def test_setup_loader_none_for_missing_path(self):
        # A nonexistent directory returns None gracefully
        loader = setup_model_loader(Path("/nonexistent/path"))
        self.assertIsNone(loader)

    def test_setup_loader_with_model(self):
        # Test with testing-artifacts model if present
        model_path = Path(__file__).resolve().parent.parent.parent / "testing-artifacts" / "models" / "Ternary-Bonsai-2-27B-mlx-2bit"
        if not (model_path / "runtime").exists():
            self.skipTest("Bonsai model runtime directory not found")

        try:
            import mlx_lm
            import transformers
        except ImportError:
            self.skipTest("mlx_lm and transformers required for server loader test")

        loader = setup_model_loader(model_path, kv_bits=8, kv_group_size=64)
        self.assertIsNotNone(loader)

        model, tokenizer = loader()
        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)

        # Verify custom quantized cache is instantiated
        cache = model.make_cache()
        self.assertTrue(len(cache) > 0)
        from mlx_lm.models.cache import QuantizedKVCache
        self.assertTrue(any(isinstance(c, QuantizedKVCache) for c in cache))

if __name__ == "__main__":
    unittest.main()
