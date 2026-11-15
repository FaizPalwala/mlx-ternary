import sys
from pathlib import Path
import mlx.core as mx

def setup_model_loader(model_path: Path, kv_bits: int = None, kv_group_size: int = 64, max_kv_size: int = None):
    model_path = Path(model_path)
    runtime_dir = model_path / "runtime"
    
    if runtime_dir.exists():
        sys.path.insert(0, str(runtime_dir))
        try:
            from artifact import load_model
            from transformers import PreTrainedTokenizerFast
            from mlx_lm.models.cache import QuantizedKVCache, KVCache, RotatingKVCache, ArraysCache

            def custom_load(*args, **kwargs):
                print(f"[mlx_ternary] Loading Ternary Model from {model_path}...")
                model, config = load_model(model_path)
                
                from .loader import auto_fuse_model_layers
                fused = auto_fuse_model_layers(model)
                if fused > 0:
                    print(f"[mlx_ternary] Auto-fused {fused} projection layers.")
                
                if kv_bits in (4, 8) or max_kv_size is not None:
                    def make_custom_cache():
                        c = []
                        layers = getattr(model, "model", model).layers
                        for l in layers:
                            if getattr(l, "is_linear", False):
                                c.append(ArraysCache(size=2))
                            elif max_kv_size is not None:
                                c.append(RotatingKVCache(max_size=max_kv_size))
                            elif kv_bits in (4, 8):
                                c.append(QuantizedKVCache(bits=kv_bits, group_size=kv_group_size))
                            else:
                                c.append(KVCache())
                        return c
                    model.make_cache = make_custom_cache
                    print(f"[mlx_ternary] Configured KV Cache: bits={kv_bits or 16}, group_size={kv_group_size}, max_kv_size={max_kv_size}")

                from mlx_lm.tokenizer_utils import TokenizerWrapper
                raw_tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(model_path / "tokenizer.json"))
                chat_template_file = model_path / "chat_template.jinja"
                if chat_template_file.exists():
                    raw_tokenizer.chat_template = chat_template_file.read_text()
                tokenizer = TokenizerWrapper(raw_tokenizer)
                return model, tokenizer

            return custom_load
        except ImportError as e:
            print(f"[mlx_ternary] Notice: Could not import artifact loader ({e}). Falling back to standard mlx_lm.")
            return None
    return None

def run_server(
    model_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 8080,
    draft_model: str = None,
    kv_bits: int = None,
    kv_group_size: int = 64,
    max_kv_size: int = None
):
    model_path = Path(model_dir)
    custom_loader = setup_model_loader(model_path, kv_bits=kv_bits, kv_group_size=kv_group_size, max_kv_size=max_kv_size)
    
    try:
        import mlx_lm.utils
        import mlx_lm.server
    except ImportError:
        raise ImportError(
            "mlx_lm is required for serving. Install it via: pip install mlx-lm"
        )
        
    if custom_loader is not None:
        mlx_lm.utils.load = custom_loader
        mlx_lm.server.load = custom_loader
        mlx_lm.load = custom_loader

    server_args = [
        "mlx_lm.server",
        "--model", str(model_path),
        "--host", host,
        "--port", str(port),
    ]

    if draft_model:
        from .cli import SPECULATIVE_EXPERIMENTAL_BANNER
        print(SPECULATIVE_EXPERIMENTAL_BANNER, file=sys.stderr)
        print(f"[mlx_ternary] Speculative Decoding enabled with draft model: {draft_model}")
        server_args.extend(["--draft-model", draft_model])

    sys.argv = server_args
    print(f"[mlx_ternary] Starting OpenAI-compatible server on http://{host}:{port}...")
    mlx_lm.server.main()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="mlx_ternary OpenAI-Compatible Model Server")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument(
        "--speculative-experimental",
        "--draft-model",
        dest="draft_model",
        type=str,
        default=None,
        help="Draft model for experimental speculative decoding (RFC-002)",
    )
    parser.add_argument("--kv-bits", type=int, choices=[4, 8, 16], default=None, help="KV cache quantization bits (4, 8, or 16/FP16)")
    parser.add_argument("--kv-group-size", type=int, default=64, help="KV cache quantization group size")
    parser.add_argument("--max-kv-size", type=int, default=None, help="Maximum KV cache size for rotating/sliding window")
    args = parser.parse_args()

    run_server(args.model, args.host, args.port, args.draft_model, kv_bits=args.kv_bits, kv_group_size=args.kv_group_size, max_kv_size=args.max_kv_size)
