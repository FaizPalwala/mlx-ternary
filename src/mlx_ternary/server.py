import sys
from pathlib import Path
import mlx.core as mx

def setup_model_loader(model_path: Path):
    model_path = Path(model_path)
    runtime_dir = model_path / "runtime"
    
    if runtime_dir.exists():
        sys.path.insert(0, str(runtime_dir))
        try:
            from artifact import load_model
            from transformers import PreTrainedTokenizerFast

            def custom_load(*args, **kwargs):
                print(f"[mlx_ternary] Loading Ternary Model from {model_path}...")
                model, config = load_model(model_path)
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(model_path / "tokenizer.json"))
                chat_template_file = model_path / "chat_template.jinja"
                if chat_template_file.exists():
                    tokenizer.chat_template = chat_template_file.read_text()
                return model, tokenizer

            return custom_load
        except ImportError as e:
            print(f"[mlx_ternary] Notice: Could not import artifact loader ({e}). Falling back to standard mlx_lm.")
            return None
    return None

def run_server(model_dir: str | Path, host: str = "127.0.0.1", port: int = 8080, draft_model: str = None):
    model_path = Path(model_dir)
    custom_loader = setup_model_loader(model_path)
    
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
    parser.add_argument("--draft-model", type=str, default=None, help="Draft model for speculative decoding")
    args = parser.parse_args()

    run_server(args.model, args.host, args.port, args.draft_model)
