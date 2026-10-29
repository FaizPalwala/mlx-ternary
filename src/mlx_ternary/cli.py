import argparse
from pathlib import Path
import mlx.core as mx
import time
import sys
import subprocess

from .quantize import pack_interleaved

CACHE_DIR = Path.home() / ".cache" / "mlx_ternary"
DEFAULT_DRAFT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

def download_and_transcode(repo_id: str, output_dir: str = None) -> Path:
    local_path = Path(repo_id)
    if local_path.exists() and local_path.is_dir():
        model_dir = local_path
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download models from Hugging Face. "
                "Install via: pip install huggingface_hub"
            )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Checking for {repo_id}...")
        model_dir = CACHE_DIR / repo_id.replace("/", "--")
        if not model_dir.exists():
            print(f"Downloading {repo_id} from Hugging Face...")
            snapshot_download(repo_id=repo_id, local_dir=str(model_dir))

    target_dir = Path(output_dir) if output_dir else model_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    interleaved_path = target_dir / "model_interleaved.safetensors"

    if interleaved_path.exists() and output_dir is None:
        print("Packed weights found. Ready for inference.")
        return model_dir

    print(f"Packed weights not found in {target_dir}. JIT Transcoding...")
    safetensors_files = sorted([
        f for f in model_dir.glob("*.safetensors")
        if f.name != "model_interleaved.safetensors"
    ])
    if not safetensors_files:
        raise FileNotFoundError(f"Could not find any .safetensors files in {model_dir}")

    weights = {}
    for sf in safetensors_files:
        print(f"Loading shard {sf.name}...")
        weights.update(mx.load(str(sf)))

    new_weights = {}
    prefixes = set()
    for k in weights.keys():
        if ".weight" in k and weights[k].dtype == mx.uint32:
            prefixes.add(k.replace(".weight", ""))

    for k, v in weights.items():
        if "signs" in k:
            new_weights[k] = v
            continue
        if any(k.startswith(p) for p in prefixes):
            continue
        new_weights[k] = v

    start = time.time()
    print(f"Found {len(prefixes)} layers to pack into interleaved format.")

    for i, p in enumerate(prefixes):
        print(f"[{i+1}/{len(prefixes)}] Packing {p} ...\r", end="", flush=True)
        w = weights[f"{p}.weight"]
        s = weights.get(f"{p}.scales")
        b = weights.get(f"{p}.biases")

        if s is None or b is None:
            continue

        w_fp16 = mx.dequantize(w, s, b, group_size=128, bits=2)
        w_interleaved, pad_k = pack_interleaved(w_fp16.T, group_size=128)
        mx.eval(w_interleaved)

        new_weights[f"{p}.weight_packed"] = w_interleaved
        if pad_k > 0:
            new_weights[f"{p}.pad_k"] = mx.array(pad_k, dtype=mx.uint32)

        del weights[f"{p}.weight"]
        del weights[f"{p}.scales"]
        del weights[f"{p}.biases"]

    print(f"\nSaving packed model to {interleaved_path} (took {time.time()-start:.2f}s)...")
    mx.save_safetensors(str(interleaved_path), new_weights)
    print("Packing complete!")
    return target_dir

def serve(
    repo_id: str,
    speculate: bool = False,
    draft_model: str = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    kv_bits: int = None,
    kv_group_size: int = 64,
    max_kv_size: int = None
):
    from .server import run_server
    
    # If repo_id points to a local directory, use it directly
    local_path = Path(repo_id)
    if local_path.exists() and local_path.is_dir():
        model_dir = local_path
    else:
        model_dir = download_and_transcode(repo_id)
        
    print(f"Booting server for {repo_id}...")
    draft = None
    if speculate or draft_model:
        draft = draft_model if draft_model else DEFAULT_DRAFT_MODEL
        print(f"Speculative Decoding ENABLED. Draft model: {draft}")

    run_server(
        model_dir=model_dir,
        host=host,
        port=port,
        draft_model=draft,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        max_kv_size=max_kv_size
    )

def generate_cli(
    repo_id: str,
    prompt: str,
    speculative: str = "auto",
    draft_layers: int = 16,
    draft_model: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    max_tokens: int = 256,
    verbose: bool = False,
):
    from .server import setup_model_loader
    from .speculative.router import generate_speculative

    local_path = Path(repo_id)
    if local_path.exists() and local_path.is_dir():
        model_dir = local_path
    else:
        model_dir = download_and_transcode(repo_id)

    custom_loader = setup_model_loader(model_dir)
    if custom_loader is not None:
        model, tokenizer = custom_loader()
    else:
        from mlx_lm import load
        model, tokenizer = load(str(model_dir))

    for chunk in generate_speculative(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        mode=speculative,
        max_tokens=max_tokens,
        draft_layers=draft_layers,
        draft_model_path=draft_model,
        verbose=verbose,
    ):
        print(chunk, end="", flush=True)
    print()

def main():
    parser = argparse.ArgumentParser(prog="mlx_ternary")
    subparsers = parser.add_subparsers(dest="command")
    
    # Serve subparser
    serve_parser = subparsers.add_parser("serve", help="Serve a model")
    serve_parser.add_argument("repo_id", help="Hugging Face Repo ID or local directory")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    serve_parser.add_argument("--speculate", action="store_true", help="Enable speculative decoding with a default draft model.")
    serve_parser.add_argument("--draft", dest="draft_model", type=str, help="Specify a custom draft model for speculative decoding.")
    serve_parser.add_argument("--kv-bits", type=int, choices=[4, 8, 16], default=None, help="KV cache quantization bits (4, 8, or 16/FP16)")
    serve_parser.add_argument("--kv-group-size", type=int, default=64, help="KV cache quantization group size")
    serve_parser.add_argument("--max-kv-size", type=int, default=None, help="Maximum KV cache size for rotating/sliding window")
    
    # Pack subparser
    pack_parser = subparsers.add_parser("pack", help="Explicitly pack a model into hardware-vectorized interleaved format")
    pack_parser.add_argument("repo_id", help="Hugging Face Repo ID or local directory")
    pack_parser.add_argument("--output", "-o", type=str, default=None, help="Target directory for packed weights (defaults to model directory)")

    # Generate subparser
    generate_parser = subparsers.add_parser("generate", help="Generate text using ternary inference")
    generate_parser.add_argument("repo_id", help="Hugging Face Repo ID or local directory")
    generate_parser.add_argument("prompt_pos", nargs="?", default=None, help="Prompt text (if not provided via --prompt)")
    generate_parser.add_argument("--prompt", "-p", type=str, default=None, help="Prompt text")
    generate_parser.add_argument(
        "--speculative",
        type=str,
        choices=["auto", "pld", "self", "qwen", "none"],
        default="auto",
        help="Speculative decoding mode (default: auto)",
    )
    generate_parser.add_argument(
        "--draft-layers",
        type=int,
        default=16,
        help="Number of layers for self-speculative draft model (default: 16)",
    )
    generate_parser.add_argument(
        "--draft-model",
        type=str,
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="Draft model repo ID for cross-model speculative decoding",
    )
    generate_parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    generate_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print speculative routing telemetry and token stats to stderr",
    )
    
    args = parser.parse_args()
    
    if args.command == "serve":
        serve(
            args.repo_id,
            args.speculate,
            args.draft_model,
            host=args.host,
            port=args.port,
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
            max_kv_size=args.max_kv_size
        )
    elif args.command == "pack":
        download_and_transcode(args.repo_id, output_dir=args.output)
    elif args.command == "generate":
        prompt = args.prompt or args.prompt_pos
        if not prompt:
            generate_parser.error("A prompt is required (provide positionally or via --prompt/-p)")
        generate_cli(
            args.repo_id,
            prompt,
            speculative=args.speculative,
            draft_layers=args.draft_layers,
            draft_model=args.draft_model,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
