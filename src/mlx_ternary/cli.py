import argparse
from pathlib import Path
import mlx.core as mx
import time
import sys
import subprocess

from .quantize import pack_interleaved

CACHE_DIR = Path.home() / ".cache" / "mlx_ternary"
DEFAULT_DRAFT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

def download_and_transcode(repo_id: str):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download models. Install via: pip install huggingface_hub"
        )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Checking for {repo_id}...")
    model_dir = CACHE_DIR / repo_id.replace("/", "--")
    
    if not model_dir.exists():
        print(f"Downloading {repo_id} from Hugging Face...")
        snapshot_download(repo_id=repo_id, local_dir=str(model_dir))
    
    interleaved_path = model_dir / "model_interleaved.safetensors"
    
    if interleaved_path.exists():
        print("Packed weights found. Ready for inference.")
        return model_dir
        
    print("Packed weights not found. JIT Transcoding...")
    original_safetensors = model_dir / "model.safetensors"
    if not original_safetensors.exists():
        raise FileNotFoundError(f"Could not find model.safetensors in {model_dir}")
        
    weights = mx.load(str(original_safetensors))
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
    print(f"Found {len(prefixes)} layers to transcode.")
    
    for i, p in enumerate(prefixes):
        print(f"[{i+1}/{len(prefixes)}] Transcoding {p} ...\r", end="", flush=True)
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
            
    print(f"\nSaving transcoded model to {interleaved_path} (took {time.time()-start:.2f}s)...")
    mx.save_safetensors(str(interleaved_path), new_weights)
    print("Transcoding complete!")
    return model_dir

def serve(repo_id: str, speculate: bool = False, draft_model: str = None, host: str = "127.0.0.1", port: int = 8080):
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

    run_server(model_dir=model_dir, host=host, port=port, draft_model=draft)

def main():
    parser = argparse.ArgumentParser(prog="mlx_ternary")
    subparsers = parser.add_subparsers(dest="command")
    
    serve_parser = subparsers.add_parser("serve", help="Serve a model")
    serve_parser.add_argument("repo_id", help="Hugging Face Repo ID")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    serve_parser.add_argument("--speculate", action="store_true", help="Enable speculative decoding with a default draft model.")
    serve_parser.add_argument("--draft", dest="draft_model", type=str, help="Specify a custom draft model for speculative decoding.")
    
    pack_parser = subparsers.add_parser("pack", help="Explicitly pack a model")
    pack_parser.add_argument("repo_id", help="Hugging Face Repo ID")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        serve(args.repo_id, args.speculate, args.draft_model, host=args.host, port=args.port)
    elif args.command == "pack":
        download_and_transcode(args.repo_id)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
