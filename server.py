import sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import PreTrainedTokenizerFast

# We'll build a very simple dummy wrapper around mlx_lm.server
# Wait, let's just patch mlx_lm.utils.load
import mlx_lm.utils
import mlx_lm.server

model_path = Path("models/Ternary-Bonsai-2-27B-mlx-2bit")
sys.path.insert(0, str(model_path / "runtime"))
from artifact import load_model

def custom_load(*args, **kwargs):
    print("Intercepting mlx_lm.load -> Loading Bonsai Ternary Model...")
    model, config = load_model(model_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(model_path / "tokenizer.json"))
    tokenizer.chat_template = (model_path / "chat_template.jinja").read_text()
    return model, tokenizer

mlx_lm.utils.load = custom_load
mlx_lm.server.load = custom_load
mlx_lm.load = custom_load

if __name__ == "__main__":
    sys.argv = ["mlx_lm.server", "--model", str(model_path)]
    print("Starting MLX-LM OpenAI Compatible Server...")
    mlx_lm.server.main()
