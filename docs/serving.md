# Serving & Frontend Integration

`mlx-ternary` includes a high-performance, OpenAI-compatible HTTP server built directly on top of `mlx-lm` and FastAPI.

---

## 1. Starting the Server

```bash
# Basic startup
mlx_ternary serve prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --port 8080

# Speculative Decoding (runs 27B model with a 1.5B draft model)
mlx_ternary serve prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --speculate --port 8080

# 4-bit Quantized KV Cache (for deep 32k-64k agentic workflows)
mlx_ternary serve prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --kv-bits 4 --port 8080
```

---

## 2. LM Studio Integration

You can connect **LM Studio** directly to `mlx-ternary`:

1. Open **LM Studio**.
2. Navigate to **Developer / Local Server** or the settings panel.
3. Add a custom OpenAI-compatible server endpoint:
   - **Base URL:** `http://127.0.0.1:8080/v1`
   - **Model Name:** `Ternary-Bonsai-2-27B-mlx-2bit`
4. Start chatting! The inference requests will be routed directly to your local Apple Silicon Metal kernels.

---

## 3. Direct API Usage (cURL & Python)

### cURL

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Ternary-Bonsai-2-27B-mlx-2bit",
    "messages": [
      {"role": "user", "content": "Explain Rayleigh scattering in one sentence."}
    ],
    "temperature": 0.7
  }'
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="none"  # Local server does not require authentication
)

response = client.chat.completions.create(
    model="Ternary-Bonsai-2-27B-mlx-2bit",
    messages=[
        {"role": "user", "content": "Write a quick Python script to calculate Fibonacci numbers."}
    ]
)

print(response.choices[0].message.content)
```
