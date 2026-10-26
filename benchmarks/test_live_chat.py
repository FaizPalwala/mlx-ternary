#!/usr/bin/env python3
"""Live OpenAI Client Test for mlx-ternary server."""

import time
import json
import requests

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"

def test_chat(prompt: str, stream: bool = True):
    print("=" * 70)
    print(f"QUERY: \"{prompt}\"")
    print(f"STREAMING: {stream}")
    print("=" * 70)

    payload = {
        "messages": [
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 64,
        "stream": stream
    }

    t0 = time.perf_counter()
    resp = requests.post(SERVER_URL, json=payload, stream=stream)
    
    if not resp.ok:
        print(f"Error {resp.status_code}: {resp.text}")
        return

    if not stream:
        data = resp.json()
        total_time = time.perf_counter() - t0
        content = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("completion_tokens", len(content.split()))
        print(f"\nResponse:\n{content}\n")
        print(f"Total Time: {total_time:.2f}s | Speed: {tokens / total_time:.1f} tok/s")
    else:
        print("\nStreaming Response:")
        first_chunk_time = None
        full_text = []
        token_count = 0

        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                raw_json = line[6:].strip()
                if raw_json == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw_json)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter() - t0
                        print(delta, end="", flush=True)
                        full_text.append(delta)
                        token_count += 1
                except Exception:
                    pass

        total_time = time.perf_counter() - t0
        print("\n" + "-" * 70)
        ttft = first_chunk_time if first_chunk_time else total_time
        gen_time = total_time - ttft
        tps = token_count / gen_time if gen_time > 0 else 0
        print(f"\n[Live HTTP Metrics]:")
        print(f"  • TTFT (Time To First Token): {ttft * 1000:.1f} ms")
        print(f"  • Live Streaming Speed:       {tps:.1f} tokens/s")
        print(f"  • Total Tokens Received:      {token_count}")
        print(f"  • Total Request Latency:      {total_time:.2f} s")

if __name__ == "__main__":
    test_chat("Explain in two sentences what a ternary weight is in AI.", stream=True)
