"""Verify Ollama is serving a model through the Jetson CUDA backend."""

import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

host = os.environ.get("OLLAMA_TEST_HOST", "http://127.0.0.1:11434")
if "://" not in host:
    host = "http://" + host
if host.startswith("http://0.0.0.0"):
    host = host.replace("http://0.0.0.0", "http://127.0.0.1", 1)
model = os.environ.get("OLLAMA_TEST_MODEL", "gemma3:1b")

def get(path):
    with urllib.request.urlopen(host.rstrip("/") + path, timeout=15) as response:
        return json.load(response)

for attempt in range(60):
    try:
        tags = get("/api/tags")
        break
    except (urllib.error.URLError, TimeoutError):
        if attempt == 59:
            raise
        time.sleep(1)
models = {item.get("name") for item in tags.get("models", [])}
if model not in models:
    raise SystemExit(f"Model {model!r} is not installed; run: ollama pull {model}")

payload = json.dumps({
    "model": model,
    "prompt": "Reply with one short greeting.",
    "stream": False,
}).encode()
request = urllib.request.Request(
    host.rstrip("/") + "/api/generate",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=120) as response:
    result = json.load(response)
if not result.get("response", "").strip():
    raise SystemExit("Ollama returned no generated text")

ps = subprocess.run(["ollama", "ps"], check=True, capture_output=True, text=True).stdout
rows = [line for line in ps.splitlines() if model in line]
if not rows or "100% GPU" not in rows[0]:
    print(ps, file=sys.stderr)
    raise SystemExit(f"{model} is not fully GPU-offloaded")

print(f"PASS Ollama {model}: 100% GPU offload")
