"""Minimal client for NVIDIA's hosted vision-language models (build.nvidia.com, OpenAI-compatible endpoint).

The API key is taken from the NVIDIA_API_KEY environment variable or, failing that, from the credential the user
stored in OpenCode (`opencode auth export`). It is kept in memory only: never printed, logged or written to disk.
"""
import base64
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import requests

URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def api_key() -> str:
    if os.environ.get("NVIDIA_API_KEY"):
        return os.environ["NVIDIA_API_KEY"]
    exe = shutil.which("opencode") or shutil.which("opencode.cmd")
    out = subprocess.run([exe, "auth", "export"], capture_output=True, text=True, check=True).stdout
    for entry in json.loads(out):
        if "nvidia" in (entry.get("integrationID", "") + entry.get("label", "")).lower():
            return entry["value"]["key"]
    raise SystemExit("No NVIDIA key: set NVIDIA_API_KEY or connect NVIDIA in OpenCode (/connect)")


def image_part(path: Path) -> dict:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def chat(key: str, model: str, text: str, image: Path | None = None, max_tokens: int = 400,
         temperature: float | None = None, retries: int = 4) -> dict:
    """One request; returns the parsed response JSON. Retries on rate limits and server errors."""
    content = [{"type": "text", "text": text}] + ([image_part(image)] if image else [])
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens}
    if temperature is not None:
        body["temperature"] = temperature
    for attempt in range(retries + 1):
        response = requests.post(URL, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
                                 json=body, timeout=180)
        if response.status_code == 200:
            return response.json()
        if response.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            time.sleep(5 * 2 ** attempt)
            continue
        raise RuntimeError(f"{model}: HTTP {response.status_code}: {response.text[:300]}")
    raise RuntimeError("unreachable")
