"""Qwen3.5-4B on the sandbox prompts, one independent request per image (the MedGemma/Llama protocol), on the Colab VM.

Expects the vLLM server from q4_serve.py on localhost:8000, /content/q4/prompts.json (scripts/sandbox_medgemma.py
prepare; same prompts as setups 10/11) and /content/q4/images/. Sampling: Qwen's recommended thinking-mode settings
for general tasks (the model ships no generation config), fixed seed per run, image and attempt. Appends one JSON line
per generation to /content/q4/raw.jsonl and skips work already in it. Prints QWEN35_SINGLE_DONE at the end.
"""
import base64
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Lock

JOB = "/content/q4"
OUT = f"{JOB}/raw.jsonl"
RUNS = (1, 2, 3)
SETUP = {"setup10_medgemma_vision_only": "setup12_qwen35_4b_vision_only",
         "setup11_medgemma_vision_tools": "setup13_qwen35_4b_vision_tools"}
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
            "repetition_penalty": 1.0, "max_tokens": 32768}
lock = Lock()


def has_score(text: str) -> bool:
    """Same acceptance rule as scripts/sandbox_llama.parse: a JSON score or a stated score, in [0, 4]."""
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            if 0 <= float(json.loads(match.group(0))["score"]) <= 4:
                return True
        except (ValueError, KeyError, TypeError):
            pass
    found = re.search(r"\bscore\b[^0-9\n]{0,15}(\d+(?:\.\d+)?)|\brated as\D{0,5}(\d+(?:\.\d+)?)", text, re.I)
    return bool(found) and 0 <= float(found.group(1) or found.group(2)) <= 4


def chat(prompt: str, image: str, seed: int) -> dict:
    url = "data:image/png;base64," + base64.b64encode(open(image, "rb").read()).decode()
    body = {"model": "qwen3.5-4b", "seed": seed, **SAMPLING,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                      {"type": "image_url", "image_url": {"url": url}}]}]}
    request = urllib.request.Request("http://localhost:8000/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=3600))


def work(job: tuple) -> None:
    run, index, item = job
    setup = SETUP[item["setup"]]
    for attempt in (1, 2, 3):
        start, began = datetime.now(), time.time()
        try:
            response = chat(item["prompt"], f"{JOB}/images/{item['image_id']}.png", run * 100000 + index * 10 + attempt)
            choice = response["choices"][0]
            text = choice["message"].get("content") or ""
            reasoning = choice["message"].get("reasoning_content") or choice["message"].get("reasoning") or ""
            finish, usage = choice.get("finish_reason"), response.get("usage", {})
        except Exception as error:  # noqa: BLE001 - record and retry
            text, reasoning, finish, usage = f"ERROR {error}", "", "error", {}
        final = has_score(text) or attempt == 3
        record = {"type": "generation", "setup": setup, "run": run, "image_id": item["image_id"], "attempt": attempt,
                  "final": final, "seconds": round(time.time() - began, 1), "prompt_tokens": usage.get("prompt_tokens"),
                  "new_tokens": usage.get("completion_tokens"), "finish_reason": finish, "start": start.isoformat(),
                  "end": datetime.now().isoformat(), "response": text, "reasoning": reasoning}
        with lock, open(OUT, "a", encoding="utf-8") as out:
            out.write(json.dumps(record) + "\n")
        if final:
            break
    print(run, setup, item["image_id"], attempt, finish, flush=True)


prompts = json.load(open(f"{JOB}/prompts.json", encoding="utf-8"))
done = set()
if os.path.exists(OUT):
    for line in open(OUT, encoding="utf-8"):
        r = json.loads(line)
        if r.get("type") == "generation" and r["final"]:
            done.add((r["setup"], r["run"], r["image_id"]))
models = json.load(urllib.request.urlopen("http://localhost:8000/v1/models"))
with open(OUT, "a", encoding="utf-8") as out:
    out.write(json.dumps({"type": "meta", "model": "Qwen/Qwen3.5-4B", "server": "vLLM", "models": models["data"],
                          "sampling": SAMPLING, "time": datetime.now().isoformat()}) + "\n")
jobs = [(run, index, item) for run in RUNS for index, item in enumerate(prompts)
        if (SETUP[item["setup"]], run, item["image_id"]) not in done]
with ThreadPoolExecutor(8) as pool:  # requests are independent; batching only changes speed
    list(pool.map(work, jobs))
print("QWEN35_SINGLE_DONE", flush=True)
