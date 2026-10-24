"""Empero's Qwen3.8-4B-Distill on the sandbox prompts, one independent request per image (the protocol of setups 12/13),
on the Colab VM.

Expects the vLLM server from e4_serve.py on localhost:8000, /content/e4/prompts.json (scripts/sandbox_empero.py
prepare), /content/e4/images/ and, for the examples setup, /content/e4/images/examples/ with examples.json. A prompt
with "examples" gets the 10 rated training slices attached before the image to rate, each after a line with its
score. Sampling: the model card's settings, fixed seed per run, image and attempt; max_tokens as for Qwen3.5-4B.
Appends one JSON line per generation to /content/e4/raw.jsonl and skips work already in it. Prints EMPERO_SINGLE_DONE
at the end.
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

JOB = "/content/e4"
OUT = f"{JOB}/raw.jsonl"
RUNS = (1, 2, 3)
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
            "repetition_penalty": 1.0, "max_tokens": 32768}
EXAMPLES = json.load(open(f"{JOB}/examples.json"))["scores"]
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


def image_part(path: str) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()}}


def content(item: dict) -> list:
    parts = [{"type": "text", "text": item["prompt"]}]
    if item.get("examples"):
        for example in EXAMPLES:
            parts += [{"type": "text", "text": f"Example {example['image_id']}, radiologist score {example['score']}:"},
                      image_part(f"{JOB}/images/examples/{example['image_id']}.png")]
        parts.append({"type": "text", "text": "Image to rate:"})
    return parts + [image_part(f"{JOB}/images/{item['image_id']}.png")]


def chat(item: dict, seed: int) -> dict:
    body = {"model": "empero-4b", "seed": seed, **SAMPLING, "messages": [{"role": "user", "content": content(item)}]}
    request = urllib.request.Request("http://localhost:8000/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=3600))


def work(job: tuple) -> None:
    run, index, item = job
    for attempt in (1, 2, 3):
        start, began = datetime.now(), time.time()
        try:
            response = chat(item, run * 100000 + index * 10 + attempt)
            choice = response["choices"][0]
            text = choice["message"].get("content") or ""
            reasoning = choice["message"].get("reasoning_content") or choice["message"].get("reasoning") or ""
            finish, usage = choice.get("finish_reason"), response.get("usage", {})
        except Exception as error:  # noqa: BLE001 - record and retry
            text, reasoning, finish, usage = f"ERROR {error}", "", "error", {}
        final = has_score(text) or attempt == 3
        record = {"type": "generation", "setup": item["setup"], "run": run, "image_id": item["image_id"],
                  "attempt": attempt, "final": final, "seconds": round(time.time() - began, 1),
                  "prompt_tokens": usage.get("prompt_tokens"), "new_tokens": usage.get("completion_tokens"),
                  "finish_reason": finish, "start": start.isoformat(), "end": datetime.now().isoformat(),
                  "response": text, "reasoning": reasoning}
        with lock, open(OUT, "a", encoding="utf-8") as out:
            out.write(json.dumps(record) + "\n")
        if final:
            break
    print(run, item["setup"], item["image_id"], attempt, finish, flush=True)


prompts = json.load(open(f"{JOB}/prompts.json", encoding="utf-8"))
done = set()
if os.path.exists(OUT):
    for line in open(OUT, encoding="utf-8"):
        r = json.loads(line)
        if r.get("type") == "generation" and r["final"]:
            done.add((r["setup"], r["run"], r["image_id"]))
models = json.load(urllib.request.urlopen("http://localhost:8000/v1/models"))
with open(OUT, "a", encoding="utf-8") as out:
    out.write(json.dumps({"type": "meta", "model": "empero-ai/Qwen3.8-4B-Distill", "server": "vLLM",
                          "models": models["data"], "sampling": SAMPLING, "examples": EXAMPLES,
                          "time": datetime.now().isoformat()}) + "\n")
jobs = [(run, index, item) for run in RUNS for index, item in enumerate(prompts)
        if (item["setup"], run, item["image_id"]) not in done]
with ThreadPoolExecutor(8) as pool:  # requests are independent; batching only changes speed
    list(pool.map(work, jobs))
print("EMPERO_SINGLE_DONE", flush=True)
