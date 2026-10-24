"""Raw Qwen3.8-27B without the expert on the 300-image test (scripts/make_noexpert.py), on the VM, against the vLLM
server of q27_serve.py:
- single requests (ne16 image only, ne17 image + measurements): /content/q27/prompts300.json, sampling and acceptance
  as colab/vm/qwen38_single.py, one run, up to 3 attempts; one JSON line per generation to /content/q27/raw300.jsonl.
- agents (ne18, ne19, ne26): q27_agent.py per batch, one run, 4 OpenCode sessions at a time as q27_test300.py.
Both run at once (the server batches up to 16 sequences). Skips finished work, so it can be restarted. Zips the agent
results to /content/q27/noexp_results.zip and prints NOEXP_DONE.
"""
import base64, json, os, re, subprocess, sys, threading, time, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path("/content/q27")
RESULTS, IMAGES, RAW = ROOT / "agent", ROOT / "images300", ROOT / "raw300.jsonl"
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
            "repetition_penalty": 1.0, "max_tokens": 32768}
AGENTS, SINGLE_WORKERS, AGENT_WORKERS = ("ne26", "ne19", "ne18"), 10, 4
lock = threading.Lock()


def has_score(text: str) -> bool:
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            if 0 <= float(json.loads(match.group(0))["score"]) <= 4:
                return True
        except (ValueError, KeyError, TypeError):
            pass
    found = re.search(r"\bscore\b[^0-9\n]{0,15}(\d+(?:\.\d+)?)|\brated as\D{0,5}(\d+(?:\.\d+)?)", text, re.I)
    return bool(found) and 0 <= float(found.group(1) or found.group(2)) <= 4


def chat(prompt: str, image: Path, seed: int) -> dict:
    url = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()
    body = {"model": "qwen3.8-27b", "seed": seed, **SAMPLING,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                      {"type": "image_url", "image_url": {"url": url}}]}]}
    request = urllib.request.Request("http://localhost:8000/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=3600))


def single(job: tuple) -> None:
    index, item = job
    for attempt in (1, 2, 3):
        start, began = datetime.now(), time.time()
        try:
            response = chat(item["prompt"], IMAGES / f"{item['image_id']}.png", 100000 + index * 10 + attempt)
            choice = response["choices"][0]
            text = choice["message"].get("content") or ""
            reasoning = choice["message"].get("reasoning_content") or choice["message"].get("reasoning") or ""
            finish, usage = choice.get("finish_reason"), response.get("usage", {})
        except Exception as error:  # noqa: BLE001 - record and retry
            text, reasoning, finish, usage = f"ERROR {error}", "", "error", {}
        final = has_score(text) or attempt == 3
        record = {"type": "generation", "setup": item["setup"], "run": 1, "image_id": item["image_id"],
                  "attempt": attempt, "final": final, "seconds": round(time.time() - began, 1),
                  "prompt_tokens": usage.get("prompt_tokens"), "new_tokens": usage.get("completion_tokens"),
                  "finish_reason": finish, "start": start.isoformat(), "end": datetime.now().isoformat(),
                  "response": text, "reasoning": reasoning}
        with lock, open(RAW, "a", encoding="utf-8") as out:
            out.write(json.dumps(record) + "\n")
        if final:
            break


def singles() -> None:
    prompts = json.load(open(ROOT / "prompts300.json", encoding="utf-8"))
    done = set()
    if RAW.exists():
        for line in open(RAW, encoding="utf-8"):
            r = json.loads(line)
            if r.get("type") == "generation" and r["final"]:
                done.add((r["setup"], r["image_id"]))
    with lock, open(RAW, "a", encoding="utf-8") as out:
        out.write(json.dumps({"type": "meta", "model": "Qwen/Qwen3.8-27B", "server": "vLLM",
                              "serve": json.load(open(ROOT / "serve.json")), "sampling": SAMPLING,
                              "time": datetime.now().isoformat()}) + "\n")
    jobs = [(i, item) for i, item in enumerate(prompts) if (item["setup"], item["image_id"]) not in done]
    with ThreadPoolExecutor(SINGLE_WORKERS) as pool:
        list(pool.map(single, jobs))
    print("singles done", flush=True)


def agent(job: tuple) -> None:
    i, setup = job
    out = RESULTS / setup / "run_1"
    if (out / "scores.json").exists():
        return
    if i < AGENT_WORKERS:
        time.sleep(i * 30)  # staggered first wave
    if out.exists():
        failed = RESULTS / setup / "failed" / "run_1_interrupted"
        failed.parent.mkdir(parents=True, exist_ok=True)
        out.rename(failed)
    with open(ROOT / f"log_{setup}.txt", "a") as log:
        subprocess.run([sys.executable, str(ROOT / "q27_agent.py"), setup, "1"], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT)
    print(setup, "scores" if (out / "scores.json").exists() else "NO SCORES", flush=True)


IMAGES.mkdir(exist_ok=True)
for archive in sorted(ROOT.glob("ne18_batch_*.zip")):  # the images of every batch, for the single requests
    with zipfile.ZipFile(archive) as z:
        for m in z.namelist():
            if "/images/" in m:
                batch, name = m.split("/")[1], m.split("/")[-1]
                (IMAGES / f"{batch}_{name}").write_bytes(z.read(m))
print("images", len(list(IMAGES.glob("*.png"))), flush=True)
thread = threading.Thread(target=singles)
thread.start()
setups = [f"{v}_batch_{b:02d}" for b in range(1, 16) for v in AGENTS]
with ThreadPoolExecutor(AGENT_WORKERS) as pool:
    list(pool.map(agent, enumerate(setups)))
print("agents done", flush=True)
thread.join()
with zipfile.ZipFile(ROOT / "noexp_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in RESULTS.glob("ne*_batch_*/**/*"):
        if p.is_file():
            z.write(p, p.relative_to(RESULTS).as_posix())
    z.write(RAW, RAW.name)
print("NOEXP_DONE", flush=True)
