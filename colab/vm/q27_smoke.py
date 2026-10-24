"""Smoke test: one single request (setup 13 prompt, img_14) and one short OpenCode agent call outside the sandboxes."""
import base64, json, os, shutil, subprocess, time, urllib.request
P = json.load(open("/content/q27/prompts.json"))
item = [p for p in P if p["setup"] == "setup11_medgemma_vision_tools" and p["image_id"] == "img_14"][0]
url = "data:image/png;base64," + base64.b64encode(open("/content/q27/images/img_14.png", "rb").read()).decode()
body = {"model": "qwen3.8-27b", "seed": 1, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 0.0, "max_tokens": 32768,
        "messages": [{"role": "user", "content": [{"type": "text", "text": item["prompt"]}, {"type": "image_url", "image_url": {"url": url}}]}]}
t = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
m = r["choices"][0]["message"]
reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
print("single:", round(time.time() - t), "s", r["usage"], r["choices"][0]["finish_reason"], "| reasoning chars", len(reasoning))
print("ANSWER:", (m.get("content") or "")[:400])
print("REASONING TAIL:", reasoning[-600:])
d = "/content/q27/smoke"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d + "/images")
shutil.copy("/content/q27/images/img_03.png", d + "/images/img_03.png")
subprocess.run("git init -q", shell=True, cwd=d)
import importlib.util
spec = importlib.util.spec_from_file_location("q4a", "/content/q27/q27_agent.py"); q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
q.write_config()
env = {**os.environ, "OPENCODE_CONFIG": str(q.CONFIG), "OPENCODE_DISABLE_CLAUDE_CODE": "1", "PWD": d}
t = time.time()
o = subprocess.run(["opencode", "run", "--standalone", "--format", "json", "-m", q.MODEL,
                    "Open images/img_03.png with your file-reading tool and describe what you see in one sentence. Then write that sentence to note.txt."],
                   cwd=d, env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=1200)
print("agent:", round(time.time() - t), "s, exit", o.returncode, o.stderr.decode()[-800:])
for line in o.stdout.decode().splitlines():
    try: e = json.loads(line)
    except ValueError: continue
    if e.get("type") == "tool_use": print("TOOL", e["part"]["tool"], json.dumps(e["part"]["state"].get("input"))[:120], e["part"]["state"].get("status"))
    elif e.get("type") == "text": print("TEXT", e["part"].get("text", "")[:300])
    elif e.get("type") not in ("step_start", "step_finish"): print("EV", e.get("type"), str(e)[:200])
print("note.txt:", open(d + "/note.txt").read()[:300] if os.path.exists(d + "/note.txt") else "MISSING")
