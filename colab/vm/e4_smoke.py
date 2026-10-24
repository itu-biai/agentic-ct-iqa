"""Smoke test: one setup 24 request (10 examples + img_14) and one short OpenCode agent call outside the sandboxes."""
import importlib.util, json, os, shutil, subprocess, sys, time
source = open("/content/e4/empero_single.py").read().split("prompts = json.load")[0]  # helpers only, no run
es = type(sys)("es"); exec(compile(source, "empero_single.py", "exec"), es.__dict__)
P = json.load(open("/content/e4/prompts.json"))
item = [p for p in P if p["setup"] == "setup24_empero_4b_vision_tools_examples" and p["image_id"] == "img_14"][0]
t = time.time()
r = es.chat(item, 1)
m = r["choices"][0]["message"]
reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
print("single:", round(time.time() - t), "s", r["usage"], r["choices"][0]["finish_reason"], "| reasoning chars", len(reasoning))
print("ANSWER:", (m.get("content") or "")[:400])
print("REASONING HEAD:", reasoning[:500])
print("REASONING TAIL:", reasoning[-600:])
d = "/content/e4/smoke"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d + "/images")
shutil.copy("/content/e4/images/img_03.png", d + "/images/img_03.png")
subprocess.run("git init -q", shell=True, cwd=d)
spec = importlib.util.spec_from_file_location("e4a", "/content/e4/e4_agent.py"); q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
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
