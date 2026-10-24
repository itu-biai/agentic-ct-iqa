import subprocess, time
t = time.time()
r = subprocess.run("pip install -q -U vllm 2>&1 | tail -5; pip show vllm torch transformers 2>/dev/null | grep -E '^(Name|Version)'", shell=True, capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-2000:], "seconds", round(time.time() - t))
