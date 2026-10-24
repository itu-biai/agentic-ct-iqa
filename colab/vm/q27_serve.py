"""Starts a vLLM OpenAI-compatible server for Qwen3.8-27B on the VM (background) and waits until it answers.

Weights as chosen by q27_setup.py: bf16 (the released checkpoint) when GPU and disk hold it, else Qwen's FP8 one.
Context: the model's native 262,144 tokens if the KV cache holds them, else the largest length vLLM reports it can
hold. The chosen settings go to /content/q27/serve.json (read by q27_agent.py and the single-request runner).
"""
import json, re, subprocess, time, urllib.request
import torch

memory = torch.cuda.get_device_properties(0).total_memory / 2**30
gpu = torch.cuda.get_device_name(0)
weights = open("/content/q27/weights.txt").read().strip()  # chosen by q27_setup.py
dtype = "fp8" if weights.endswith("FP8") else "bf16"
# Images per request: an OpenCode session sends its whole history, so every image it has opened counts. 64 was
# enough for setups 18-28 (and the 300-image test); setup 30's agent opened more (examples, references, views) and
# was refused at 64 ("At most 64 image(s) may be provided in one prompt"). Setup 29 lost one attempt to the same.
IMAGES = 256


def start(length: int) -> str:
    cmd = (f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup vllm serve {weights} --port 8000 "
           f"--max-model-len {length} --gpu-memory-utilization 0.92 --max-num-seqs 16 "
           f"--limit-mm-per-prompt '{{\"image\": {IMAGES}}}' --reasoning-parser qwen3 --enable-auto-tool-choice "
           "--tool-call-parser qwen3_coder --served-model-name qwen3.8-27b > /content/vllm.log 2>&1 &")
    subprocess.run("pkill -f 'vllm serv[e]'; sleep 5", shell=True)
    subprocess.run(cmd, shell=True)
    for i in range(240):
        time.sleep(15)
        try:
            urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
            print("READY after", (i + 1) * 15, "s", flush=True)
            return "ready"
        except Exception:
            if subprocess.run("pgrep -f 'vllm serv[e]'", shell=True, capture_output=True).returncode:
                return open("/content/vllm.log", encoding="utf-8", errors="replace").read()
    return "timeout"


print("gpu", gpu, round(memory, 1), "GiB ->", weights, flush=True)
length = 262144
for _ in range(3):
    state = start(length)
    if state == "ready":
        break
    # vLLM names the longest context the KV cache can hold when the requested one does not fit.
    found = re.search(r"estimated maximum model length is (\d+)", state)
    if not found:
        print("SERVER DIED\n", state[-3000:])
        raise SystemExit(1)
    length = int(found.group(1)) // 1024 * 1024
    print("context does not fit; retrying with", length, flush=True)
json.dump({"weights": weights, "max_model_len": length, "gpu": gpu,
           "server": f"vLLM on Colab {gpu}, {dtype} weights"}, open("/content/q27/serve.json", "w"), indent=1)
print(json.load(open("/content/q27/serve.json")))
print(subprocess.run("grep -iE 'error|KV cache|maximum concurrency|generation config|default sampling' /content/vllm.log | tail -15",
                     shell=True, capture_output=True, text=True).stdout)
