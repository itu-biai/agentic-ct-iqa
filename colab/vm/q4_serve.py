"""Starts a vLLM OpenAI-compatible server for Qwen3.5-4B on the VM (background) and waits until it answers."""
import subprocess, time, urllib.request
cmd = ("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup vllm serve Qwen/Qwen3.5-4B --port 8000 --dtype half --max-model-len 55296 --gpu-memory-utilization 0.88 --max-num-seqs 8 --max-num-batched-tokens 4096 "
       "--limit-mm-per-prompt '{\"image\": 64}' --reasoning-parser qwen3 --enable-auto-tool-choice "
       "--tool-call-parser qwen3_coder --served-model-name qwen3.5-4b > /content/vllm.log 2>&1 &")
subprocess.run("pkill -f 'vllm serve'; sleep 3", shell=True)
subprocess.run(cmd, shell=True)
for i in range(120):
    time.sleep(10)
    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
        print("READY after", (i + 1) * 10, "s"); break
    except Exception:
        if subprocess.run("pgrep -f 'vllm serve'", shell=True, capture_output=True).returncode:
            print("SERVER DIED"); break
print(subprocess.run("grep -iE 'error|warn|KV cache|maximum concurrency|attention backend|dtype' /content/vllm.log | tail -25", shell=True, capture_output=True, text=True).stdout)
