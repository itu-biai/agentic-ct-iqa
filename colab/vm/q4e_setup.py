# Agent-only job (setup 28): the setup2b sandbox, the Qwen3.5-4B run's vLLM version (0.31.0) and OpenCode 2.0.26.
import os, zipfile, subprocess, time
zipfile.ZipFile("/content/q4/sandboxes.zip").extractall("/content/q4/sb")
print("sb", sorted(os.listdir("/content/q4/sb")), len(os.listdir("/content/q4/sb/setup2b_vision_tools_examples/examples")), flush=True)
t = time.time()
r = subprocess.run("pip install -q vllm==0.31.0 2>&1 | grep -v 'requires\|incompatible' | tail -3; "
                   "pip install -q 'huggingface_hub>=1.5,<2' 2>&1 | tail -1; "
                   "pip show vllm transformers | grep -E '^(Name|Version)'; "
                   "npm install -g @opencode/cli@2.0.26 2>&1 | tail -2; which opencode && opencode --version; "
                   "hf download Qwen/Qwen3.5-4B 2>&1 | tail -1; nvidia-smi --query-gpu=name,memory.total --format=csv",
                   shell=True, capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-1500:], "seconds", round(time.time() - t), flush=True)
