import os, shutil, zipfile, json, subprocess, time
import torch
# Agent-only job (setup 26): just the setup2b sandbox; same install, weights choice and download as q27_setup.py.
zipfile.ZipFile("/content/q27/sandboxes.zip").extractall("/content/q27/sb")
print("sb", sorted(os.listdir("/content/q27/sb")), len(os.listdir("/content/q27/sb/setup2b_vision_tools_examples/examples")))
t = time.time()
r = subprocess.run("pip install -q -U vllm 'transformers>=5.8' 2>&1 | grep -v 'requires\\|incompatible' | tail -3; "
                   "pip show vllm transformers | grep -E '^(Name|Version)'; "
                   "npm install -g @opencode/cli@2.0.26 2>&1 | tail -2; which opencode && opencode --version; "
                   "pip install -q 'huggingface_hub>=1.5,<2' hf_transfer 2>&1 | tail -1", shell=True, capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-1500:], "seconds", round(time.time() - t), flush=True)
# bf16 (the released checkpoint) when the GPU and the disk hold it, otherwise Qwen's official FP8 checkpoint.
memory = torch.cuda.get_device_properties(0).total_memory / 2**30
free = shutil.disk_usage("/root").free / 2**30
weights = "Qwen/Qwen3.8-27B" if memory > 70 and free > 66 else "Qwen/Qwen3.8-27B-FP8"
open("/content/q27/weights.txt", "w").write(weights)
print("gpu", torch.cuda.get_device_name(0), round(memory, 1), "GiB, disk free", round(free), "GiB ->", weights, flush=True)
t = time.time()
r = subprocess.run(f"HF_HUB_ENABLE_HF_TRANSFER=1 hf download {weights} 2>&1 | tail -2; df -h /root | tail -1",
                   shell=True, capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-1000:], "seconds", round(time.time() - t))
