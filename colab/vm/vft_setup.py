import os, subprocess, time
import torch
# Stage A setup (scripts/qwen_vision_ft.py): folder, transformers with Qwen3.5 vision classes (local smoke used 5.18.0).
os.makedirs("/content/vft/out", exist_ok=True)
t = time.time()
r = subprocess.run("pip install -q 'transformers==5.18.0' 'huggingface_hub>=1.5,<2' safetensors 2>&1 | grep -v 'requires\|incompatible' | tail -3; "
                   "pip show transformers torch | grep -E '^(Name|Version)'", shell=True, capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-1000:], "seconds", round(time.time() - t), flush=True)
print("gpu", torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GiB", flush=True)
