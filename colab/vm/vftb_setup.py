import os, shutil, subprocess, time
import torch
# Stage B setup (colab/vm/vftb_train.py): libraries (transformers as the local smoke, FLA kernels for the linear-attention
# layers) and the FP8 checkpoint shards of Qwen3.8-27B (no MTP) in /content/hf.
os.makedirs("/content/vft", exist_ok=True)
t = time.time()
r = subprocess.run("pip install -q 'transformers==5.18.0' 'huggingface_hub>=1.5,<2' safetensors flash-linear-attention 2>&1 "
                   "| grep -v 'requires\\|incompatible' | tail -3; pip show transformers torch flash-linear-attention | grep -E '^(Name|Version)'",
                   shell=True, capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-1000:], "seconds", round(time.time() - t), flush=True)
from huggingface_hub import snapshot_download
t = time.time()
snapshot_download("Qwen/Qwen3.8-27B-FP8", local_dir="/content/hf", allow_patterns=["layers-*.safetensors", "outside.safetensors"],
                  max_workers=8)
size = sum(f.stat().st_size for f in __import__("pathlib").Path("/content/hf").glob("*.safetensors")) / 2**30
print(f"checkpoint {size:.1f} GiB in {(time.time() - t) / 60:.1f} min; disk free {shutil.disk_usage('/content').free / 2**30:.0f} GiB", flush=True)
r = subprocess.run("python -c 'import fla; from fla.ops.gated_delta_rule import chunk_gated_delta_rule; print(\"fla ok\", fla.__version__)'",
                   shell=True, capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-500:], flush=True)
print("gpu", torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GiB", flush=True)
print("SETUP_DONE", flush=True)
