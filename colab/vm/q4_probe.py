import subprocess, torch, transformers, importlib.util
print("gpu", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0), "torch", torch.__version__, "transformers", transformers.__version__)
for m in ("vllm", "sglang", "fla", "causal_conv1d"):
    print(m, bool(importlib.util.find_spec(m)))
print(subprocess.run("nvidia-smi --query-gpu=memory.total,driver_version --format=csv,noheader; python --version; df -h /content | tail -1", shell=True, capture_output=True, text=True).stdout)
