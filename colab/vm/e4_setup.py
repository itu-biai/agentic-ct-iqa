import os, zipfile, json, subprocess, time
zipfile.ZipFile("/content/e4/images.zip").extractall("/content/e4/images")
zipfile.ZipFile("/content/e4/sandboxes.zip").extractall("/content/e4/sb")
print("images", len(os.listdir("/content/e4/images")), "examples", len(os.listdir("/content/e4/images/examples")),
      "prompts", len(json.load(open("/content/e4/prompts.json"))), "sb", sorted(os.listdir("/content/e4/sb")),
      len(os.listdir("/content/e4/sb/setup2b_vision_tools_examples/examples")), flush=True)
t = time.time()
# vLLM version of the Qwen3.5-4B and Qwen3.8-27B runs; huggingface_hub < 2 (2.x breaks transformers 5.17).
r = subprocess.run("pip install -q vllm==0.31.0 2>&1 | grep -v 'requires\\|incompatible' | tail -3; "
                   "pip install -q 'huggingface_hub>=1.5,<2' 2>&1 | tail -1; "
                   "pip show vllm transformers huggingface_hub | grep -E '^(Name|Version)'; "
                   "npm install -g @opencode/cli@2.0.26 2>&1 | tail -2; which opencode && opencode --version; "
                   "hf download empero-ai/Qwen3.8-4B-Distill 2>&1 | tail -1; nvidia-smi --query-gpu=name,memory.total --format=csv",
                   shell=True, capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-1500:], "seconds", round(time.time() - t), flush=True)
