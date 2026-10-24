import subprocess, time
# L4 setup for colab/vm/enc329.py: the transformers of Stage B and only the encoder file of the 27B checkpoint.
t = time.time()
r = subprocess.run("pip install -q 'transformers==5.18.0' 'huggingface_hub>=1.5,<2' safetensors timm 2>&1 | tail -1; "
                   "mkdir -p /content/hf /content/vft/cnn /content/vft/i329; "
                   "hf download Qwen/Qwen3.8-27B-FP8 outside.safetensors --local-dir /content/hf 2>&1 | tail -1; ls -la /content/hf",
                   shell=True, capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-400:], "seconds", round(time.time() - t), "SETUP_DONE", flush=True)
