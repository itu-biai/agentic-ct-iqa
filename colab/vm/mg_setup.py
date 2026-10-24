import os, shutil, stat, zipfile, json, subprocess
shutil.move("/content/.hf_token_upload", "/root/.cache/huggingface/token")
os.chmod("/root/.cache/huggingface/token", stat.S_IRUSR | stat.S_IWUSR)
zipfile.ZipFile("/content/mg/images.zip").extractall("/content/mg/images")
print("images", len(os.listdir("/content/mg/images")), "prompts", len(json.load(open("/content/mg/prompts.json"))))
import torch, transformers; print("gpu", torch.cuda.get_device_name(0), "transformers", transformers.__version__)
from transformers import AutoConfig; print("config", AutoConfig.from_pretrained("google/medgemma-1.5-4b-it").model_type)
