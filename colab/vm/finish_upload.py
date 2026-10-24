"""Reassemble the uploaded expert_inputs.npz parts, check the MD5, and move the HF token into place."""
import glob
import hashlib
import os
import shutil
import stat

parts = sorted(glob.glob("/content/npzparts/part_*"))
with open("/content/expert_inputs.npz", "wb") as out:
    for part in parts:
        with open(part, "rb") as f:
            out.write(f.read())
with open("/content/expert_inputs.npz", "rb") as f:
    print("parts", len(parts), "md5", hashlib.md5(f.read()).hexdigest())
shutil.rmtree("/content/npzparts")
os.makedirs("/root/.cache/huggingface", exist_ok=True)
shutil.move("/content/.hf_token_upload", "/root/.cache/huggingface/token")
os.chmod("/root/.cache/huggingface/token", stat.S_IRUSR | stat.S_IWUSR)
print("token in place:", os.path.getsize("/root/.cache/huggingface/token"), "bytes")
