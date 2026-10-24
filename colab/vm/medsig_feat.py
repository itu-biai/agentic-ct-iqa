"""E5 (issue #329): frozen MedSigLIP features for the no-reference SSIM / PSNR control (scripts/nr_eval.py). Tolga's
fold-1 checkpoint (itu-biai/medsiglip-ldct-iqa ec00ad1, the shared Drive folder, as colab/vm/medsig_pool.py) with his
eval preprocessing (uint8 RGB -> Resize 448 -> processor without resize); the features are what his heads read: the
FiLM-modulated patch tokens pooled globally (mean), locally (4 means) and as texture (2 maxima), concatenated
(7 x 1152). Images: nr.npz (300 lab train slices x 13) then lab329.npz (192 lab test slices x 13), the order of
nr_feat.npz. Writes /content/tolga/medsig_feat.npz (ids, feat fp16, pred); prints MEDSIG_FEAT_DONE. Run as a plain
python process (nohup), not in the kernel.
"""
import glob
import os
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image

W = "/content/tolga"
os.chdir(W)
os.environ["HF_TOKEN"] = open("/content/.hf_token_upload").read().strip()  # gated google/medsiglip-448; never printed
t = time.time()
if not os.path.isdir("ckpt"):
    subprocess.run("pip install -q gdown sentencepiece tifffile imageio pydicom 2>&1 | tail -1; python -m gdown -q --folder "
                   "https://drive.google.com/drive/folders/1ezXhLMT6kA0JV_vftoBYebm7mOOu8rX5 -O ckpt", shell=True)
sys.path.insert(0, os.path.join(W, "repo"))
from config import CONFIG  # noqa: E402
from data import build_processor, build_transforms, to_uint8_rgb  # noqa: E402
from model import build_model, load_checkpoint, resolve_inference_config  # noqa: E402

ckpt = sorted(glob.glob(os.path.join(W, "ckpt", "*.pth")))[0]
cfg = resolve_inference_config(CONFIG, os.path.dirname(ckpt), None)
processor = build_processor(cfg["model_name"])
_, transform = build_transforms(cfg["image_size"])
model = build_model(cfg, processor)
load_checkpoint(model, ckpt)
model.to("cuda").eval()
captured = {}
for name in ("global_pool", "local_pool", "texture_pool"):
    getattr(model, name).register_forward_hook(lambda m, i, o, name=name: captured.__setitem__(name, o.flatten(1)))
nr, lab = np.load("/content/vft/nr.npz", allow_pickle=True), np.load("/content/vft/lab329.npz")
ids = np.concatenate([nr["ids"].astype(str), lab["ids"].astype(str)])
images = np.concatenate([nr["images"], lab["images"]])
feats, preds = [], []
with torch.inference_mode():
    for s in range(0, len(images), 32):
        batch = [transform(Image.fromarray(to_uint8_rgb(im))) for im in images[s:s + 32]]
        inputs = processor(images=batch, return_tensors="pt", do_resize=False, do_center_crop=False)
        preds.append(model({"pixel_values": inputs["pixel_values"].to("cuda")}).float().cpu().numpy())
        feats.append(torch.cat([captured[n] for n in ("global_pool", "local_pool", "texture_pool")], 1).float().cpu().numpy())
np.savez(os.path.join(W, "medsig_feat.npz"), ids=ids, feat=np.concatenate(feats).astype(np.float16), pred=np.concatenate(preds),
         ckpt=os.path.basename(ckpt))
print("MEDSIG_FEAT_DONE", os.path.basename(ckpt), len(ids), round(time.time() - t), "s", flush=True)
