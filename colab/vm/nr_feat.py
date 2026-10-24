"""Frozen Stage B features for no-reference PSNR / SSIM prediction (scripts/nr_eval.py): the fold 1 model of the Stage B
rerun (ROOT/weights_fold_1.pt), reloaded as in colab/vm/teacher.py, gives the last-token hidden state (centre view,
fp16) and its llm and vision predictions for the 300 lab train slices (ROOT/nr.npz, scripts/nr_pool.py) and the lab
IQA test split (set "methods" of ROOT/cross.npz). Writes ROOT/nr_feat.npz (ids, feat, llm, vision); prints NR_DONE.
No sys.argv (`colab exec`); VFT_ROOT and VFT_SMOKE=1 for a local check.
"""
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(os.environ.get("VFT_ROOT", "/content/vft"))
vft = {"__name__": "vftb_train"}
exec(compile((ROOT / "vftb_train.py").read_text(), str(ROOT / "vftb_train.py"), "exec"), vft)
DEVICE, K = vft["DEVICE"], 1


def main() -> None:
    from transformers import AutoImageProcessor, AutoProcessor
    z = np.load(ROOT / "train.npz")
    nr, cross = np.load(ROOT / "nr.npz", allow_pickle=True), np.load(ROOT / "cross.npz", allow_pickle=True)
    keep = cross["set"] == "methods"
    ids = np.concatenate([nr["ids"], cross["ids"][keep]])
    images = np.concatenate([nr["images"], cross["images"][keep]])
    tools = np.concatenate([nr["tools"], cross["tools"][keep]])
    if vft["SMOKE"]:
        ids, images, tools = ids[:3], images[:3], tools[:3]
    config = vft["configs"]()
    image_processor = AutoImageProcessor.from_pretrained(ROOT)
    processor = AutoProcessor.from_pretrained(vft["REPO_ID"])
    model, _ = vft["language_model"](config)
    loras = {name: m for name, m in model.language_model.named_modules() if isinstance(m, vft["LoRA"])}
    start = time.time()
    w = torch.load(ROOT / f"weights_fold_{K}.pt", map_location="cpu")
    assert set(w["lora"]) == set(loras), "LoRA module names differ"
    enc = vft["encoder"](config)
    enc.load_state_dict({key: v.float() for key, v in w["encoder"].items()})
    enc = enc.to(DEVICE).eval()
    for name, m in loras.items():
        m.a.data.copy_(w["lora"][name]["a"].to(m.a.dtype))
        m.b.data.copy_(w["lora"][name]["b"].to(m.b.dtype))
    llm_head = nn.Linear(model.config.text_config.hidden_size, 1).to(DEVICE)
    llm_head.load_state_dict(w["llm_head"])
    head = vft["Head"](enc.config.hidden_size, z["tools"].shape[1]).to(DEVICE)
    head.load_state_dict(w["vision_head"])
    model.visual = enc
    tr = np.nonzero(z["fold"] != K)[0]
    mean, std = z["mos"][tr].mean(), z["mos"][tr].std()
    feat = vft["llm_predict"](model, enc, llm_head, processor, images, tools, features=True)
    print(f"features done, {(time.time() - start) / 60:.1f} min", flush=True)
    with torch.inference_mode():  # the llm head on the same hidden states, so no second pass
        hidden = torch.as_tensor(feat, device=DEVICE, dtype=llm_head.weight.dtype)
        llm = llm_head(hidden).squeeze(1).float().cpu().numpy() * std + mean
    vision = vft["vision_predict"](enc, head, image_processor, images, tools) * std + mean
    np.savez(ROOT / "nr_feat.npz", ids=ids, feat=feat.astype(np.float16), llm=llm, vision=vision)
    print(f"NR_DONE {len(ids)} images, {(time.time() - start) / 60:.1f} min", flush=True)


main()
