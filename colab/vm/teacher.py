"""Teacher scores for the unlabelled pool (scripts/pseudo_pool.py): the 5 fold models of the Stage B rerun
(ROOT/weights_fold_<k>.pt, from vftb_train.py) reloaded on the public 27B checkpoint, each scoring the pool (llm and
vision heads, centre crop). As a check that the weights load correctly, every fold also re-predicts the 300 test
images; locally these must match the run's saved llm_test.

Expects ROOT/vftb_train.py, train.npz, test.npz, pseudo.npz, config.json, preprocessor_config.json, the weights and the
checkpoint shards (colab/vm/vftb_setup.py). Writes ROOT/teacher.npz (llm_<k>, vision_<k>, test_llm_<k>); prints
TEACHER_DONE. With the flag file ROOT/cross it scores ROOT/cross.npz (scripts/cross_pool.py) into ROOT/cross_pred.npz. No sys.argv (`colab exec`); VFT_ROOT and VFT_SMOKE=1 for a local check.
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
vft["OUT"].mkdir(parents=True, exist_ok=True)
DEVICE, FOLDS = vft["DEVICE"], ([1] if vft["SMOKE"] else range(1, 6))
INPUT, RESULT = ("cross.npz", "cross_pred.npz") if (ROOT / "cross").exists() else ("pseudo.npz", "teacher.npz")


def main() -> None:
    from transformers import AutoImageProcessor, AutoProcessor
    z, t, p = np.load(ROOT / "train.npz"), np.load(ROOT / "test.npz"), np.load(ROOT / INPUT, allow_pickle=True)
    t, p = ({key: v[:3] if vft["SMOKE"] else v for key, v in f.items()} for f in (t, p))
    config = vft["configs"]()
    image_processor = AutoImageProcessor.from_pretrained(ROOT)
    processor = AutoProcessor.from_pretrained(vft["REPO_ID"])
    model, _ = vft["language_model"](config)
    loras = {name: m for name, m in model.language_model.named_modules() if isinstance(m, vft["LoRA"])}
    out, start = {}, time.time()
    for k in FOLDS:
        w = torch.load(ROOT / f"weights_fold_{k}.pt", map_location="cpu")
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
        tr = np.nonzero(z["fold"] != k)[0]
        mean, std = z["mos"][tr].mean(), z["mos"][tr].std()
        out[f"test_llm_{k}"] = vft["llm_predict"](model, enc, llm_head, processor, t["images"], t["tools"]) * std + mean
        out[f"llm_{k}"] = vft["llm_predict"](model, enc, llm_head, processor, p["images"], p["tools"]) * std + mean
        out[f"vision_{k}"] = vft["vision_predict"](enc, head, image_processor, p["images"], p["tools"]) * std + mean
        np.savez(ROOT / RESULT, ids=p["ids"], **out)
        print(f"teacher fold {k} done, {(time.time() - start) / 60:.1f} min", flush=True)
        model.visual = None
        del enc, head, llm_head, w
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    print("TEACHER_DONE", flush=True)


main()
