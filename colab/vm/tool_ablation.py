"""Tool ablation at inference (scripts/tool_ablation_eval.py): the 5 fold models of the Stage B rerun
(ROOT/weights_fold_<k>.pt), reloaded as in colab/vm/teacher.py, score every fold's validation slices (out-of-fold) and
the 300 test images once with the real tool inputs and once per ablation, where the ablated inputs are set to 0.5 (the
bank median: "as usual"; the prompt then reads 50). No retraining, so this measures how much the trained model relies
on each input. Writes ROOT/ablation.npz (va_<k>, <head>_<part>_<ablation>_<k>; head llm / vision, part val / test)
after every fold; prints ABLATION_DONE. No sys.argv (`colab exec`); VFT_ROOT and VFT_SMOKE=1 for a local check.
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
DEVICE, FOLDS = vft["DEVICE"], ([1] if vft["SMOKE"] else range(1, 6))
# columns of TOOL_NAMES: residual_std, gradient_deviation_mean, hf_residual_mean, noise_score, gradient_p95,
# streak_anisotropy
ABLATIONS = {"full": (), "residual_std": (0,), "gradient_deviation_mean": (1,), "hf_residual_mean": (2,),
             "noise_score": (3,), "gradient_p95": (4,), "streak_anisotropy": (5,), "noise": (0, 1, 2, 3),
             "all": (0, 1, 2, 3, 4, 5)}
NEUTRAL = 0.5


def ablate(tools: np.ndarray, cols: tuple) -> np.ndarray:
    tools = tools.copy()
    tools[:, list(cols)] = NEUTRAL
    return tools


def main() -> None:
    from transformers import AutoImageProcessor, AutoProcessor
    z, t = np.load(ROOT / "train.npz"), np.load(ROOT / "test.npz")
    test = {key: t[key][:3] if vft["SMOKE"] else t[key] for key in ("images", "tools")}
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
        tr, va = np.nonzero(z["fold"] != k)[0], np.nonzero(z["fold"] == k)[0]
        va = va[:3] if vft["SMOKE"] else va
        mean, std = z["mos"][tr].mean(), z["mos"][tr].std()
        out[f"va_{k}"] = va
        parts = {"val": (z["images"][va], z["tools"][va]), "test": (test["images"], test["tools"])}
        for ab, cols in ABLATIONS.items():
            for part, (images, tools) in parts.items():
                tools = ablate(tools, cols)
                out[f"llm_{part}_{ab}_{k}"] = vft["llm_predict"](model, enc, llm_head, processor, images, tools) * std + mean
                out[f"vision_{part}_{ab}_{k}"] = vft["vision_predict"](enc, head, image_processor, images, tools) * std + mean
            print(f"fold {k} {ab}, {(time.time() - start) / 60:.1f} min", flush=True)
        np.savez(ROOT / "ablation.npz", **out)
        print(f"ablation fold {k} done, {(time.time() - start) / 60:.1f} min", flush=True)
        model.visual = None
        del enc, head, llm_head, w
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    print("ABLATION_DONE", flush=True)


main()
