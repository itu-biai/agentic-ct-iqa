"""The encoder-only tasks of colab/vm/infer329.py (enc: E19; probe: T1 on the lab set; cnn: ResNeXt-50 on the lab set),
unchanged, for a small GPU (L4): the 27B language model is not loaded. Same definitions as infer329.py:
  enc    mean over patch tokens of the encoder's last hidden state (centre 480 crop), pretrained encoder and the fold-1
         tuned encoder (ROOT/weights_fold_1.pt), on the 1000 training and 300 test slices.
  probe  qwen_probe features (blocks 8, 17, 26 mean and std, merger mean; full 512 slice as RGB, float32) of the
         pretrained encoder on the first 50 training slices (check against the local features.npz) and the lab images.
  cnn    ResNeXt-50 (ROOT/cnn/weights_fold_<k>.pt) on the lab images, 4 flip views, 5 folds.
Writes ROOT/i329/{enc,probe,cnn}.npz; prints ENC_DONE. Run as a plain process (nohup).
"""
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("VFT_ROOT", "/content/vft"))
OUT = ROOT / "i329"
OUT.mkdir(parents=True, exist_ok=True)
vft = {"__name__": "vftb_train"}
exec(compile((ROOT / "vftb_train.py").read_text(), str(ROOT / "vftb_train.py"), "exec"), vft)
DEVICE = vft["DEVICE"]


def main() -> None:
    from transformers import AutoImageProcessor
    config = vft["configs"]()
    processor = AutoImageProcessor.from_pretrained(ROOT)
    z, t, lab = np.load(ROOT / "train.npz"), np.load(ROOT / "test.npz"), np.load(ROOT / "lab329.npz")
    if not (OUT / "enc.npz").exists():
        out = {}
        for name in ("pretrained", "tuned"):
            enc = vft["encoder"](config)
            if name == "tuned":
                w = torch.load(ROOT / "weights_fold_1.pt", map_location="cpu")
                enc.load_state_dict({key: v.float() for key, v in w["encoder"].items()})
            enc = enc.to(DEVICE).eval()
            for part, images in (("train", z["images"]), ("test", t["images"])):
                feats = []
                with torch.inference_mode():
                    for s in range(0, len(images), 16):
                        inputs = processor(images=vft["crops"](images[s:s + 16], False, None), return_tensors="pt")
                        grid = inputs["image_grid_thw"].to(DEVICE)
                        with vft["autocast"]():
                            tokens = enc(inputs["pixel_values"].to(DEVICE), grid_thw=grid).last_hidden_state
                        feats += [x.float().mean(0).cpu().numpy() for x in torch.split(tokens, grid.prod(1).tolist())]
                out[f"{name}_{part}"] = np.stack(feats).astype(np.float16)
            del enc
            torch.cuda.empty_cache()
        np.savez(OUT / "enc.npz", **out)
        print("ENC task done", flush=True)
    if not (OUT / "probe.npz").exists():
        enc = vft["encoder"](config).to(DEVICE).eval()
        captured = {}
        for layer in (8, 17, 26):
            enc.blocks[layer].register_forward_hook(lambda m, i, o, layer=layer: captured.__setitem__(layer, o))
        out = {}
        for part, images in (("train50", z["images"][:50]), ("lab", lab["images"])):
            rows = []
            with torch.inference_mode():
                for img in images:
                    inputs = processor(images=[np.repeat(img[:, :, None], 3, 2)], return_tensors="pt")
                    o = enc(inputs["pixel_values"].float().to(DEVICE), grid_thw=inputs["image_grid_thw"].to(DEVICE))
                    parts = []
                    for layer in (8, 17, 26):
                        tokens = captured[layer][0] if isinstance(captured[layer], tuple) else captured[layer]
                        parts += [tokens.mean(0), tokens.std(0)]
                    parts.append(o.pooler_output.mean(0))
                    rows.append(torch.cat(parts).float().cpu().numpy())
            out[part] = np.stack(rows).astype(np.float32)
        np.savez(OUT / "probe.npz", **out)
        del enc
        torch.cuda.empty_cache()
        print("PROBE task done", flush=True)
    if not (OUT / "cnn.npz").exists():
        import timm
        cnn = {"__name__": "cnn_train"}
        exec(compile((ROOT / "cnn_train.py").read_text(), str(ROOT / "cnn_train.py"), "exec"), cnn)
        out = {}
        for k in range(1, 6):
            model = timm.create_model(cnn["ARCH"], pretrained=False, num_classes=1).to(DEVICE)
            model.load_state_dict(torch.load(ROOT / f"cnn/weights_fold_{k}.pt", map_location=DEVICE))
            tr = z["fold"] != k
            out[f"cnn_{k}"] = cnn["predict"](model, lab["images"]) * z["mos"][tr].std() + z["mos"][tr].mean()
            del model
        np.savez(OUT / "cnn.npz", **out)
        print("CNN task done", flush=True)
    print("ENC_DONE", flush=True)


main()
