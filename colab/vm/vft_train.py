"""Stage A on the VM (scripts/qwen_vision_ft.py): fine-tune Qwen3.8-27B's vision encoder + a small head that also reads
the tool percentiles, 5 folds grouped by base slice, fixed recipe, final epoch (no epoch selection).

Expects ROOT/train.npz, config.json, preprocessor_config.json (colab/vft_job/). Downloads the visual tensors of the
FP8 checkpoint (they are bf16) by HTTP range if ROOT/qwen38_visual.safetensors is missing. Writes ROOT/out/: oof.csv,
fold_<k>.pt (encoder + head), log.txt; skips finished folds; prints VFT_DONE. No sys.argv (`colab exec`); VFT_ROOT and
VFT_SMOKE=1 (4 slices, 2 steps, 1 fold, CPU) come from the environment for a local check.
"""
import json
import os
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(os.environ.get("VFT_ROOT", "/content/vft"))
SMOKE = os.environ.get("VFT_SMOKE") == "1"
OUT = ROOT / "out"
TEST = ROOT / "test.npz"  # present only for the test look (scripts/qwen_vision_ft.py test-prep)
REQUIRE_TEST = True  # the test look: refuse to train without the test slices (a run once lost them silently)
REPO = "https://huggingface.co/Qwen/Qwen3.8-27B-FP8/resolve/main"
EPOCHS, BATCH, CROP = 10, (2 if SMOKE else 8), 480  # 480 = 15 x (patch 16 x merge 2); training: random crop + horizontal flip
LR_ENCODER, LR_HEAD, WEIGHT_DECAY, WARMUP = 2e-5, 1e-3, 0.05, 0.05
SEED = 42
DEVICE = "cpu" if SMOKE else "cuda"


def fetch(name, start=None, end=None):
    headers = {"Range": f"bytes={start}-{end}"} if start is not None else {}
    return urllib.request.urlopen(urllib.request.Request(f"{REPO}/{name}", headers=headers), timeout=600).read()


def weights() -> Path:
    """Same as scripts/qwen_probe.py weights(): the model.visual.* block of outside.safetensors as its own file."""
    path = ROOT / "qwen38_visual.safetensors"
    if path.exists():
        return path
    size = struct.unpack("<Q", fetch("outside.safetensors", 0, 7))[0]
    header = json.loads(fetch("outside.safetensors", 8, 8 + size - 1))
    visual = {k: v for k, v in header.items() if k.startswith("model.visual.")}
    first = min(v["data_offsets"][0] for v in visual.values())
    last = max(v["data_offsets"][1] for v in visual.values())
    block = fetch("outside.safetensors", 8 + size + first, 8 + size + last - 1)
    new = {k.removeprefix("model.visual."): {**v, "data_offsets": [v["data_offsets"][0] - first, v["data_offsets"][1] - first]}
           for k, v in visual.items()}
    raw = json.dumps(new).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + block)
    return path


def encoder():
    from safetensors.torch import load_file
    from transformers import AutoImageProcessor
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    config = Qwen3_5Config.from_pretrained(ROOT).vision_config
    config._attn_implementation = "sdpa"
    model = Qwen3_5VisionModel(config)
    missing, unexpected = model.load_state_dict(load_file(weights()), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    for p in model.merger.parameters():  # the head pools the block output; the merger is not used
        p.requires_grad_(False)
    return model.float(), AutoImageProcessor.from_pretrained(ROOT)


class Head(nn.Module):
    """Mean and std of the patch tokens (std carries texture, i.e. noise), plus the tool percentiles."""

    def __init__(self, width: int, tools: int):
        super().__init__()
        self.norm = nn.LayerNorm(2 * width)
        self.mlp = nn.Sequential(nn.Linear(2 * width + tools, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))

    def forward(self, tokens, counts, tools):
        pooled = torch.stack([torch.cat([t.mean(0), t.std(0)]) for t in torch.split(tokens, counts)])
        return self.mlp(torch.cat([self.norm(pooled.float()), tools], 1)).squeeze(1)


def batch_images(images: np.ndarray, train: bool, rng: np.random.Generator) -> list[np.ndarray]:
    out = []
    for img in images:
        if train:
            y, x = rng.integers(0, img.shape[0] - CROP + 1, 2)
            img = img[y:y + CROP, x:x + CROP]
            if rng.random() < 0.5:
                img = img[:, ::-1]
        else:
            o = (img.shape[0] - CROP) // 2
            img = img[o:o + CROP, o:o + CROP]
        out.append(np.repeat(np.ascontiguousarray(img)[:, :, None], 3, 2))
    return out


def forward(model, head, processor, images, tools):
    inputs = processor(images=images, return_tensors="pt")
    grid = inputs["image_grid_thw"].to(DEVICE)
    with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        tokens = model(inputs["pixel_values"].to(DEVICE), grid_thw=grid).last_hidden_state
    counts = grid.prod(1).tolist()
    return head(tokens, counts, torch.as_tensor(tools, device=DEVICE))


def predict(model, head, processor, images, tools) -> np.ndarray:
    model.eval(); head.eval()
    preds = []
    with torch.inference_mode():
        for s in range(0, len(images), BATCH):
            preds.append(forward(model, head, processor, batch_images(images[s:s + BATCH], False, None),
                                 tools[s:s + BATCH]).float().cpu().numpy())
    return np.concatenate(preds)


def log(text: str) -> None:
    print(text, flush=True)
    with open(OUT / "log.txt", "a") as handle:
        handle.write(text + "\n")


def train_fold(k: int, data) -> None:
    from scipy.stats import pearsonr, spearmanr
    torch.manual_seed(SEED + k)
    rng = np.random.default_rng(SEED + k)
    tr, va = np.nonzero(data["fold"] != k)[0], np.nonzero(data["fold"] == k)[0]
    epochs, steps_cap = EPOCHS, None
    if SMOKE:
        tr, va, epochs, steps_cap = tr[:4], va[:4], 1, 2
    mean, std = data["mos"][tr].mean(), data["mos"][tr].std()
    model, processor = encoder()
    model.to(DEVICE)
    if not SMOKE:
        model.gradient_checkpointing_enable()
    model.train()
    head = Head(model.config.hidden_size, data["tools"].shape[1]).to(DEVICE)
    params = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": LR_ENCODER},
              {"params": head.parameters(), "lr": LR_HEAD}]
    optim = torch.optim.AdamW(params, weight_decay=WEIGHT_DECAY)
    per_epoch = int(np.ceil(len(tr) / BATCH)) if steps_cap is None else steps_cap
    total = per_epoch * epochs
    warm = max(1, int(WARMUP * total))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1, s / total))))
    start = time.time()
    for epoch in range(epochs):
        model.train(); head.train()
        order = rng.permutation(tr)
        losses = []
        for step in range(per_epoch):
            idx = order[step * BATCH:(step + 1) * BATCH]
            target = torch.as_tensor((data["mos"][idx] - mean) / std, device=DEVICE)
            pred = forward(model, head, processor, batch_images(data["images"][idx], True, rng), data["tools"][idx])
            loss = nn.functional.mse_loss(pred.float(), target.float())
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
            optim.step(); sched.step()
            losses.append(loss.item())
        val = predict(model, head, processor, data["images"][va], data["tools"][va]) * std + mean
        y = data["mos"][va]
        r = abs(pearsonr(val, y)[0]) + abs(spearmanr(val, y)[0])
        log(f"fold {k} epoch {epoch + 1}: train loss {np.mean(losses):.4f}, val PLCC+SROCC {r:.3f}, "
            f"{(time.time() - start) / 60:.1f} min")
    np.save(OUT / f"oof_fold_{k}.npy", np.stack([va, val]))
    if TEST.exists():  # the test look: each fold model scores the 300 test slices (no MOS on the VM)
        t = np.load(TEST)
        np.save(OUT / f"test_fold_{k}.npy", predict(model, head, processor, t["images"], t["tools"]) * std + mean)
    torch.save({"encoder": {n: p.to(torch.bfloat16) for n, p in model.state_dict().items()}, "head": head.state_dict(),
                "mean": mean, "std": std}, OUT / f"fold_{k}.pt")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert SMOKE or not REQUIRE_TEST or TEST.exists(), f"{TEST} is missing"
    z = np.load(ROOT / "train.npz")
    data = {k: z[k] for k in ("ids", "mos", "images", "tools", "fold")}
    for k in ([1] if SMOKE else range(1, 6)):
        if not (OUT / f"oof_fold_{k}.npy").exists() or (TEST.exists() and not (OUT / f"test_fold_{k}.npy").exists()):
            train_fold(k, data)
    if SMOKE:
        print("SMOKE_OK", flush=True)
        return
    pred = np.full(len(data["ids"]), np.nan)
    for k in range(1, 6):
        va, val = np.load(OUT / f"oof_fold_{k}.npy")
        pred[va.astype(int)] = val
    with open(OUT / "oof.csv", "w") as handle:
        handle.write("image_id,pred\n" + "".join(f"{i},{p:.5f}\n" for i, p in zip(data["ids"], pred)))
    if TEST.exists():
        ids = np.load(TEST)["ids"]
        preds = np.stack([np.load(OUT / f"test_fold_{k}.npy") for k in range(1, 6)])
        with open(OUT / "test.csv", "w") as handle:
            handle.write("image_id," + ",".join(f"fold{k}" for k in range(1, 6)) + ",pred\n" + "".join(
                f"{i}," + ",".join(f"{v:.5f}" for v in col) + f",{col.mean():.5f}\n" for i, col in zip(ids, preds.T)))
    print("VFT_DONE", flush=True)


main()
