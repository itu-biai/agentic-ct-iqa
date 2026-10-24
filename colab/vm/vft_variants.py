"""Stage A, improvement round (scripts/qwen_vision_ft.py variants): the Stage A recipe (colab/vm/vft_train.py) plus a
pairwise ranking loss, scored with and without test-time augmentation, in two variants: with the tool percentiles
(rank_tools) and without them (rank_vision). Same 5 grouped folds, same seeds, final epoch.

Selection is fixed in advance and made on the out-of-fold scores only: among Stage A (2.836) and each variant with or
without TTA, the highest out-of-fold overall wins. Each fold model also scores the test slices, but those scores are
opened only for the winner and its with/without-tools counterpart.

Writes ROOT/out_<variant>/fold_<k>.npz (val index, val and test predictions, centre crop and TTA), log.txt; skips
finished folds; prints VARIANTS_DONE. No sys.argv; VFT_ROOT and VFT_SMOKE=1 from the environment as in vft_train.py.
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
TEST = ROOT / "test.npz"
VARIANTS = {"rank_tools": True, "rank_vision": False}  # name: reads the tool percentiles
REPO = "https://huggingface.co/Qwen/Qwen3.8-27B-FP8/resolve/main"
EPOCHS, BATCH, CROP = 10, (2 if SMOKE else 8), 480
LR_ENCODER, LR_HEAD, WEIGHT_DECAY, WARMUP = 2e-5, 1e-3, 0.05, 0.05
RANK_WEIGHT = 1.0  # loss = MSE + RANK_WEIGHT * mean pairwise logistic loss over the batch pairs with different MOS
SEED = 42
DEVICE = "cpu" if SMOKE else "cuda"


def fetch(name, start=None, end=None):
    headers = {"Range": f"bytes={start}-{end}"} if start is not None else {}
    return urllib.request.urlopen(urllib.request.Request(f"{REPO}/{name}", headers=headers), timeout=600).read()


def weights() -> Path:
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
    for p in model.merger.parameters():
        p.requires_grad_(False)
    return model.float(), AutoImageProcessor.from_pretrained(ROOT)


class Head(nn.Module):
    def __init__(self, width: int, tools: int):
        super().__init__()
        self.norm = nn.LayerNorm(2 * width)
        self.mlp = nn.Sequential(nn.Linear(2 * width + tools, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))

    def forward(self, tokens, counts, tools):
        pooled = torch.stack([torch.cat([t.mean(0), t.std(0)]) for t in torch.split(tokens, counts)])
        return self.mlp(torch.cat([self.norm(pooled.float()), tools], 1)).squeeze(1)


def views(img: np.ndarray) -> list[np.ndarray]:
    """TTA: centre and four corner crops, each also mirrored (10 views); the first is the plain centre crop."""
    o, e = (img.shape[0] - CROP) // 2, img.shape[0] - CROP
    crops = [img[y:y + CROP, x:x + CROP] for y, x in ((o, o), (0, 0), (0, e), (e, 0), (e, e))]
    return [v for c in crops for v in (c, c[:, ::-1])]


def rgb(img: np.ndarray) -> np.ndarray:
    return np.repeat(np.ascontiguousarray(img)[:, :, None], 3, 2)


def train_images(images: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    out = []
    for img in images:
        y, x = rng.integers(0, img.shape[0] - CROP + 1, 2)
        img = img[y:y + CROP, x:x + CROP]
        out.append(rgb(img[:, ::-1] if rng.random() < 0.5 else img))
    return out


def forward(model, head, processor, images, tools):
    inputs = processor(images=images, return_tensors="pt")
    grid = inputs["image_grid_thw"].to(DEVICE)
    with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        tokens = model(inputs["pixel_values"].to(DEVICE), grid_thw=grid).last_hidden_state
    return head(tokens, grid.prod(1).tolist(), torch.as_tensor(tools, device=DEVICE))


def predict(model, head, processor, images, tools, tta: bool) -> np.ndarray:
    """Centre crop only, or the mean over the 10 TTA views."""
    model.eval(); head.eval()
    n = 10 if tta else 1
    preds = []
    with torch.inference_mode():
        for img, t in zip(images, tools):
            v = [rgb(x) for x in views(img)[:n]]
            preds.append(forward(model, head, processor, v, np.repeat(t[None], n, 0)).float().mean().item())
    return np.array(preds)


def rank_loss(pred, target):
    diff_p = pred[:, None] - pred[None, :]
    sign = torch.sign(target[:, None] - target[None, :])
    mask = sign != 0
    return nn.functional.softplus(-sign[mask] * diff_p[mask]).mean() if mask.any() else pred.sum() * 0


def log(out: Path, text: str) -> None:
    print(text, flush=True)
    with open(out / "log.txt", "a") as handle:
        handle.write(text + "\n")


def train_fold(name: str, use_tools: bool, k: int, data, test) -> None:
    from scipy.stats import pearsonr, spearmanr
    out = ROOT / f"out_{name}"
    torch.manual_seed(SEED + k)
    rng = np.random.default_rng(SEED + k)
    tr, va = np.nonzero(data["fold"] != k)[0], np.nonzero(data["fold"] == k)[0]
    epochs, steps_cap = EPOCHS, None
    tools = data["tools"] if use_tools else np.zeros((len(data["mos"]), 0), np.float32)
    test_tools = test["tools"] if use_tools else np.zeros((len(test["ids"]), 0), np.float32)
    test_images = test["images"]
    if SMOKE:
        tr, va, epochs, steps_cap, test_images, test_tools = tr[:4], va[:3], 1, 2, test_images[:2], test_tools[:2]
    mean, std = data["mos"][tr].mean(), data["mos"][tr].std()
    model, processor = encoder()
    model.to(DEVICE)
    if not SMOKE:
        model.gradient_checkpointing_enable()
    head = Head(model.config.hidden_size, tools.shape[1]).to(DEVICE)
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
            target = torch.as_tensor((data["mos"][idx] - mean) / std, device=DEVICE).float()
            pred = forward(model, head, processor, train_images(data["images"][idx], rng), tools[idx]).float()
            loss = nn.functional.mse_loss(pred, target) + RANK_WEIGHT * rank_loss(pred, target)
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
            optim.step(); sched.step()
            losses.append(loss.item())
        if epoch == epochs - 1 or epoch % 3 == 2:
            val = predict(model, head, processor, data["images"][va], tools[va], False) * std + mean
            r = abs(pearsonr(val, data["mos"][va])[0]) + abs(spearmanr(val, data["mos"][va])[0])
            log(out, f"{name} fold {k} epoch {epoch + 1}: loss {np.mean(losses):.4f}, val PLCC+SROCC {r:.3f}, "
                     f"{(time.time() - start) / 60:.1f} min")
    result = {"va": va}
    for tta in (False, True):
        tag = "tta" if tta else "centre"
        result[f"val_{tag}"] = predict(model, head, processor, data["images"][va], tools[va], tta) * std + mean
        result[f"test_{tag}"] = predict(model, head, processor, test_images, test_tools, tta) * std + mean
    np.savez(out / f"fold_{k}.npz", **result)
    log(out, f"{name} fold {k} saved, {(time.time() - start) / 60:.1f} min")


def main() -> None:
    assert TEST.exists(), f"{TEST} is missing"
    z, t = np.load(ROOT / "train.npz"), np.load(TEST)
    data = {k: z[k] for k in ("ids", "mos", "images", "tools", "fold")}
    test = {k: t[k] for k in ("ids", "images", "tools")}
    for name, use_tools in (list(VARIANTS.items())[:1] if SMOKE else VARIANTS.items()):
        (ROOT / f"out_{name}").mkdir(parents=True, exist_ok=True)
        for k in ([1] if SMOKE else range(1, 6)):
            if not (ROOT / f"out_{name}" / f"fold_{k}.npz").exists():
                train_fold(name, use_tools, k, data, test)
    print("SMOKE_OK" if SMOKE else "VARIANTS_DONE", flush=True)


main()
