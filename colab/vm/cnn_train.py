"""A CNN view to ensemble with Stage B (scripts/qwen_vision_ft.py cnn-final): ResNeXt-50 (timm, ImageNet weights) on
the full 512 x 512 slice, image only (no tool inputs, unlike Stage B), the same 5 grouped folds and seeds. MSE on
standardised MOS, AdamW (backbone 1e-4, head 1e-3, weight decay 0.05), 30 epochs, cosine with 5% warm-up, batch 16,
bf16 autocast, horizontal and vertical flips. Predictions (validation and test) average 4 flip views.

Expects ROOT/train.npz and test.npz (colab/vft_job). Writes ROOT/out_cnn/fold_<k>.npz (va, cnn_val, cnn_test) and
weights_fold_<k>.pt, log.txt; skips finished folds; prints CNN_DONE. No sys.argv (`colab exec`); VFT_ROOT and
VFT_SMOKE=1 (resnet18, no pretrained weights, a few steps on the CPU) for a local check.
"""
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(os.environ.get("VFT_ROOT", "/content/vft"))
SMOKE = os.environ.get("VFT_SMOKE") == "1"
OUT = ROOT / "out_cnn"
ARCH = "resnet18" if SMOKE else "resnext50_32x4d.a1h_in1k"
EPOCHS, BATCH, LR_BACKBONE, LR_HEAD, WEIGHT_DECAY, WARMUP = (1, 2, 1e-4, 1e-3, 0.05, 0.05) if SMOKE else (30, 16, 1e-4, 1e-3, 0.05, 0.05)
SEED = 42
DEVICE = "cpu" if SMOKE else "cuda"
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
VIEWS = ((False, False), (True, False), (False, True), (True, True))  # (horizontal, vertical) flips


def log(text: str) -> None:
    print(text, flush=True)
    with open(OUT / "log.txt", "a") as handle:
        handle.write(text + "\n")


def batch_tensor(images: np.ndarray, flips) -> torch.Tensor:
    x = images.astype(np.float32) / 255
    x = np.stack([np.flip(np.flip(img, 1) if h else img, 0) if v else (np.flip(img, 1) if h else img)
                  for img, (h, v) in zip(x, flips)])
    x = np.repeat(x[:, None], 3, 1)
    x = (x - MEAN[None, :, None, None]) / STD[None, :, None, None]
    return torch.from_numpy(np.ascontiguousarray(x)).to(DEVICE)


def predict(model, images: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        for s in range(0, len(images), BATCH):
            chunk = images[s:s + BATCH]
            out.append(np.mean([model(batch_tensor(chunk, [view] * len(chunk))).float().squeeze(1).cpu().numpy()
                                for view in VIEWS], axis=0))
    return np.concatenate(out)


def train_fold(k, data, test, tr, va):
    import timm
    torch.manual_seed(SEED + k)
    rng = np.random.default_rng(SEED + k)
    mean, std = data["mos"][tr].mean(), data["mos"][tr].std()
    model = timm.create_model(ARCH, pretrained=not SMOKE, num_classes=1).to(DEVICE)
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    params = [{"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": LR_BACKBONE},
              {"params": head, "lr": LR_HEAD}]
    optim = torch.optim.AdamW(params, weight_decay=WEIGHT_DECAY)
    per_epoch = 2 if SMOKE else int(np.ceil(len(tr) / BATCH))
    total, warm = per_epoch * EPOCHS, max(1, int(WARMUP * per_epoch * EPOCHS))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1, s / total))))
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(tr)
        losses = []
        for step in range(per_epoch):
            idx = order[step * BATCH:(step + 1) * BATCH]
            flips = [(rng.random() < 0.5, rng.random() < 0.5) for _ in idx]
            target = torch.as_tensor((data["mos"][idx] - mean) / std, device=DEVICE, dtype=torch.float32)
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                pred = model(batch_tensor(data["images"][idx], flips)).float().squeeze(1)
            loss = nn.functional.mse_loss(pred, target)
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step()
            losses.append(loss.item())
        if epoch == EPOCHS - 1 or epoch % 5 == 4:
            from scipy.stats import pearsonr, spearmanr
            val = predict(model, data["images"][va])
            r = abs(pearsonr(val, data["mos"][va])[0]) + abs(spearmanr(val, data["mos"][va])[0])
            log(f"fold {k} cnn epoch {epoch + 1}: loss {np.mean(losses):.4f}, val PLCC+SROCC {r:.3f}, {(time.time() - start) / 60:.1f} min")
    out = {"va": va, "cnn_val": predict(model, data["images"][va]) * std + mean,
           "cnn_test": predict(model, test["images"]) * std + mean}
    torch.save({k_: v.detach().to("cpu", torch.bfloat16) for k_, v in model.state_dict().items()}, OUT / f"weights_fold_{k}.pt")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z, t = np.load(ROOT / "train.npz"), np.load(ROOT / "test.npz")
    data = {key: z[key] for key in ("mos", "images", "fold")}
    test = {"images": t["images"][:4] if SMOKE else t["images"]}
    for k in ([1] if SMOKE else range(1, 6)):
        if (OUT / f"fold_{k}.npz").exists() and not SMOKE:
            continue
        tr, va = np.nonzero(data["fold"] != k)[0], np.nonzero(data["fold"] == k)[0]
        if SMOKE:
            tr, va = tr[:8], va[:4]
        np.savez(OUT / f"fold_{k}.npz", **train_fold(k, data, test, tr, va))
        log(f"fold {k} saved")
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    print("SMOKE_OK" if SMOKE else "CNN_DONE", flush=True)


main()
