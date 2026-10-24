"""Stage B on the VM (scripts/qwen_vision_ft.py llm-*): the fine-tuned eye plugged back into Qwen3.8-27B, whose language
model reads the image and the tool measurements (as text) and is trained with LoRA to give the quality score.

Per fold (same 5 grouped folds and seeds as Stage A):
  1. vision: the Stage A recipe unchanged (encoder fully fine-tuned + head on pooled tokens and tool percentiles, 10
     epochs). Its val/test predictions are kept ("vision").
  2. llm: the trained encoder is frozen and feeds the 27B language model through the merger. Trainable: the merger,
     LoRA (rank 16) on every projection of all 64 layers, and a linear head on the last token's hidden state; 3 epochs,
     MSE on standardised MOS. The prompt gives the six tool percentiles as two-digit numbers, so every prompt has the
     same length (no padding).
The language model weights are the official FP8 checkpoint, dequantised to bf16 on load (block 128 scales).

Expects ROOT/train.npz, test.npz, config.json, preprocessor_config.json and the checkpoint shards in HF_DIR
(colab/vm/vftb_setup.py). Writes ROOT/out_llm/fold_<k>.npz (va, vision_val, vision_test, llm_val, llm_test), log.txt;
skips finished folds; prints LLM_DONE.
ROOT/full_data present: the same recipe trained on all 1000 slices with 3 seeds (no held-out fold; the "val" columns
are 16 training slices, in-sample, only a progress check) -> ROOT/out_full/seed_<s>.npz.
ROOT/upscale2 present: every crop (training, validation, test) is enlarged 2x (bicubic, 480 -> 960) before the
encoder, so a 16-pixel patch covers 8 x 8 CT pixels and the noise texture is resolved (3600 patches, 900 LLM tokens per
image) -> ROOT/out_up2/fold_<k>.npz. ROOT/synth present: before the folds, the encoder is rank pre-trained once on
synthetic LDCT-IQA-style degradations of MSD Task09 Spleen slices (61 other patients, colab/vm/synth.py; no MOS, so no
fold leakage), and every fold starts from it; ROOT/windows3 present: the encoder's three input channels are the stored
window (1400 / -300), the radiologists' soft-tissue window (350 / 40) and a narrow one (150 / 40), from
train16.npz / test16.npz -> ROOT/out_syn/fold_<k>.npz. ROOT/rank present: both phases add a soft pairwise ranking
term to the MSE (every pair in the batch; target probability sigmoid(dMOS / T), so its optimum is the MSE's), the final
predictions are also made with test-time augmentation (6 crop/flip views, "*_tta_*" columns), and the language model's
last-token features (centre view) of the fold's training, validation and test slices are kept in every run ("feat_*", fp16) ->
ROOT/out_rank/fold_<k>.npz. ROOT/pseudo present: teacher-student; ROOT/pseudo.npz (1816 unlabelled lab images, scripts/pseudo_pool.py)
with ROOT/pseudo_labels.npz (label_<k>: teacher scores for fold k, scripts/teacher_labels.py) join training: every epoch of both
phases sees the fold's training slices plus as many pool images drawn at random -> ROOT/out_pseudo/fold_<k>.npz.
Issue #329 ablations: ROOT/notools present: no tool percentiles anywhere (the vision head has no tool inputs, the
prompt no numbers), same seeds -> ROOT/out_notools (E9; its vision head is Stage A without tools, E6). ROOT/phase2only:
phase 1 skipped, phase 2 starts from the pretrained encoder (its merger trained as usual) -> ROOT/out_p2only (E10).
ROOT/patientfolds: phase 1 only, folds from ROOT/patient_folds.npy (scripts/patient_folds.py), no weights kept ->
ROOT/out_patient (E8). ROOT/seed2 or seed3: the unchanged recipe with every seed shifted by 10000 (seed2) or 20000
(seed3) -> ROOT/out_seed2 / out_seed3 (E12).
ROOT/probe_only present: 12 llm steps on fold 1 with the original encoder, prints
seconds per step and peak memory, writes nothing. No sys.argv (`colab exec`); VFT_ROOT and VFT_SMOKE=1 (tiny random
model on the CPU) from the environment for a local check.
"""
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

ROOT = Path(os.environ.get("VFT_ROOT", "/content/vft"))
SMOKE = os.environ.get("VFT_SMOKE") == "1"
PROBE = (ROOT / "probe_only").exists()
FULL = (ROOT / "full_data").exists()
UPSCALE = 2 if (ROOT / "upscale2").exists() else 1
SYNTH = (ROOT / "synth").exists()
WINDOWS3 = (ROOT / "windows3").exists()
PAIRS = (ROOT / "rank").exists()
PSEUDO = (ROOT / "pseudo").exists()
NOTOOLS = (ROOT / "notools").exists()
P2ONLY = (ROOT / "phase2only").exists()
PATIENT = (ROOT / "patientfolds").exists()
SEED_RUN = 3 if (ROOT / "seed3").exists() else 2 if (ROOT / "seed2").exists() else 1
POOL = np.zeros(0, int)  # indices of the unlabelled pool in data (PSEUDO), set in main
SEEDS = 3
HF_DIR = Path("/content/hf")
MSD_DIR = Path(os.environ.get("VFT_MSD", "/content/msd/Task09_Spleen"))
OUT = ROOT / ("out_full" if FULL else "out_up2" if UPSCALE == 2 else "out_syn" if SYNTH or WINDOWS3
              else "out_rank" if PAIRS else "out_pseudo" if PSEUDO else "out_notools" if NOTOOLS
              else "out_p2only" if P2ONLY else "out_patient" if PATIENT else f"out_seed{SEED_RUN}" if SEED_RUN > 1
              else "out_llm")
PAIR_WEIGHT, PAIR_TEMP = 0.1, 0.1  # ranking term weight; temperature in standardised MOS units (~0.12 MOS)
TTA_VIEWS = ((0.5, 0.5, False), (0.5, 0.5, True), (0, 0, False), (1, 1, True), (0, 1, True), (1, 0, False))  # y, x, flip
PRE_BASES, PRE_VARIANTS, PRE_EPOCHS, PRE_PER_STEP = 20, 8, 2, 2  # slices per volume, types per slice, epochs, slices per step
UNIT = "seed" if FULL else "fold"
REPO_ID = "Qwen/Qwen3.8-27B-FP8"
EPOCHS, BATCH, CROP = 10, (2 if SMOKE else 8), 480  # phase 1, = Stage A
LR_ENCODER, LR_HEAD, WEIGHT_DECAY, WARMUP = 2e-5, 1e-3, 0.05, 0.05
LLM_EPOCHS, LR_LORA, LR_MERGER, LR_LLM_HEAD = 3, 1e-4, 2e-5, 1e-3  # phase 2
RANK, ALPHA, LORA_DROPOUT = 16, 32, 0.05
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj",
           "down_proj")
TOOL_NAMES = ("residual_std", "gradient_deviation_mean", "hf_residual_mean", "noise_score", "gradient_p95",
              "streak_anisotropy")
SEED = 42
DEVICE = "cpu" if SMOKE else "cuda"


def log(text: str) -> None:
    print(text, flush=True)
    if not PROBE:
        with open(OUT / "log.txt", "a") as handle:
            handle.write(text + "\n")


def configs():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    config = Qwen3_5Config.from_pretrained(ROOT)
    if SMOKE:  # tiny random model with the real token ids and processor
        t, v = config.text_config, config.vision_config
        t.hidden_size, t.intermediate_size, t.num_hidden_layers, t.layer_types = 64, 128, 4, t.layer_types[:4]
        t.num_attention_heads, t.num_key_value_heads, t.head_dim = 2, 1, 32
        t.linear_num_key_heads, t.linear_num_value_heads, t.linear_key_head_dim, t.linear_value_head_dim = 2, 4, 16, 16
        v.depth, v.hidden_size, v.intermediate_size, v.num_heads, v.out_hidden_size = 2, 64, 128, 2, 64
    config.vision_config._attn_implementation = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    config._attn_implementation = "sdpa"
    return config


# ---------------------------------------------------------------- phase 1: the Stage A recipe (colab/vm/vft_train.py)

def encoder(config):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    model = Qwen3_5VisionModel(config.vision_config)
    if not SMOKE:
        from safetensors import safe_open
        with safe_open(HF_DIR / "outside.safetensors", "pt") as f:
            state = {k.removeprefix("model.visual."): f.get_tensor(k) for k in f.keys() if k.startswith("model.visual.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
    for p in model.merger.parameters():
        p.requires_grad_(False)
    return model.float()


class Head(nn.Module):
    def __init__(self, width: int, tools: int):
        super().__init__()
        self.norm = nn.LayerNorm(2 * width)
        self.mlp = nn.Sequential(nn.Linear(2 * width + tools, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))

    def forward(self, tokens, counts, tools):
        pooled = torch.stack([torch.cat([t.mean(0), t.std(0)]) for t in torch.split(tokens, counts)])
        return self.mlp(torch.cat([self.norm(pooled.float()), tools], 1)).squeeze(1)


def channels(img: np.ndarray) -> np.ndarray:
    """Grey uint8 -> the same value in all three channels; uint16 (WINDOWS3) -> stored, soft-tissue and narrow window."""
    if img.dtype == np.uint8:
        return np.repeat(img[:, :, None], 3, 2)
    v = img.astype(np.float32) / 65535
    hu = v * 1400 - 1000
    return np.round(np.stack([v, (hu + 135) / 350, (hu + 35) / 150], -1).clip(0, 1) * 255).astype(np.uint8)


def crops(images: np.ndarray, train: bool, rng, same: bool = False, view=None) -> list[np.ndarray]:
    """same: one crop position and flip for the whole batch (the variants of one synthetic slice); view: (y, x, flip)
    for evaluation, the crop offsets as fractions of the free margin (default centre, no flip)."""
    out = []
    if train and same:
        y, x = rng.integers(0, images.shape[1] - CROP + 1, 2)
        flip = rng.random() < 0.5
    for img in images:
        if train:
            if not same:
                y, x = rng.integers(0, img.shape[0] - CROP + 1, 2)
                flip = rng.random() < 0.5
            img = img[y:y + CROP, x:x + CROP]
            if flip:
                img = img[:, ::-1]
        else:
            fy, fx, flip = view or (0.5, 0.5, False)
            y, x = int(round(fy * (img.shape[0] - CROP))), int(round(fx * (img.shape[1] - CROP)))
            img = img[y:y + CROP, x:x + CROP]
            if flip:
                img = img[:, ::-1]
        img = np.ascontiguousarray(img)
        if UPSCALE > 1:
            img = np.asarray(Image.fromarray(img).resize((CROP * UPSCALE,) * 2, Image.BICUBIC))
        out.append(channels(img))
    return out


def autocast():
    return torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda")


def vision_forward(enc, head, image_processor, images, tools):
    inputs = image_processor(images=images, return_tensors="pt")
    grid = inputs["image_grid_thw"].to(DEVICE)
    with autocast():
        tokens = enc(inputs["pixel_values"].to(DEVICE), grid_thw=grid).last_hidden_state
    return head(tokens, grid.prod(1).tolist(), torch.as_tensor(tools, device=DEVICE))


def objective(pred, target):
    """MSE on standardised MOS, plus (PAIRS) a soft RankNet term over every pair in the batch: the predicted
    probability that one slice beats the other, sigmoid(d_pred / T), against sigmoid(d_MOS / T)."""
    pred, target = pred.float(), target.float()
    loss = nn.functional.mse_loss(pred, target)
    if PAIRS:
        i, j = torch.triu_indices(len(pred), len(pred), 1, device=pred.device)
        loss = loss + PAIR_WEIGHT * nn.functional.binary_cross_entropy_with_logits(
            (pred[i] - pred[j]) / PAIR_TEMP, torch.sigmoid((target[i] - target[j]) / PAIR_TEMP))
    return loss


def schedule(optim, total):
    warm = max(1, int(WARMUP * total))
    return torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1, s / total))))


def phase1(k, enc, image_processor, data, test, tr, va, mean, std, rng):
    from scipy.stats import pearsonr, spearmanr
    epochs, cap = (1, 2) if SMOKE else (EPOCHS, None)
    if not SMOKE:
        enc.gradient_checkpointing_enable()
    head = Head(enc.config.hidden_size, data["tools"].shape[1]).to(DEVICE)
    params = [{"params": [p for p in enc.parameters() if p.requires_grad], "lr": LR_ENCODER},
              {"params": head.parameters(), "lr": LR_HEAD}]
    optim = torch.optim.AdamW(params, weight_decay=WEIGHT_DECAY)
    per_epoch = cap or int(np.ceil(len(tr) * (2 if PSEUDO else 1) / BATCH))
    sched = schedule(optim, per_epoch * epochs)
    start = time.time()
    for epoch in range(epochs):
        enc.train(); head.train()
        order = rng.permutation(np.concatenate([tr, rng.choice(POOL, len(tr), replace=False)]) if PSEUDO else tr)
        losses = []
        for step in range(per_epoch):
            idx = order[step * BATCH:(step + 1) * BATCH]
            target = torch.as_tensor((data["mos"][idx] - mean) / std, device=DEVICE)
            pred = vision_forward(enc, head, image_processor, crops(data["images"][idx], True, rng), data["tools"][idx])
            loss = objective(pred, target)
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
            optim.step(); sched.step()
            losses.append(loss.item())
        if epoch == epochs - 1 or epoch % 3 == 2:
            val = vision_predict(enc, head, image_processor, data["images"][va], data["tools"][va]) * std + mean
            r = abs(pearsonr(val, data["mos"][va])[0]) + abs(spearmanr(val, data["mos"][va])[0])
            log(f"{UNIT} {k} vision epoch {epoch + 1}: loss {np.mean(losses):.4f}, val PLCC+SROCC {r:.3f}, "
                f"{(time.time() - start) / 60:.1f} min")
    out = {"vision_val": vision_predict(enc, head, image_processor, data["images"][va], data["tools"][va]) * std + mean,
           "vision_test": vision_predict(enc, head, image_processor, test["images"], test["tools"]) * std + mean}
    if PAIRS:
        for part, images, tools in (("val", data["images"][va], data["tools"][va]), ("test", test["images"], test["tools"])):
            out[f"vision_tta_{part}"] = np.mean([vision_predict(enc, head, image_processor, images, tools, v)
                                                 for v in TTA_VIEWS], axis=0) * std + mean
    if not SMOKE:
        enc.gradient_checkpointing_disable()
    out["_vision_head"] = {k: v.detach().cpu() for k, v in head.state_dict().items()}
    del head, optim
    return out


def vision_predict(enc, head, image_processor, images, tools, view=None) -> np.ndarray:
    enc.eval(); head.eval()
    preds = []
    with torch.inference_mode():
        for s in range(0, len(images), BATCH):
            preds.append(vision_forward(enc, head, image_processor, crops(images[s:s + BATCH], False, None, view=view),
                                        tools[s:s + BATCH]).float().cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------- phase 0: rank pre-training on synthetic slices

def pretrain(config, image_processor) -> dict:
    """Encoder + a tool-free head, RankNet loss on the pairs of one synthetic slice where one variant has no more noise
    and no more streaks than the other (and differs). 5% of the slices are held out to report pair accuracy. Returns
    the encoder's state dict (cached in OUT/pretrained.pt)."""
    import sys
    cache = OUT / "pretrained.pt"
    if cache.exists():
        log("pretrained encoder loaded from cache")
        return torch.load(cache, map_location="cpu")
    sys.path.insert(0, str(ROOT))  # synth.py is uploaded next to the data
    import synth
    paths = sorted(p for d in ("imagesTr", "imagesTs") for p in (MSD_DIR / d).glob("*.nii.gz") if not p.name.startswith("._"))
    per_volume, variants, epochs = (2, 3, 1) if SMOKE else (PRE_BASES, PRE_VARIANTS, PRE_EPOCHS)
    if SMOKE:
        paths = paths[:2]
    start = time.time()
    bank = synth.build_bank(paths, per_volume, variants, SEED, DEVICE, log)
    log(f"synthetic bank: {len(bank['images'])} images of {bank['base'].max() + 1} slices from {len(paths)} volumes, "
        f"{(time.time() - start) / 60:.1f} min")
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    slices = rng.permutation(bank["base"].max() + 1)
    held = set(slices[:max(1, len(slices) // 20)].tolist())
    train_slices = [s for s in slices if s not in held]
    members = {s: np.nonzero(bank["base"] == s)[0] for s in slices}
    enc = encoder(config).to(DEVICE)
    if not SMOKE:
        enc.gradient_checkpointing_enable()
    head = Head(enc.config.hidden_size, 0).to(DEVICE)
    params = [{"params": [p for p in enc.parameters() if p.requires_grad], "lr": LR_ENCODER},
              {"params": head.parameters(), "lr": LR_HEAD}]
    optim = torch.optim.AdamW(params, weight_decay=WEIGHT_DECAY)
    per_epoch = 2 if SMOKE else len(train_slices) // PRE_PER_STEP
    sched = schedule(optim, per_epoch * epochs)

    def scores(idx, train):
        imgs = crops(bank["images"][idx], train, rng, same=True)
        return vision_forward(enc, head, image_processor, imgs, np.zeros((len(idx), 0), np.float32)).float()

    def pair_loss(s, idx):
        n, k = torch.as_tensor(bank["noise"][idx], device=DEVICE), torch.as_tensor(bank["streak"][idx], device=DEVICE)
        better = (n[:, None] <= n[None]) & (k[:, None] <= k[None]) & ((n[:, None] < n[None]) | (k[:, None] < k[None]))
        diff = s[:, None] - s[None]
        return nn.functional.softplus(-diff[better]), (diff[better] > 0).float()

    def held_accuracy():
        enc.eval(); head.eval()
        hits = []
        with torch.inference_mode():
            for s in sorted(held):
                _, ok = pair_loss(scores(members[s], False), members[s])
                hits.append(ok)
        enc.train(); head.train()
        return torch.cat(hits).mean().item() if hits else float("nan")

    log(f"pretrain: held-out pair accuracy before {held_accuracy():.3f}")
    for epoch in range(epochs):
        enc.train(); head.train()
        order = rng.permutation(train_slices)
        losses = []
        for step in range(per_epoch):
            group = order[step * PRE_PER_STEP:(step + 1) * PRE_PER_STEP]
            loss = torch.cat([pair_loss(scores(members[s], True), members[s])[0] for s in group]).mean()
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
            optim.step(); sched.step()
            losses.append(loss.item())
        log(f"pretrain epoch {epoch + 1}: loss {np.mean(losses):.4f}, held-out pair accuracy {held_accuracy():.3f}, "
            f"{(time.time() - start) / 60:.1f} min")
    state = {k: v.detach().cpu() for k, v in enc.state_dict().items()}
    torch.save(state, cache)
    del enc, head, optim, bank
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return state


# ---------------------------------------------------------------- phase 2: the language model

class LoRA(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(RANK, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, RANK, device=base.weight.device))
        self.drop = nn.Dropout(LORA_DROPOUT)
        self.reset()

    def reset(self):
        nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)
        nn.init.zeros_(self.b)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.a.t().to(x.dtype)) @ self.b.t().to(x.dtype) * (ALPHA / RANK)


def dequantise(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 e4m3 weight with one scale per 128 x 128 block -> bf16."""
    rows, cols = weight.shape
    s = scale.float().repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :cols]
    return (weight.float() * s).to(torch.bfloat16)


def language_model(config):
    """Qwen3_5Model without its own encoder, the 27B weights loaded shard by shard (FP8 dequantised)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    torch.set_default_dtype(torch.float32 if SMOKE else torch.bfloat16)
    with torch.device(DEVICE):
        model = Qwen3_5Model(config)
    torch.set_default_dtype(torch.float32)
    model.visual = None  # replaced by the fine-tuned encoder of each fold
    model.requires_grad_(False)
    if not SMOKE:
        from safetensors import safe_open
        params = dict(model.language_model.named_parameters())
        loaded = set()
        start = time.time()
        for shard in sorted(HF_DIR.glob("*.safetensors")):
            if shard.name == "mtp.safetensors":
                continue
            with safe_open(shard, "pt", device=DEVICE) as f:
                keys = set(f.keys())
                for key in keys:
                    if not key.startswith("model.language_model.") or key.endswith("_scale_inv"):
                        continue
                    name = key.removeprefix("model.language_model.")
                    tensor = f.get_tensor(key)
                    if f"{key}_scale_inv" in keys:
                        tensor = dequantise(tensor, f.get_tensor(f"{key}_scale_inv"))
                    params[name].data.copy_(tensor.to(params[name].dtype))
                    loaded.add(name)
        missing = set(params) - loaded
        assert not missing, sorted(missing)[:10]
        print(f"language model loaded: {len(loaded)} tensors, {(time.time() - start) / 60:.1f} min, "
              f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    loras = []
    for name, module in list(model.language_model.named_modules()):
        for child, sub in list(module.named_children()):
            if child in TARGETS and isinstance(sub, nn.Linear):
                setattr(module, child, LoRA(sub))
                loras.append(getattr(module, child))
    model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, loras


def prompt_text(processor, tools_row: np.ndarray) -> str:
    if len(tools_row) == 0:  # NOTOOLS
        text = "Low-dose CT slice. Rate the overall image quality from 0 (worst) to 4 (best)."
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    numbers = ", ".join(f"{n} {min(99, int(round(v * 100))):02d}" for n, v in zip(TOOL_NAMES, tools_row))
    text = ("Low-dose CT slice. Image-quality tools compared it with 1000 reference slices; each number is the percent "
            "of reference slices that measure higher (noise measures: higher number = less noise than usual; "
            f"gradient_p95 = structure, streak_anisotropy = streaks): {numbers}. "
            "Rate the overall image quality from 0 (worst) to 4 (best).")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)


def llm_forward(model, enc, llm_head, processor, images, tools, features=False):
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    inputs = processor(text=[prompt_text(processor, t) for t in tools], images=images, return_tensors="pt")
    assert bool(inputs["attention_mask"].all()), "prompts differ in length"
    grid = inputs["image_grid_thw"].to(DEVICE)
    with autocast():
        with torch.no_grad():
            tokens = enc(inputs["pixel_values"].to(DEVICE), grid_thw=grid).last_hidden_state
        merged = enc.merger(tokens)
        sizes = (grid.prod(-1) // enc.spatial_merge_size ** 2).tolist()
        vision = {"image": BaseModelOutputWithPooling(pooler_output=torch.split(merged, sizes))}
        extra = {k: v.to(DEVICE) for k, v in inputs.items() if k not in ("pixel_values", "input_ids")}
        out = model(input_ids=inputs["input_ids"].to(DEVICE), mm_encoder_outputs=vision, use_cache=False, **extra)
    hidden = out.last_hidden_state[:, -1].float()
    return hidden if features else llm_head(hidden).squeeze(1)


def llm_predict(model, enc, llm_head, processor, images, tools, view=None, features=False) -> np.ndarray:
    model.eval(); enc.eval(); llm_head.eval()
    preds = []
    with torch.inference_mode():
        for s in range(0, len(images), BATCH):
            preds.append(llm_forward(model, enc, llm_head, processor, crops(images[s:s + BATCH], False, None, view=view),
                                     tools[s:s + BATCH], features).float().cpu().numpy())
    return np.concatenate(preds)


def phase2(k, model, loras, enc, processor, data, test, tr, va, mean, std, rng):
    from scipy.stats import pearsonr, spearmanr
    epochs, cap = (1, 2) if SMOKE else (LLM_EPOCHS, 12 if PROBE else None)
    enc.requires_grad_(False)
    enc.merger.requires_grad_(True)
    enc.eval()
    for lora in loras:
        lora.reset()
    llm_head = nn.Linear(model.config.text_config.hidden_size, 1).to(DEVICE)
    nn.init.zeros_(llm_head.weight); nn.init.zeros_(llm_head.bias)  # the probe's small random init started at loss 47
    lora_params = [p for lora in loras for p in (lora.a, lora.b)]
    params = [{"params": lora_params, "lr": LR_LORA}, {"params": list(enc.merger.parameters()), "lr": LR_MERGER},
              {"params": llm_head.parameters(), "lr": LR_LLM_HEAD}]
    optim = torch.optim.AdamW(params, weight_decay=0.0)
    per_epoch = cap or int(np.ceil(len(tr) * (2 if PSEUDO else 1) / BATCH))
    sched = schedule(optim, per_epoch * epochs)
    start = time.time()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train(); llm_head.train()
        order = rng.permutation(np.concatenate([tr, rng.choice(POOL, len(tr), replace=False)]) if PSEUDO else tr)
        losses = []
        for step in range(per_epoch):
            idx = order[step * BATCH:(step + 1) * BATCH]
            target = torch.as_tensor((data["mos"][idx] - mean) / std, device=DEVICE)
            pred = llm_forward(model, enc, llm_head, processor, crops(data["images"][idx], True, rng), data["tools"][idx])
            loss = objective(pred, target)
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
            optim.step(); sched.step()
            losses.append(loss.item())
            if PROBE and step in (1, per_epoch - 1):
                torch.cuda.synchronize()
                print(f"probe step {step + 1}: {(time.time() - start) / (step + 1):.2f} s/step, loss {loss.item():.3f}, "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        if PROBE:
            t = time.time()
            llm_predict(model, enc, llm_head, processor, data["images"][va[:16]], data["tools"][va[:16]])
            print(f"probe predict: {(time.time() - t) / 16:.2f} s/slice", flush=True)
            return None
        val = llm_predict(model, enc, llm_head, processor, data["images"][va], data["tools"][va]) * std + mean
        r = abs(pearsonr(val, data["mos"][va])[0]) + abs(spearmanr(val, data["mos"][va])[0])
        log(f"{UNIT} {k} llm epoch {epoch + 1}: loss {np.mean(losses):.4f}, val PLCC+SROCC {r:.3f}, "
            f"{(time.time() - start) / 60:.1f} min")
    out = {"llm_val": val, "llm_test": llm_predict(model, enc, llm_head, processor, test["images"], test["tools"]) * std + mean}
    parts = (("val", data["images"][va], data["tools"][va]), ("test", test["images"], test["tools"]))
    if PAIRS:
        for part, images, tools in parts:
            out[f"llm_tta_{part}"] = np.mean([llm_predict(model, enc, llm_head, processor, images, tools, v)
                                              for v in TTA_VIEWS], axis=0) * std + mean
    for part, images, tools in (("tr", data["images"][tr], data["tools"][tr]), *parts):  # after training, no RNG used
        out[f"feat_{part}"] = llm_predict(model, enc, llm_head, processor, images, tools, features=True).astype(np.float16)
    out["tr"] = tr
    log(f"{UNIT} {k} tta and features done, {(time.time() - start) / 60:.1f} min")
    out["_llm_head"] = {k: v.detach().cpu() for k, v in llm_head.state_dict().items()}
    del llm_head, optim
    return out


def save_weights(path, model, enc, vision_head, llm_head) -> None:
    """Everything a fold trained, in bf16 (the 27B base is the public FP8 checkpoint): the encoder with its merger,
    both heads and the LoRA matrices by module name."""
    lora = {name: {"a": m.a.detach().to("cpu", torch.bfloat16), "b": m.b.detach().to("cpu", torch.bfloat16)}
            for name, m in model.language_model.named_modules() if isinstance(m, LoRA)}
    torch.save({"encoder": {k: v.detach().to("cpu", torch.bfloat16) for k, v in enc.state_dict().items()},
                "vision_head": vision_head, "llm_head": llm_head, "lora": lora,
                "lora_config": {"rank": RANK, "alpha": ALPHA, "targets": TARGETS}, "repo": REPO_ID}, path)
    log(f"weights saved: {path.name}, {path.stat().st_size / 2**30:.2f} GiB")


def main() -> None:
    from transformers import AutoImageProcessor, AutoProcessor
    OUT.mkdir(parents=True, exist_ok=True)
    bits = "16" if WINDOWS3 else ""
    z, t = np.load(ROOT / f"train{bits}.npz"), np.load(ROOT / f"test{bits}.npz")
    data = {k: z[k] for k in ("ids", "mos", "images", "tools", "fold")}
    test = {k: t[k] for k in ("ids", "images", "tools")}
    if NOTOOLS:
        data["tools"], test["tools"] = (np.zeros((len(d["ids"]), 0), np.float32) for d in (data, test))
    if PATIENT:
        data["fold"] = np.load(ROOT / "patient_folds.npy")
        assert data["fold"].shape == data["mos"].shape and set(data["fold"]) == {1, 2, 3, 4, 5}
    if PSEUDO:
        global POOL
        pool, labels = np.load(ROOT / "pseudo.npz", allow_pickle=True), np.load(ROOT / "pseudo_labels.npz", allow_pickle=True)
        assert list(labels["ids"]) == list(pool["ids"])
        POOL = np.arange(len(data["mos"]), len(data["mos"]) + len(pool["ids"]))
        data["images"] = np.concatenate([data["images"], pool["images"]])
        data["tools"] = np.concatenate([data["tools"], pool["tools"]])
        data["mos"] = np.concatenate([data["mos"].astype(np.float64), np.full(len(POOL), np.nan)])
    units = range(1, SEEDS + 1) if FULL else range(1, 6)
    folds = [1] if SMOKE or PROBE else [k for k in units if not (OUT / f"{UNIT}_{k}.npz").exists()]
    if folds:
        config = configs()
        image_processor = AutoImageProcessor.from_pretrained(ROOT)
        processor = AutoProcessor.from_pretrained(REPO_ID)
        pretrained = pretrain(config, image_processor) if SYNTH else None
        model, loras = (None, None) if PATIENT else language_model(config)
    for k in folds:
        seed = SEED + 1000 * FULL + 10000 * (SEED_RUN - 1) + k
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        if FULL:
            tr = np.arange(len(data["mos"]))
            va = tr[:16]
        else:
            tr, va = np.nonzero(data["fold"] != k)[0], np.nonzero(data["fold"] == k)[0]
        fold_test = test
        if SMOKE:
            tr, va, fold_test = tr[:4], va[:3], {key: v[:2] for key, v in test.items()}
        if PSEUDO:
            data["mos"][POOL] = labels[f"label_{k}"][:len(POOL)]
        mean, std = data["mos"][tr].mean(), data["mos"][tr].std()
        enc = encoder(config)
        if pretrained is not None:
            enc.load_state_dict(pretrained)
        enc = enc.to(DEVICE)
        result = {"va": va}
        if not PROBE and not P2ONLY:
            result.update(phase1(k, enc, image_processor, data, fold_test, tr, va, mean, std, rng))
        if PATIENT:
            np.savez(OUT / f"{UNIT}_{k}.npz", **{key: v for key, v in result.items() if not key.startswith("_")})
            log(f"{UNIT} {k} saved")
            del enc
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            continue
        model.visual = enc
        rng = np.random.default_rng(seed + 100)
        out = phase2(k, model, loras, enc, processor, data, fold_test, tr, va, mean, std, rng)
        if PROBE:
            print("PROBE_DONE", flush=True)
            return
        result.update(out)
        save_weights(OUT / f"weights_{UNIT}_{k}.pt", model, enc, result.pop("_vision_head", None), result.pop("_llm_head"))
        np.savez(OUT / f"{UNIT}_{k}.npz", **result)
        log(f"{UNIT} {k} saved")
        model.visual = None
        del enc
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    print("SMOKE_OK" if SMOKE else "LLM_DONE", flush=True)


if __name__ == "__main__":  # colab exec runs it; colab/vm/teacher.py imports the pieces
    main()
