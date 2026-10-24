"""E11 (issue #329): Q-Align retrained on our 5 grouped folds with the official code (github.com/Q-Future/Q-Align at
fd0c3de) and its own per-dataset LoRA recipe (scripts/csiq-5split-lora.sh: start from q-future/one-align, LoRA on the
language model, visual abstractor lr 2e-5, vision model not frozen, 5 epochs, batch 32, lr 2e-4, cosine, warm-up 3%,
bf16, gradient checkpointing). Run on one GPU without DeepSpeed (the script's zero3 config only shards across GPUs).

MOS -> its 5 text levels by its rule (5 equal intervals of the score range [0, 4], right-closed: (0, 0.8] bad,
(0.8, 1.6] poor, (1.6, 2.4] fair, (2.4, 3.2] good, (3.2, 4] excellent; 0 -> bad), in its training format ("How would you
rate the quality of this image?\\n<|image|>" -> "The quality of the image is <level>."). Score: its evaluation
(q_align/evaluate/iqa_eval_lora_split.py: the base + LoRA model, prompt + " The quality of the image is", last-token
logits) with its scorer's probability weighting: softmax over [excellent, good, fair, poor, bad] times [4, 3, 2, 1, 0]
(its weights 1, 0.75, 0.5, 0.25, 0, scaled to the MOS range).

modes (first argument): data -> images (uint8 slices of train / test / lab329 as RGB PNG) and per-fold JSONs under
/content/qa/data; eval <k> -> /content/qa/fold_<k>.npz (va, val_logits, test_logits, lab_logits: the 5 level logits).
Runs inside the Q-Align environment (colab/vm/qalign329.sh).
"""
import json
import os
import sys

import numpy as np
from PIL import Image

W = "/content/qa"
DATA = os.path.join(W, "data")
LEVELS = ("excellent", "good", "fair", "poor", "bad")
PROMPT = "How would you rate the quality of this image?"


def level(mos: float) -> str:
    i = int(min(5, max(1, np.ceil(round(mos, 6) / 0.8))))  # 1 = bad ... 5 = excellent
    return ("bad", "poor", "fair", "good", "excellent")[i - 1]


def data() -> None:
    z, t, lab = (np.load(f"/content/vft/{n}.npz") for n in ("train", "test", "lab329"))
    os.makedirs(os.path.join(DATA, "ldct"), exist_ok=True)
    for prefix, ids, images in (("", z["ids"], z["images"]), ("", t["ids"], t["images"]),
                                ("lab_", np.arange(len(lab["images"])), lab["images"])):
        for i, im in zip(ids, images):
            path = os.path.join(DATA, "ldct", f"{prefix}{i}.png")
            if not os.path.exists(path):
                Image.fromarray(np.repeat(im[:, :, None], 3, 2)).save(path)
    mos = z["mos"].astype(np.float64)
    for k in range(1, 6):
        items = [{"id": f"ldct/{i}.png->{m:.4f}", "image": f"ldct/{i}.png",
                  "conversations": [{"from": "human", "value": PROMPT + "\n<|image|>"},
                                    {"from": "gpt", "value": f"The quality of the image is {level(m)}."}],
                  "gt_score": float(m)} for i, m, f in zip(z["ids"], mos, z["fold"]) if f != k]
        with open(os.path.join(DATA, f"train_split_{k}.json"), "w") as f:
            json.dump(items, f)
    print("data ok", len(os.listdir(os.path.join(DATA, "ldct"))), "images", flush=True)


def evaluate(k: int) -> None:
    import torch
    from q_align.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from q_align.conversation import conv_templates
    from q_align.mm_utils import tokenizer_image_token
    from q_align.model.builder import load_pretrained_model
    import q_align.model.modeling_llama2 as llama2  # its SDPA path calls a function it never imports
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
    llama2._prepare_4d_causal_attention_mask_for_sdpa = _prepare_4d_causal_attention_mask_for_sdpa
    path = os.path.join(W, "Q-Align", f"q-align-ldct-lora-{k}")
    from huggingface_hub import hf_hub_download  # the hub LoRA repos carry the base's image processor config; ours too
    import shutil
    shutil.copy(hf_hub_download("q-future/one-align", "preprocessor_config.json"), os.path.join(path, "preprocessor_config.json"))
    tokenizer, model, processor, _ = load_pretrained_model(path, "q-future/one-align", f"q-align-ldct-lora-{k}", device="cuda:0")
    conv = conv_templates["mplug_owl2"].copy()
    conv.append_message(conv.roles[0], PROMPT + "\n" + DEFAULT_IMAGE_TOKEN)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " The quality of the image is"
    ids = [tokenizer(t)["input_ids"][1] for t in LEVELS]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda:0")
    z, t, lab = (np.load(f"/content/vft/{n}.npz") for n in ("train", "test", "lab329"))
    va = np.nonzero(z["fold"] == k)[0]
    sets = {"val": [f"ldct/{i}.png" for i in z["ids"][va]], "test": [f"ldct/{i}.png" for i in t["ids"]],
            "lab": [f"ldct/lab_{i}.png" for i in range(len(lab["images"]))]}
    out = {"va": va}
    for name, files in sets.items():
        logits = []
        for s in range(0, len(files), 16):
            images = [Image.open(os.path.join(DATA, f)).convert("RGB") for f in files[s:s + 16]]  # square: no padding needed
            tensor = processor.preprocess(images, return_tensors="pt")["pixel_values"].half().to("cuda:0")
            with torch.inference_mode():
                last = model(input_ids.repeat(len(images), 1), images=tensor)["logits"][:, -1]
            logits.append(last[:, ids].float().cpu().numpy())
        out[f"{name}_logits"] = np.concatenate(logits)
    np.savez(os.path.join(W, f"fold_{k}.npz"), **out)
    print(f"qalign eval fold {k} done", flush=True)


if __name__ == "__main__":
    data() if sys.argv[1] == "data" else evaluate(int(sys.argv[2]))
