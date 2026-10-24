"""Pre-test before fine-tuning Qwen3.8-27B: do its image features hold quality information that MedSigLIP-FiLM misses?

Only the vision encoder (0.4B parameters, bf16 in Qwen's FP8 checkpoint) runs, on the CPU, on the 1000 training slices;
the test set is not touched. A linear probe (ridge, repeated 5-fold cross-validation) on pooled encoder features is
asked to predict MedSigLIP's out-of-fold error (MOS - expert) and the MOS itself. If the features cannot predict the
expert's error, fine-tuning the same model is unlikely to beat the expert.

    python -m scripts.qwen_probe weights    # vision tensors only (879 MiB, HTTP range) -> outputs/agentic_ct_iqa/probe/
    python -m scripts.qwen_probe features   # pooled encoder features of the 1000 training slices -> features.npz
    python -m scripts.qwen_probe probe      # cross-validated probe -> report.md
    python -m scripts.qwen_probe features test   # the same features for the 300 test slices -> features_test.npz
    python -m scripts.qwen_probe test       # single test look, everything fixed above -> test_report.md
"""
import json
import struct
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG
from src.data import load_ldctiqa_split
from src.experts import expert_scores, load_fold_predictions, strip_ext
from src.metrics import challenge_scores

REPO = "https://huggingface.co/Qwen/Qwen3.8-27B-FP8/resolve/main"
OUT = CONFIG.output_dir / "probe"
WEIGHTS = OUT / "qwen38_visual.safetensors"
FEATURES = OUT / "features.npz"
TEST_FEATURES = OUT / "features_test.npz"
LAYERS = (8, 17, 26)  # encoder blocks whose tokens are pooled (mean and std); plus the merger output (mean)
REPEATS = 5


def fetch(name: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = {"Range": f"bytes={start}-{end}"} if start is not None else {}
    return urllib.request.urlopen(urllib.request.Request(f"{REPO}/{name}", headers=headers), timeout=600).read()


def weights() -> None:
    """The visual tensors are one contiguous block of outside.safetensors; download that block and rewrite it as a
    safetensors file of its own."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "preprocessor_config.json"):
        (OUT / name).write_bytes(fetch(name))
    size = struct.unpack("<Q", fetch("outside.safetensors", 0, 7))[0]
    header = json.loads(fetch("outside.safetensors", 8, 8 + size - 1))
    visual = {k: v for k, v in header.items() if k.startswith("model.visual.")}
    first = min(v["data_offsets"][0] for v in visual.values())
    last = max(v["data_offsets"][1] for v in visual.values())
    block = fetch("outside.safetensors", 8 + size + first, 8 + size + last - 1)
    new = {k.removeprefix("model.visual."): {**v, "data_offsets": [v["data_offsets"][0] - first,
                                                                    v["data_offsets"][1] - first]}
           for k, v in visual.items()}
    raw = json.dumps(new).encode()
    WEIGHTS.write_bytes(struct.pack("<Q", len(raw)) + raw + block)
    print(len(new), "tensors,", round(len(block) / 2**20), "MiB ->", WEIGHTS)


def encoder():
    from safetensors.torch import load_file
    from transformers import AutoImageProcessor
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    config = Qwen3_5Config.from_pretrained(OUT).vision_config
    config._attn_implementation = "sdpa"
    model = Qwen3_5VisionModel(config)
    missing, unexpected = model.load_state_dict(load_file(WEIGHTS), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return model.float().eval(), AutoImageProcessor.from_pretrained(OUT)


def features(split: str = "train") -> None:
    """Per slice: mean and std over the patch tokens of blocks LAYERS, and the mean of the merger output (what the
    language model receives). The slice goes in as the sandbox PNG (same windowing), converted to RGB."""
    from scripts.make_sandboxes import png
    import io

    torch.set_num_threads(12)
    model, processor = encoder()
    captured = {}
    for layer in LAYERS:
        model.blocks[layer].register_forward_hook(lambda m, i, o, layer=layer: captured.__setitem__(layer, o))
    samples = (load_ldctiqa_split(CONFIG.train_images, CONFIG.train_labels) if split == "train"
               else load_ldctiqa_split(CONFIG.test_images, CONFIG.test_labels))
    target = FEATURES if split == "train" else TEST_FEATURES
    rows, start = [], time.time()
    with torch.inference_mode():
        for k, sample in enumerate(samples):
            image = Image.open(io.BytesIO(png(sample.image))).convert("RGB")
            inputs = processor(images=[image], return_tensors="pt")
            out = model(inputs["pixel_values"].float(), grid_thw=inputs["image_grid_thw"])
            parts = []
            for layer in LAYERS:
                tokens = captured[layer][0] if isinstance(captured[layer], tuple) else captured[layer]
                parts += [tokens.mean(0), tokens.std(0)]
            parts.append(out.pooler_output.mean(0))
            rows.append(torch.cat(parts).numpy())
            if k % 50 == 0:
                print(k, sample.image_id, tuple(inputs["image_grid_thw"][0].tolist()),
                      f"{(time.time() - start) / (k + 1):.1f} s/slice", flush=True)
    np.savez_compressed(target, ids=np.array([s.image_id for s in samples]), mos=np.array([s.mos for s in samples]),
                        x=np.stack(rows).astype(np.float32))
    print("features", np.stack(rows).shape, "->", target)


def cv_predict(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Out-of-fold ridge predictions; alpha by internal cross-validation on each training fold."""
    pred = np.empty(len(y))
    for train, held in KFold(5, shuffle=True, random_state=seed).split(x):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 6, 15)))
        pred[held] = model.fit(x[train], y[train]).predict(x[held])
    return pred


def overall(pred: np.ndarray, mos: np.ndarray) -> float:
    return challenge_scores(pred, mos)["overall"]


def bootstrap(a: np.ndarray, b: np.ndarray, mos: np.ndarray, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(CONFIG.seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(mos), len(mos))
        diffs.append(overall(a[i], mos[i]) - overall(b[i], mos[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return round(lo, 3), round(hi, 3)


def probe() -> None:
    data = np.load(FEATURES)
    ids, mos, x = data["ids"], data["mos"], data["x"].astype(np.float64)
    train = expert_scores(load_fold_predictions(CONFIG.expert_dir))["train"]
    train.index = strip_ext(train.index)
    expert = train.reindex(ids).to_numpy()
    tools = pd.read_csv(CONFIG.output_dir / "measurements_train.csv", dtype={"image_id": str}).set_index("image_id")
    tools = tools.drop(columns="mos").reindex(ids)
    tools = tools.fillna(tools.median()).to_numpy()
    residual = mos - expert
    shuffled = x[np.random.default_rng(CONFIG.seed).permutation(len(x))]
    methods = {"expert": [], "expert + tools residual": [], "expert + Qwen residual": [],
               "expert + shuffled Qwen residual": [], "Qwen alone": [], "expert/Qwen 50/50": []}
    r2 = {"tools": [], "Qwen": [], "shuffled Qwen": []}
    for seed in range(REPEATS):
        fixes = {"tools": cv_predict(tools, residual, seed), "Qwen": cv_predict(x, residual, seed),
                 "shuffled Qwen": cv_predict(shuffled, residual, seed)}
        for name, fix in fixes.items():
            r2[name].append(1 - np.mean((residual - fix) ** 2) / np.var(residual))
        alone = cv_predict(x, mos, seed)
        methods["expert"].append(expert)
        methods["expert + tools residual"].append(expert + fixes["tools"])
        methods["expert + Qwen residual"].append(expert + fixes["Qwen"])
        methods["expert + shuffled Qwen residual"].append(expert + fixes["shuffled Qwen"])
        methods["Qwen alone"].append(alone)
        methods["expert/Qwen 50/50"].append((expert + alone) / 2)
    rows = []
    for name, preds in methods.items():
        scores = [overall(p, mos) for p in preds]
        mean_pred = np.mean(preds, axis=0)
        lo, hi = bootstrap(mean_pred, expert, mos) if name != "expert" else ("", "")
        rows.append({"method": name, "overall": round(np.mean(scores), 4), "sd over repeats": round(np.std(scores), 4),
                     "minus expert, 95% CI": f"[{lo:+.3f}, {hi:+.3f}]" if lo != "" else ""})
    table = pd.DataFrame(rows).set_index("method")
    fit = {k: f"{np.mean(v):+.3f}" for k, v in r2.items()}
    corr = pearsonr(np.mean(methods["Qwen alone"], axis=0) - mos, expert - mos)[0]
    lines = ["# Qwen3.8-27B vision encoder probe (1000 training slices, no test images)", "",
             f"Features: mean and std of blocks {LAYERS} plus the merger output mean, {x.shape[1]} dimensions. Ridge, "
             f"{REPEATS} x 5-fold cross-validation; the expert score is MedSigLIP-FiLM's out-of-fold score.", "",
             table.to_markdown(), "",
             f"Cross-validated R^2 for the expert's error (MOS - expert): {fit}.",
             f"Correlation between the errors of 'Qwen alone' and the expert: {corr:.3f}.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def test() -> None:
    """The one test look, with everything fixed by the cross-validated probe: ridge (same pipeline and alpha grid) fit
    on all 1000 training slices, applied once to the 300 test slices. Primary: Qwen alone; secondary: 50/50 with the
    expert and expert + Qwen residual. The expert is the raw 5-fold-mean MedSigLIP-FiLM score (Table 1, raw)."""
    train, held = np.load(FEATURES), np.load(TEST_FEATURES)
    x, mos, xt, mos_t = train["x"].astype(np.float64), train["mos"], held["x"].astype(np.float64), held["mos"]
    scores = expert_scores(load_fold_predictions(CONFIG.expert_dir))
    for name in ("train", "test"):
        scores[name].index = strip_ext(scores[name].index)
    expert, expert_t = scores["train"].reindex(train["ids"]).to_numpy(), scores["test"].reindex(held["ids"]).to_numpy()
    head = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 6, 15)))
    alone_model, residual_model = head().fit(x, mos), head().fit(x, mos - expert)
    alone = alone_model.predict(xt)
    methods = {"MedSigLIP-FiLM (expert)": expert_t, "Qwen encoder + ridge (primary)": alone,
               "expert/Qwen 50/50": (expert_t + alone) / 2,
               "expert + Qwen residual": expert_t + residual_model.predict(xt)}
    rows = []
    for name, pred in methods.items():
        s = challenge_scores(pred, mos_t)
        lo, hi = bootstrap(pred, expert_t, mos_t) if pred is not expert_t else (np.nan, np.nan)
        rows.append({"method": name, **{k: round(v, 4) for k, v in s.items()},
                     "minus expert": round(s["overall"] - challenge_scores(expert_t, mos_t)["overall"], 4),
                     "95% CI": "" if np.isnan(lo) else f"[{lo:+.3f}, {hi:+.3f}]"})
    alphas = {"alone": alone_model[-1].alpha_, "residual": residual_model[-1].alpha_}
    lines = ["# Qwen3.8-27B vision encoder probe on the 300-image test (single look)", "",
             f"Ridge fit on all 1000 training slices (alphas chosen on training: {alphas}); paired bootstrap over test "
             "images for the difference to the expert. No logistic fitting (raw, as Table 1's overall_raw).", "",
             pd.DataFrame(rows).set_index("method").to_markdown(), "",
             "Reference: challenge #1 2.7427; MedSigLIP-FiLM in Tolga's paper 2.7436; setup 26 agent 2.51.", ""]
    (OUT / "test_report.md").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame({"test_id": held["ids"], "mos": mos_t, **{k: v for k, v in methods.items()}}).round(4).to_csv(
        OUT / "test_predictions.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    {"weights": weights, "features": lambda: features(*sys.argv[2:]), "probe": probe, "test": test}[sys.argv[1]]()
