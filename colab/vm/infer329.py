"""Issue #329 inference on the VM, no training (scripts/issue329_eval.py scores the outputs). Every task writes
ROOT/i329/<task>.npz and is skipped when that file exists; prints <TASK>_DONE, then INFER_DONE.

Stage B = the rerun weights ROOT/weights_fold_<k>.pt (scripts/qwen_vision_ft.py rerun, test llm 2.7345), reloaded as in
colab/vm/teacher.py; every prediction is the llm head unless named vision. Tool columns (TOOL_NAMES of vftb_train.py):
0 residual_std, 1 gradient_deviation_mean, 2 hf_residual_mean, 3 noise_score, 4 gradient_p95, 5 streak_anisotropy;
groups as fixed in configs/tool_groups.py: noise (0, 3), sharpness (1, 4), hf (2), streaks (5).
  shuffle  E3 + E17: the 300 test images, tool rows permuted across images, either all six columns together ("all", E3)
           or one group's columns (E17, the same permutation for the group's columns); permutation seeds 0-4
           (default_rng(42 + s)), the same permutation for every fold, so each seed is a 5-fold ensemble. Also the
           unshuffled prediction ("base", must match the saved llm_test). Keys <cond>_s<s>_<k>, base_<k>.
  labshuf  E3 on the lab IQA test (lab329.npz: 192 slices x 13 images): all six columns permuted across the 2496 images,
           seed 0 only, 5 folds. Keys llm_<k>.
  pd       E17 partial dependence, fold 1 only: each group's columns set to p / 100, p = 0, 10, ..., 100 for every test
           image (the prompt prints at most 99), everything else kept. Keys <group>_<p>.
  occl     E18: the 4 cases in lab329.npz (occl_images, occl_tools): 48 x 48 patches with stride 24 over the 512 slice
           (20 x 20 positions), each filled with its own mean, then the usual centre crop; 5 folds. Keys base_<k>
           (cases), map_<k> (cases x 20 x 20, occluded prediction).
  enc      E19: mean over patch tokens of the encoder's last hidden state (centre 480 crop, as in training), for the
           pretrained encoder and the fold-1 tuned encoder, on the 1000 training and 300 test slices.
  probe    T1 (frozen encoder + ridge): the qwen_probe features (blocks 8, 17, 26 mean and std, merger mean; full 512
           slice as RGB, float32) of the pretrained encoder on the first 50 training slices (to check against the local
           features.npz) and the 2496 lab images.
  lab_<run> E2 denoiser tau for the trained ablations: ROOT/out_<run>/weights_fold_<k>.pt (notools, p2only) on the lab
           images, llm and (if it has one) vision head, 5 folds. Keys llm_<k>, vision_<k>.
  cnn      E2: ResNeXt-50 (ROOT/cnn/weights_fold_<k>.pt, colab/vm/cnn_train.py) on the lab images, 4 flip views.
Resumable: a task's partial file (<task>.partial.npz, written after every fold) is loaded and finished folds are
skipped. No sys.argv (`colab exec`); VFT_ROOT and VFT_SMOKE=1 for a local check.
"""
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(os.environ.get("VFT_ROOT", "/content/vft"))
OUT = ROOT / "i329"
vft = {"__name__": "vftb_train"}
exec(compile((ROOT / "vftb_train.py").read_text(), str(ROOT / "vftb_train.py"), "exec"), vft)
SMOKE, DEVICE = vft["SMOKE"], vft["DEVICE"]
FOLDS = [1] if SMOKE else [1, 2, 3, 4, 5]
vft["BATCH"] = 2 if SMOKE else 16  # inference only
GROUPS = {"noise": (0, 3), "sharpness": (1, 4), "hf": (2,), "streaks": (5,)}
CONDITIONS = {"all": (0, 1, 2, 3, 4, 5), **GROUPS}
SEEDS = 5
PATCH, STRIDE = 48, 24
PROBE_LAYERS = (0, 1) if SMOKE else (8, 17, 26)  # the tiny smoke encoder has 2 blocks
START = time.time()


def say(text: str) -> None:
    print(f"{text}, {(time.time() - START) / 60:.1f} min", flush=True)


def cut(a: np.ndarray, n: int = 3) -> np.ndarray:
    return a[:n] if SMOKE else a


class Stage:
    """The 27B language model, loaded once; fold weights swapped in."""

    def __init__(self):
        from transformers import AutoImageProcessor, AutoProcessor
        self.config = vft["configs"]()
        self.image_processor = AutoImageProcessor.from_pretrained(ROOT)
        self.processor = AutoProcessor.from_pretrained(vft["REPO_ID"])
        self.model, _ = vft["language_model"](self.config)
        self.loras = {n: m for n, m in self.model.language_model.named_modules() if isinstance(m, vft["LoRA"])}
        self.z = np.load(ROOT / "train.npz")

    def load(self, path: Path, k: int, tools: int):
        w = torch.load(path, map_location="cpu")
        assert set(w["lora"]) == set(self.loras), "LoRA module names differ"
        enc = vft["encoder"](self.config)
        enc.load_state_dict({key: v.float() for key, v in w["encoder"].items()})
        self.enc = enc.to(DEVICE).eval()
        for name, m in self.loras.items():
            m.a.data.copy_(w["lora"][name]["a"].to(m.a.dtype))
            m.b.data.copy_(w["lora"][name]["b"].to(m.b.dtype))
        self.llm_head = nn.Linear(self.model.config.text_config.hidden_size, 1).to(DEVICE)
        self.llm_head.load_state_dict(w["llm_head"])
        self.head = None
        if w.get("vision_head") is not None:
            self.head = vft["Head"](self.enc.config.hidden_size, tools).to(DEVICE)
            self.head.load_state_dict(w["vision_head"])
        self.model.visual = self.enc
        tr = self.z["fold"] != k
        self.mean, self.std = self.z["mos"][tr].mean(), self.z["mos"][tr].std()

    def llm(self, images, tools) -> np.ndarray:
        return vft["llm_predict"](self.model, self.enc, self.llm_head, self.processor, images, tools) * self.std + self.mean

    def vision(self, images, tools) -> np.ndarray:
        return vft["vision_predict"](self.enc, self.head, self.image_processor, images, tools) * self.std + self.mean

    def free(self):
        self.model.visual = None
        del self.enc, self.head, self.llm_head
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


def done(task: str) -> bool:
    return (OUT / f"{task}.npz").exists()


def save(task: str, out: dict, final: bool = False) -> None:
    np.savez(OUT / (f"{task}.npz" if final else f"{task}.partial.npz"), **out)
    if final:
        (OUT / f"{task}.partial.npz").unlink(missing_ok=True)
        print(f"{task.upper()}_DONE", flush=True)


def partial(task: str) -> dict:
    path = OUT / f"{task}.partial.npz"
    return dict(np.load(path)) if path.exists() else {}


DONE_KEY = {"shuffle": "streaks_s4_{k}", "labshuf": "llm_{k}", "occl": "map_{k}"}


def permuted(tools: np.ndarray, cols: tuple, seed: int) -> np.ndarray:
    perm = np.random.default_rng(42 + seed).permutation(len(tools))
    out = tools.copy()
    out[:, list(cols)] = tools[perm][:, list(cols)]
    return out


def occluded(image: np.ndarray) -> np.ndarray:
    """All 20 x 20 occluded versions of one uint8 slice, row-major."""
    n = (image.shape[0] - PATCH) // STRIDE + 1
    out = []
    for i in range(n):
        for j in range(n):
            im = image.copy()
            y, x = i * STRIDE, j * STRIDE
            im[y:y + PATCH, x:x + PATCH] = int(round(image[y:y + PATCH, x:x + PATCH].mean()))
            out.append(im)
    return np.stack(out)


def stage_b_tasks(stage: Stage, test: dict, lab: dict) -> None:
    tasks = [t for t in ("shuffle", "labshuf", "occl") if not done(t)]
    pd_todo = not done("pd")
    if not tasks and not pd_todo:
        return
    out = {t: partial(t) for t in tasks}
    for k in FOLDS:
        todo = [t for t in tasks if DONE_KEY[t].format(k=k) not in out[t]]
        if not todo and not (pd_todo and k == 1):
            continue
        stage.load(ROOT / f"weights_fold_{k}.pt", k, 6)
        if "shuffle" in todo:
            o = out["shuffle"]
            o[f"base_{k}"] = stage.llm(test["images"], test["tools"])
            for cond, cols in CONDITIONS.items():
                for s in range(SEEDS):
                    o[f"{cond}_s{s}_{k}"] = stage.llm(test["images"], permuted(test["tools"], cols, s))
            save("shuffle", o, k == FOLDS[-1])
            say(f"shuffle fold {k}")
        if "labshuf" in todo:
            out["labshuf"][f"llm_{k}"] = stage.llm(lab["images"], permuted(lab["tools"], CONDITIONS["all"], 0))
            save("labshuf", out["labshuf"], k == FOLDS[-1])
            say(f"labshuf fold {k}")
        if "occl" in todo:
            o = out["occl"]
            o[f"base_{k}"] = stage.llm(lab["occl_images"], lab["occl_tools"])
            maps = []
            for img, tools in zip(lab["occl_images"], lab["occl_tools"]):
                variants = occluded(img)
                maps.append(stage.llm(variants, np.repeat(tools[None], len(variants), 0)).reshape(20, 20))
            o[f"map_{k}"] = np.stack(maps)
            save("occl", o, k == FOLDS[-1])
            say(f"occl fold {k}")
        if pd_todo and k == 1:
            o = {}
            for group, cols in GROUPS.items():
                for p in range(0, 101, 10):
                    tools = test["tools"].copy()
                    tools[:, list(cols)] = p / 100
                    o[f"{group}_{p}"] = stage.llm(test["images"], tools)
                say(f"pd {group}")
            save("pd", o, True)
        stage.free()
    for t in tasks:  # a resumed task whose last folds were already in its partial file
        if not done(t) and all(DONE_KEY[t].format(k=k) in out[t] for k in FOLDS):
            save(t, out[t], True)


def encoder_tasks(stage: Stage, train: dict, test: dict, lab: dict) -> None:
    from transformers import AutoImageProcessor
    if not done("enc"):
        out = {}
        for name in ("pretrained", "tuned"):
            if name == "pretrained":
                enc = vft["encoder"](stage.config).to(DEVICE).eval()
            else:
                w = torch.load(ROOT / "weights_fold_1.pt", map_location="cpu")
                enc = vft["encoder"](stage.config)
                enc.load_state_dict({key: v.float() for key, v in w["encoder"].items()})
                enc = enc.to(DEVICE).eval()
            for part, images in (("train", train["images"]), ("test", test["images"])):
                feats = []
                with torch.inference_mode():
                    for s in range(0, len(images), 16):
                        inputs = stage.image_processor(images=vft["crops"](images[s:s + 16], False, None), return_tensors="pt")
                        grid = inputs["image_grid_thw"].to(DEVICE)
                        with vft["autocast"]():
                            tokens = enc(inputs["pixel_values"].to(DEVICE), grid_thw=grid).last_hidden_state
                        feats += [t.float().mean(0).cpu().numpy() for t in torch.split(tokens, grid.prod(1).tolist())]
                out[f"{name}_{part}"] = np.stack(feats).astype(np.float16)
            del enc
        save("enc", out, True)
        say("enc")
    if not done("probe"):
        probe_processor = AutoImageProcessor.from_pretrained(ROOT)
        enc = vft["encoder"](stage.config).to(DEVICE).eval()
        captured = {}
        for layer in PROBE_LAYERS:
            enc.blocks[layer].register_forward_hook(lambda m, i, o, layer=layer: captured.__setitem__(layer, o))
        out = {}
        for part, images in (("train50", train["images"][:50]), ("lab", lab["images"])):
            rows = []
            with torch.inference_mode():
                for img in cut(images):
                    inputs = probe_processor(images=[np.repeat(img[:, :, None], 3, 2)], return_tensors="pt")
                    o = enc(inputs["pixel_values"].float().to(DEVICE), grid_thw=inputs["image_grid_thw"].to(DEVICE))
                    parts = []
                    for layer in PROBE_LAYERS:
                        tokens = captured[layer][0] if isinstance(captured[layer], tuple) else captured[layer]
                        parts += [tokens.mean(0), tokens.std(0)]
                    parts.append(o.pooler_output.mean(0))
                    rows.append(torch.cat(parts).float().cpu().numpy())
            out[part] = np.stack(rows).astype(np.float32)
        del enc
        save("probe", out, True)
        say("probe")


def ablation_tasks(stage: Stage, lab: dict) -> None:
    for run, tools in (("notools", 0), ("p2only", 6)):
        task = f"lab_{run}"
        if done(task) or not all((ROOT / f"out_{run}/weights_fold_{k}.pt").exists() for k in FOLDS):
            continue
        lab_tools = lab["tools"][:, :tools]
        out = partial(task)
        for k in FOLDS:
            if f"llm_{k}" in out:
                continue
            stage.load(ROOT / f"out_{run}/weights_fold_{k}.pt", k, tools)
            out[f"llm_{k}"] = stage.llm(lab["images"], lab_tools)
            if stage.head is not None:
                out[f"vision_{k}"] = stage.vision(lab["images"], lab_tools)
            stage.free()
            save(task, out, k == FOLDS[-1])
            say(f"{task} fold {k}")
        if not done(task) and all(f"llm_{k}" in out for k in FOLDS):
            save(task, out, True)


def cnn_task(lab: dict) -> None:
    if done("cnn") or not (ROOT / "cnn/weights_fold_1.pt").exists():
        return
    import timm
    cnn = {"__name__": "cnn_train"}
    exec(compile((ROOT / "cnn_train.py").read_text(), str(ROOT / "cnn_train.py"), "exec"), cnn)
    z, out = np.load(ROOT / "train.npz"), {}
    for k in FOLDS:
        model = timm.create_model(cnn["ARCH"], pretrained=False, num_classes=1).to(DEVICE)
        model.load_state_dict(torch.load(ROOT / f"cnn/weights_fold_{k}.pt", map_location=DEVICE))
        tr = z["fold"] != k
        out[f"cnn_{k}"] = cnn["predict"](model, lab["images"]) * z["mos"][tr].std() + z["mos"][tr].mean()
        del model
    save("cnn", out, True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z, t, lab = np.load(ROOT / "train.npz"), np.load(ROOT / "test.npz"), np.load(ROOT / "lab329.npz")
    train = {"images": cut(z["images"])}
    test = {key: cut(t[key]) for key in ("images", "tools")}
    lab = {key: cut(lab[key], 2) for key in lab.files if key != "ids"}
    stage = Stage()
    stage_b_tasks(stage, test, lab)
    encoder_tasks(stage, train, test, lab)
    ablation_tasks(stage, lab)
    cnn_task(lab)
    print("INFER_DONE", flush=True)


main()
