"""E7 (issue #329): the LDCT-IQA challenge winner's code (github.com/agaldran/ldct_iqa at 7f39d61, unchanged) retrained on
our 5 grouped folds with its own recipe as its README gives it: Swin-T, 4 heads, RPS overall loss (hypar 10), 10 cycles
of 4 epochs, 512 px, batch 8, NAdam 1e-4, seed = fold, targets 5 x MOS -> 21 classes, the checkpoint of the cycle with
the best validation AUC (its own selection, on the fold's validation slices). The README trains a ResNeXt-50 set as well
and the test-time ensembling of the submission is not in the repository; the issue asks for one 5-fold ensemble, so
this is the Swin set only.

Per fold, after training: the saved model in eval mode (mean of the 4 heads' softmax, the repository's own inference
output) on the fold's validation slices, the 300 test slices and the 2496 lab images -> probabilities over the 21
classes. Expected value sum_c p_c c / 5 = "as is"; argmax class / 5 = "grid". Images are the uint8 slices of
train.npz / test.npz / lab329.npz written as PNG (the repository reads tif floats through img_as_ubyte).

Expects /content/agal/repo (the repository), /content/vft/{train,test,lab329}.npz. Writes /content/agal/fold_<k>.npz
(va, val_probs, test_probs, lab_probs) and prints AGAL_DONE. Run as a plain process (nohup).
"""
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

W = "/content/agal"
REPO = os.path.join(W, "repo")
IMAGES = os.path.join(W, "images")
os.makedirs(IMAGES, exist_ok=True)
os.makedirs(os.path.join(REPO, "data"), exist_ok=True)
z, t, lab = (np.load(f"/content/vft/{n}.npz") for n in ("train", "test", "lab329"))
names = {}
for prefix, ids, images in (("", z["ids"], z["images"]), ("", t["ids"], t["images"]),
                            ("lab_", np.arange(len(lab["images"])), lab["images"])):
    for i, im in zip(ids, images):
        name = f"{prefix}{i}.png"
        names.setdefault(prefix or "ldct", []).append(name)
        if not os.path.exists(os.path.join(IMAGES, name)):
            Image.fromarray(im).save(os.path.join(IMAGES, name))
mos = np.round(z["mos"].astype(np.float64) * 5) / 5
fold = z["fold"]
train_names = [f"{i}.png" for i in z["ids"]]
test_csv, lab_csv = os.path.join(REPO, "data/test329.csv"), os.path.join(REPO, "data/lab329.csv")
pd.DataFrame({"image_id": [f"{i}.png" for i in t["ids"]], "score": 0.0}).to_csv(test_csv, index=False)
pd.DataFrame({"image_id": [f"lab_{i}.png" for i in range(len(lab["images"]))], "score": 0.0}).to_csv(lab_csv, index=False)
# One change to the repository: its validation AUC one-hot encodes with the number of classes present in the fold, which
# fails when a validation fold lacks a class (our fold 3 has 20 of the 21); the AUC is now the mean over the classes
# present, with the full 21-column encoding. Nothing else is touched.
evaluation = os.path.join(REPO, "utils/evaluation.py")
code = open(evaluation).read()
if "PATCHED_329" not in code:
    for old, new in (("get_one_hot_np(np.array(y_true), len(present_classes))", "get_one_hot_np(np.array(y_true), y_proba.shape[1])"),
                     ("get_one_hot_np(np.array(y_pred), len(present_classes))", "get_one_hot_np(np.array(y_pred), y_proba.shape[1])"),
                     ("for i in range(len(present_classes))]", "for i in present_classes]  # PATCHED_329"),
                     ("mean_auc = roc_auc_score(y_true, y_proba, multi_class='ovr')", "mean_auc = np.mean(auc_all)")):
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    open(evaluation, "w").write(code)
sys.path.insert(0, REPO)
from utils.data_handling import get_class_test_loader  # noqa: E402
from utils.get_model import get_arch  # noqa: E402


def probs(model, csv) -> np.ndarray:
    loader = get_class_test_loader(csv, IMAGES, (512, 512), 16, 8)
    out = []
    with torch.inference_mode():
        for x, _ in loader:
            out.append(model(x.cuda()).float().cpu().numpy())
    return np.concatenate(out)


start = time.time()
for k in range(1, 6):
    if os.path.exists(os.path.join(W, f"fold_{k}.npz")):
        continue
    tr, va = fold != k, fold == k
    pd.DataFrame({"image_id": np.array(train_names)[tr], "score": mos[tr]}).to_csv(os.path.join(REPO, f"data/train_f{k}.csv"), index=False)
    pd.DataFrame({"image_id": np.array(train_names)[va], "score": mos[va]}).to_csv(os.path.join(REPO, f"data/val_f{k}.csv"), index=False)
    with open(os.path.join(W, f"train_f{k}.log"), "w") as log:
        subprocess.run([sys.executable, "train_PMH.py", "--csv_train", f"data/train_f{k}.csv", "--data_path", IMAGES,
                        "--model", "swin", "--n_heads", "4", "--overall_loss", "rps", "--hypar", "10",
                        "--save_path", f"swin/f{k}", "--seed", str(k), "--cycle_lens", "10/4", "--num_workers", "8"],
                       cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    model = get_arch("swin", 21, 4)
    model.load_state_dict(torch.load(os.path.join(REPO, f"experiments/swin/f{k}/model_checkpoint.pth"),
                                     map_location="cpu")["model_state_dict"])
    model = model.cuda().eval()
    np.savez(os.path.join(W, f"fold_{k}.npz"), va=np.nonzero(va)[0], val_probs=probs(model, os.path.join(REPO, f"data/val_f{k}.csv")),
             test_probs=probs(model, test_csv), lab_probs=probs(model, lab_csv))
    print(f"agaldran fold {k} done, {(time.time() - start) / 60:.1f} min", flush=True)
    del model
    torch.cuda.empty_cache()
print("AGAL_DONE", flush=True)
