"""E12 (issue #329): MedSigLIP-FiLM retrained at another seed with Tolga's own recipe, unchanged: his train.py with the
config.py defaults (22 epochs, checkpoint = lowest validation loss per fold), then his test.py on the 300 test images
(per-fold predictions in preds.csv). The issue asks for the same folds: his KFold takes random_state=seed, so the copy
of train.py run here fixes it at 42 (his folds) and only the training seed (set_seed) changes. Data: the original
challenge .tif files and JSONs (/content/ms/ldct.tar).
Usage: python medsig_seed.py <seed>; writes /content/ms/seed<seed>/{preds.csv, test_results.json, *_cv_summary.json},
prints MS_DONE.
"""
import os
import subprocess
import sys
import tarfile

SEED = int(sys.argv[1])
W = "/content/ms"
OUT = f"{W}/seed{SEED}"
TRAIN = f"{W}/data/train/LDCTIQAG2023_train"
TEST = f"{W}/data/test/LDCTIQAC_test"
os.environ["HF_TOKEN"] = open("/content/.hf_token_upload").read().strip()  # gated google/medsiglip-448; never printed
subprocess.run("pip install -q sentencepiece tifffile imageio pydicom 2>&1 | tail -1", shell=True)
train_py = f"{W}/repo/train.py"
code = open(train_py).read()
old = 'random_state=cfg["seed"]'
if old in code:
    open(train_py, "w").write(code.replace(old, "random_state=42"))
assert 'random_state=42' in open(train_py).read()
if not os.path.exists(TRAIN):
    tarfile.open(f"{W}/ldct.tar").extractall(f"{W}/data")


def run(args: list) -> None:
    subprocess.run([sys.executable, *args], cwd=f"{W}/repo", check=True)


run(["train.py", "--json", f"{TRAIN}/train.json", "--data-dir", f"{TRAIN}/image", "--checkpoint-dir", OUT,
     "--exp-name", "medsiglip_iqa", "--seed", str(SEED), "--num-workers", "4"])
run(["test.py", "--ckpt-dir", OUT, "--exp-name", "medsiglip_iqa", "--test-json", f"{TEST}/test.json",
     "--test-dir", f"{TEST}/images", "--save-preds", f"{OUT}/preds.csv", "--num-workers", "4"])
print("MS_DONE", flush=True)
