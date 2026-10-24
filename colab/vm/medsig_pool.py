import glob, os, subprocess, time, zipfile
# MedSigLIP teacher for the unlabelled pool: Tolga's own predict.py (itu-biai/medsiglip-ldct-iqa, commit ec00ad1, as in
# colab/vm/tolga_test.py) with each of his 5 shared fold checkpoints on the pool PNGs (pool.zip from
# scripts/pseudo_pool.py) and, as a check, on 20 of the official test images (must match his preds.csv).
# Writes /content/tolga/pool_fold<k>.csv; prints MEDSIG_DONE. With /content/tolga/cross_flag it scores cross.zip
# (scripts/cross_pool.py) into cross_fold<k>.csv instead.
W = "/content/tolga"
os.makedirs(W, exist_ok=True)
os.chdir(W)
POOL = "cross" if os.path.exists("cross_flag") else "pool"
for name in (POOL + ".zip", "test.zip"):
    parts = sorted(glob.glob(name + ".part*"))
    if parts:
        with open(name, "wb") as out:
            for p in parts:
                out.write(open(p, "rb").read()); os.remove(p)
zipfile.ZipFile(POOL + ".zip").extractall(POOL)
zipfile.ZipFile("test.zip").extractall("data")
token = open("/content/.hf_token_upload").read().strip()  # gated google/medsiglip-448; never printed
os.environ["HF_TOKEN"] = token
t = time.time()
if not os.path.isdir("ckpt"):
    r = subprocess.run("pip install -q gdown sentencepiece tifffile imageio pydicom 2>&1 | tail -2; "
                       "python -m gdown -q --folder https://drive.google.com/drive/folders/1ezXhLMT6kA0JV_vftoBYebm7mOOu8rX5 -O ckpt; ls ckpt",
                       shell=True, capture_output=True, text=True)
    print(r.stdout[-800:], r.stderr[-500:].replace(token, "***"), "setup s", round(time.time() - t), flush=True)
pool = sorted(glob.glob(os.path.join(W, POOL, "*.png")))
check = sorted(glob.glob(os.path.join(W, "data", "LDCTIQAC_test", "images", "*")))[:20]
for k, ckpt in enumerate(sorted(glob.glob(os.path.join(W, "ckpt", "*.pth"))), 1):
    r = subprocess.run(["python", "predict.py", *pool, *check, "--ckpt", ckpt, "--out", os.path.join(W, f"{POOL}_fold{k}.csv")],
                       cwd=os.path.join(W, "repo"), capture_output=True, text=True, env=os.environ)
    print(k, os.path.basename(ckpt), "rows", r.stdout.count("\n"), r.stderr[-600:].replace(token, "***"), flush=True)
print("MEDSIG_DONE", round(time.time() - t), "s", flush=True)
