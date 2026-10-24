import glob, hashlib, os, subprocess, time
# Run Tolga's own test.py (itu-biai/medsiglip-ldct-iqa, commit ec00ad1) unchanged on his 5 shared fold checkpoints
# (Drive folder of colab/vm/tolga_job.sh) and the official 300 test images; per-image predictions -> /content/tolga/preds.csv.
W = "/content/tolga"
os.chdir(W)
with open("test.zip", "wb") as out:
    for p in sorted(glob.glob("test.zip.part*")):
        out.write(open(p, "rb").read()); os.remove(p)
md5 = hashlib.md5(open("test.zip", "rb").read()).hexdigest()
assert md5 == "0c47d0e1e5655b28f1943a4c399d2b50", md5
subprocess.run("unzip -q -o test.zip -d data", shell=True, check=True)
token = open("/content/.hf_token_upload").read().strip()  # gated google/medsiglip-448; never printed
os.environ["HF_TOKEN"] = token
t = time.time()
r = subprocess.run("pip install -q gdown sentencepiece tifffile imageio pydicom 2>&1 | tail -2; "
                   "python -m gdown -q --folder https://drive.google.com/drive/folders/1ezXhLMT6kA0JV_vftoBYebm7mOOu8rX5 -O ckpt; ls ckpt",
                   shell=True, capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-800:], "setup s", round(time.time() - t), flush=True)
r = subprocess.run("cd repo && python test.py --ckpt-dir ../ckpt --exp-name medsiglip_iqa "
                   "--test-json ../data/LDCTIQAC_test/test.json --test-dir ../data/LDCTIQAC_test/images "
                   "--out ../results.json --save-preds ../preds.csv --num-workers 2",
                   shell=True, capture_output=True, text=True, env=os.environ)
print(r.stdout[-3000:], r.stderr[-2000:].replace(token, "***"), flush=True)
print("TOLGA_TEST_DONE", flush=True)
