import glob, hashlib, os, subprocess, time
# Synthetic pre-training setup (colab/vm/vftb_train.py with ROOT/synth and ROOT/windows3), after colab/vm/vftb_setup.py:
# reassemble train16/test16 from their 30 MB parts (md5 checked), nibabel, and MSD Task09 Spleen (61 abdominal CT
# volumes, CC-BY-SA 4.0) from the MONAI mirror into /content/msd.
EXPECTED = {"train16.npz": "d593602490cb96d0554f28e7f8572120", "test16.npz": "3381d7c45fdc0e3efcb97ae412b69c75"}
os.chdir("/content/vft")
for name, md5 in EXPECTED.items():
    parts = sorted(glob.glob(name + ".part*"))
    if parts:
        with open(name, "wb") as out:
            for p in parts:
                out.write(open(p, "rb").read())
        for p in parts:
            os.remove(p)
    got = hashlib.md5(open(name, "rb").read()).hexdigest()
    print(name, got, "OK" if got == md5 else "MISMATCH", flush=True)
    assert got == md5, name
t = time.time()
r = subprocess.run("pip install -q nibabel 2>&1 | tail -1; mkdir -p /content/msd && cd /content/msd && "
                   "curl -s -o spleen.tar https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar && "
                   "tar -xf spleen.tar && rm spleen.tar && ls Task09_Spleen/imagesTr Task09_Spleen/imagesTs | grep -c 'spleen_'",
                   shell=True, capture_output=True, text=True)
print("msd volumes (incl. macOS ._ files):", r.stdout.strip()[-200:], r.stderr[-300:], "seconds", round(time.time() - t), flush=True)
print("SYNTH_SETUP_DONE", flush=True)
