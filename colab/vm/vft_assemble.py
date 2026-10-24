import glob, hashlib, os
# Reassemble the 30 MB parts (colab upload drops large files) and check them against the local md5s.
EXPECTED = {"train.npz": "5a2c808f2b2569f19c53e5035dc10516", "test.npz": "b0e165a568024700148d19fcb3be5c09"}
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
print("ASSEMBLED_OK", flush=True)
