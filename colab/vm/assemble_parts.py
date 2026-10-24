import glob, hashlib, os
# Reassemble every file listed in manifest.txt ("<md5>  <relative path>", from md5sum) out of its 30 MB parts
# (<path>.partNN; colab upload drops large files) under /content, and check its md5. Prints ASSEMBLE_OK.
os.chdir("/content")
for line in open("/content/manifest.txt").read().split("\n"):
    if not line.strip():
        continue
    md5, name = line.split(maxsplit=1)
    parts = sorted(glob.glob(name + ".part*"))
    if parts:
        with open(name, "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    out.write(f.read())
        for p in parts:
            os.remove(p)
    h = hashlib.md5()
    with open(name, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    print(name, "OK" if h.hexdigest() == md5 else "MISMATCH", flush=True)
    assert h.hexdigest() == md5, name
print("ASSEMBLE_OK", flush=True)
