import os, zipfile, json, subprocess, time
zipfile.ZipFile("/content/q4/images.zip").extractall("/content/q4/images")
zipfile.ZipFile("/content/q4/sandboxes.zip").extractall("/content/q4/sb")
print("images", len(os.listdir("/content/q4/images")), "prompts", len(json.load(open("/content/q4/prompts.json"))),
      "sb", sorted(os.listdir("/content/q4/sb")), len(os.listdir("/content/q4/sb/setup2_vision_tools/images")))
t = time.time()
r = subprocess.run("pip install -q -U vllm 2>&1 | grep -v 'requires\|incompatible' | tail -3; pip show vllm | grep Version; "
                   "npm install -g opencode-ai@2.0.26 2>&1 | tail -2; which opencode && opencode --version; node --version",
                   shell=True, capture_output=True, text=True)
print(r.stdout[-2500:], r.stderr[-1500:], "seconds", round(time.time() - t))
