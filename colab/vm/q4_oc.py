import subprocess
r = subprocess.run("npm install -g @opencode/cli@2.0.26 2>&1 | tail -3; which opencode && opencode --version", shell=True, capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-1000:])
