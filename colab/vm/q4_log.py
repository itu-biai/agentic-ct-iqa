import subprocess;print(subprocess.run("tail -c 2500 /content/vllm.log; nvidia-smi --query-gpu=memory.used --format=csv",shell=True,capture_output=True,text=True).stdout)
