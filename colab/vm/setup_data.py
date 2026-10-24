"""Start the data download on the VM as a background job (returns immediately). Log: /content/logs/setup_data.log"""
import os
import subprocess

os.makedirs("/content/data", exist_ok=True)
os.makedirs("/content/logs", exist_ok=True)
script = """
set -x
pip -q install gdown
cd /content/data
curl -sL -o train.zip 'https://zenodo.org/records/7833096/files/LDCTIQAG2023_train.zip?download=1'
unzip -q -o train.zip && rm -rf __MACOSX train.zip
python -m gdown -q 'https://drive.google.com/uc?id=1DbZfTj4it-3QRTuJO73r1iQyHLRwPS6H' -O test.zip
unzip -q -o test.zip && rm -rf __MACOSX test.zip
ls /content/data/LDCTIQAG2023_train/image | wc -l
ls /content/data/LDCTIQAC_test/images | wc -l
echo SETUP_DONE
"""
open("/content/setup_data.sh", "w").write(script)
subprocess.Popen("nohup bash /content/setup_data.sh > /content/logs/setup_data.log 2>&1 &", shell=True)
print("setup started")
