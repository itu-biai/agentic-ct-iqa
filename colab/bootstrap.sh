#!/usr/bin/env bash
# Recreate the expert VM from scratch: T4 session, code, stress inputs, HF token, LDCT-IQA data.
# Usage (Git Bash, project root): bash colab/bootstrap.sh
set -u
C="bash colab/c.sh"
S=expert
P="/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper"
SRC="/mnt/c/Users/kerem/AppData/Local/Temp/claude/C--Users-kerem-Desktop-KB-TD-Ortak-Paper/77134f49-c432-4b28-8d0f-4c22207347e5/scratchpad/medsiglip"
$C new -s $S --gpu T4 || exit 1
echo "import os; [os.makedirs(d, exist_ok=True) for d in ('/content/medsiglip', '/content/npzparts', '/content/logs')]" | $C exec -s $S
for f in config.py data.py model.py train.py test.py predict.py requirements.txt; do
  $C upload -s $S "$SRC/$f" "/content/medsiglip/$f" >/dev/null || echo "FAIL $f"
done
$C upload -s $S "$P/colab/expert_predict.py" /content/medsiglip/expert_predict.py >/dev/null || echo "FAIL expert_predict.py"
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "rm -rf /tmp/npzparts && mkdir /tmp/npzparts && split -b 20m -d '$P/colab/expert_inputs.npz' /tmp/npzparts/part_"
for p in $(MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "ls /tmp/npzparts/part_*" | tr -d '\r'); do
  $C upload -s $S "$p" "/content/npzparts/$(basename $p)" >/dev/null || echo "FAIL $p"
done
timeout 120 wsl.exe -d Ubuntu -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; colab upload -s expert ~/.hf_token /content/.hf_token_upload' >/dev/null || echo "FAIL token"
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "rm -rf /tmp/npzparts"
$C exec -s $S -f "$P/colab/vm/finish_upload.py"
$C exec -s $S -f "$P/colab/vm/setup_data.py"
echo "local md5: $(MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "md5sum '$P/colab/expert_inputs.npz'" | cut -d' ' -f1)"
