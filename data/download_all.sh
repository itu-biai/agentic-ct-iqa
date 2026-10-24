#!/usr/bin/env bash
# Tum veri setlerini indirir (idempotent). Kullanim: bash data/download_all.sh
set -u
cd "$(dirname "$0")"
PY=../.venv/Scripts/python
dl_drive () { # $1=file id  $2=output
  [ -s "$2" ] && { echo "var: $2"; return; }
  $PY -m gdown "https://drive.google.com/uc?id=$1" -O "$2" || echo "HATA: $2"
}
[ -s LDCTIQAG2023_train.zip ] || curl -L --retry 5 -o LDCTIQAG2023_train.zip "https://zenodo.org/records/7833096/files/LDCTIQAG2023_train.zip?download=1"
dl_drive 1DbZfTj4it-3QRTuJO73r1iQyHLRwPS6H ldctiqa2023_test.zip
dl_drive 1B52WX42_yZEF8qIZ1mY3nQWgsy-Yloam dataset_biailab_IQA_2026v1.zip
dl_drive 1pIUT5beAkp17WCxyGPHZfHy4kEiJtD4y dataset_biailab_I2I_2025v1.zip
ls -la *.zip
echo "BITTI"
