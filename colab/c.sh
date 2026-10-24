#!/usr/bin/env bash
# Run the Colab CLI (installed in WSL Ubuntu) from Windows Git Bash: bash colab/c.sh <colab args...>
# Local files must be given as WSL paths, e.g. "/mnt/c/Users/kerem/Desktop/KB TD Ortak Paper/colab/vm/status.py".
export MSYS_NO_PATHCONV=1  # stop Git Bash from rewriting /mnt/... arguments into Windows paths
exec wsl.exe -d Ubuntu -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; colab "$@"' _ "$@"
