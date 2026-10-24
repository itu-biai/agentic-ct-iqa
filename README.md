# Agentic CT IQA (ISBI 2027): code

Code for the experiments of itu-biai/dlct-docs#329 (Tables 1–3, E1–E19). Code only: datasets, images, feature
arrays, run outputs and weights are not in this repository.

- `src/`: CT IQA tools, metrics, reference banks, stress sets.
- `scripts/`: local analysis and evaluation, e.g. `python -m scripts.issue329_eval` (all #329 tables),
  `python -m scripts.agent_analysis` (E14–E16a), `python -m scripts.e15_human` (E15 human check).
- `configs/`: tool groups, rationale codes (Claude: `rationale_hand_codes*.csv`, human: `rationale_human_codes*.csv`).
- `colab/`: launchers for the Colab G4 runs (`*.sh`) and the scripts that run on the VM (`colab/vm/`), e.g.
  `colab/vm/vftb_train.py` (Stage B), `colab/vm/medsig_seed.py` (MedSigLIP seeds).
- `data/download_all.sh`: downloads the datasets.
