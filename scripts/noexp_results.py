"""Raw Qwen3.8-27B without the expert on the 300-image test (scripts/make_noexpert.py, colab/vm/q27_noexp.py).

Fixed before the run (2026-10-22): one run per setup; overall (|PLCC| + |SROCC| + |KROCC|) on the slices it scored,
with the count; every setup against MedSigLIP on the same slices, and ne26 against setup 26 with the expert (the
2026-10-15 run on the same 300) for the expert's share, 95% CI by a paired bootstrap over slices. Nothing is selected
on these numbers.

    python -m scripts.noexp_results ingest     # colab/noexp_job/noexp_results.zip -> outputs/agentic_ct_iqa/test300/runs/
    python -m scripts.noexp_results report     # -> outputs/agentic_ct_iqa/noexp/report.md
"""
import json
import sys
import zipfile

import numpy as np
import pandas as pd

from scripts.sandbox_llama import parse
from scripts.test300_results import RUNS, paired_bootstrap, scores
from src.config import CONFIG, PROJECT_ROOT
from src.metrics import challenge_scores

ZIP = PROJECT_ROOT / "colab/noexp_job/noexp_results.zip"
OUT = CONFIG.output_dir / "noexp"
SETUPS = {"ne16": "single request, image only", "ne17": "single request, image + measurements",
          "ne18": "agent, image only", "ne19": "agent, images + tool files",
          "ne26": "agent, images + tool files + 10 examples"}


def ingest() -> None:
    with zipfile.ZipFile(ZIP) as z:
        for name in z.namelist():
            target = (OUT / name) if name.endswith(".jsonl") else (RUNS / name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))
    print("ingested", ZIP)


def singles(key: pd.DataFrame) -> dict:
    final = {}
    for line in open(OUT / "raw300.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("type") == "generation" and r["final"]:
            answer = parse(r["response"])
            final[(r["setup"], r["image_id"])] = answer["score"] if answer else np.nan
    names = [f"batch_{b:02d}_{i}" for b, i in zip(key["batch"], key["image_id"])]
    return {s: np.array([final.get((s, n), np.nan) for n in names]) for s in ("ne16", "ne17")}


def report() -> None:
    frame = scores()  # MOS, expert (MedSigLIP as setup 26 saw it) and setup 26's agent, per slice
    frame = frame.rename(columns={"agent": "setup26 (with expert)"})
    for name, values in singles(frame).items():
        frame[name] = values
    for name in ("ne18", "ne19", "ne26"):
        frame[name] = scores(f"{name}_batch")["agent"].to_numpy()
    rows, lines = {}, []
    for name in (*SETUPS, "setup26 (with expert)", "expert"):
        ok = frame[frame[name].notna()]
        s = challenge_scores(ok[name].to_numpy(), ok["mos"].to_numpy())
        row = {"what": SETUPS.get(name, "MedSigLIP-FiLM" if name == "expert" else "agent + tools + expert + examples"),
               "scored": len(ok), **{k: round(s[k], 3) for k in ("plcc", "srocc", "krocc", "overall")},
               "mae": round(float((ok[name] - ok["mos"]).abs().mean()), 3),
               "bias": round(float((ok[name] - ok["mos"]).mean()), 3)}
        if name != "expert":
            d = paired_bootstrap(ok, name, "expert")
            row["vs MedSigLIP (same slices)"] = f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"
        rows[name] = row
    both = frame[frame["ne26"].notna() & frame["setup26 (with expert)"].notna()]
    d = paired_bootstrap(both, "ne26", "setup26 (with expert)")
    lines += ["# Raw Qwen3.8-27B without the expert, 300 LDCT-IQA test slices", "",
              "No MedSigLIP score anywhere, no fine-tuning; one run per setup. Overall on the slices each setup "
              "scored.", "", pd.DataFrame(rows).T.to_markdown(), "",
              f"- Expert's share in setup 26 (ne26 - setup 26, {len(both)} slices): {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]",
              "- For reference (same 300): noise rule 2.600, setup 30 2.483, Stage B 2.735 (fine-tuned, no expert).", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "per_image.csv", index=False)
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"ingest": ingest, "report": report}[sys.argv[1]]()
