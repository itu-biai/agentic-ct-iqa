"""MedGemma 1.5 4B (google/medgemma-1.5-4b-it) on the 20 sandbox images, run on a Colab T4.

Setups 10 and 11 follow the Llama protocol (setups 4 and 5): a small model cannot work as a file-reading agent, so each
image is one independent request with the same concept text, and setup 11 adds the same numeric tool outputs (with the
expert score) and tool descriptions. The model sees only the request: no tools, no file system, no internet. Sampling
uses the model's own generation config (do_sample on), with a fixed seed per run, image and attempt; three runs.

    python -m scripts.sandbox_medgemma prepare            # writes colab/medgemma_job/ (prompts.json, images.zip)
    (on the VM) colab/vm/medgemma_run.py                  # writes /content/mg/raw.jsonl
    python -m scripts.sandbox_medgemma ingest <raw.jsonl>  # writes results/setup1x_medgemma_*/run_<n>/

Writes experiment_sandboxes/results/<setup>/run_<n>/ in the standard format: scores.json, transcript.jsonl,
run_info.json.
"""
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

from scripts.sandbox_llama import ANSWER, CONCEPT, IMAGES, RESULTS, measurements, parse
from src.config import PROJECT_ROOT

MODEL = "google/medgemma-1.5-4b-it"
JOB = PROJECT_ROOT / "colab/medgemma_job"
SETUPS = {"setup10_medgemma_vision_only": False, "setup11_medgemma_vision_tools": True}


def prepare() -> None:
    JOB.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.stem for p in IMAGES.glob("img_*.png"))
    prompts = [{"setup": setup, "image_id": image_id,
                "prompt": CONCEPT + ("\n" + measurements(image_id) if tools else "") + "\n" + ANSWER}
               for setup, tools in SETUPS.items() for image_id in ids]
    (JOB / "prompts.json").write_text(json.dumps(prompts, indent=1), encoding="utf-8")
    with zipfile.ZipFile(JOB / "images.zip", "w") as archive:
        for image_id in ids:
            archive.write(IMAGES / f"{image_id}.png", f"{image_id}.png")
    print(f"{len(prompts)} prompts, {len(ids)} images -> {JOB}")


def ingest(raw: Path) -> None:
    records = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    meta = next(r for r in records if r.get("type") == "meta")
    runs = defaultdict(list)
    for r in records:
        if r.get("type") == "generation":
            runs[(r["setup"], r["run"])].append(r)
    for (setup, run_id), rows in sorted(runs.items()):
        out = RESULTS / setup / f"run_{run_id}"
        if out.exists():
            print(f"{out} exists, skipped")
            continue
        out.mkdir(parents=True)
        scores = []
        for image_id in sorted({r["image_id"] for r in rows}):
            last = max((r for r in rows if r["image_id"] == image_id), key=lambda r: r["attempt"])
            # MedGemma 1.5 thinks first: <unused94>thought ... <unused95> final answer. Read the final answer; if the
            # thinking ran out of tokens there is none, and the score stated in the thinking is used.
            thought, ended, final = last["response"].partition("<unused95>")
            answer = parse(final if ended else thought)
            if answer is not None:
                scores.append({"image_id": image_id, **answer, "thinking_complete": bool(ended)})
        (out / "scores.json").write_text(json.dumps({"scores": scores}, indent=1), encoding="utf-8")
        (out / "transcript.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        (out / "run_info.json").write_text(json.dumps({
            "setup": setup, "run": run_id, "model": MODEL, "provider": f"Colab {meta['gpu']}, transformers "
            f"{meta['transformers']}, {meta['dtype']}", "generation": meta["generation"],
            "mode": "one independent request per image, no tools, no file system",
            "start": min(r["start"] for r in rows), "end": max(r["end"] for r in rows), "requests": len(rows)},
            indent=1), encoding="utf-8")
        print(f"{out}: {len(scores)}/20 scored, {sum(a['format'] == 'json' for a in scores)} as JSON")


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        ingest(Path(sys.argv[2]))
