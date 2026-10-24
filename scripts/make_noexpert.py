"""Raw Qwen3.8-27B without the expert: the sandbox setups 16-19 and 26 rebuilt on all 300 LDCT-IQA test slices with
no trace of MedSigLIP (no consult_experts entry, no expert line in TOOLS.md), no fine-tuning. Same 15 batches of 20,
image names and answer key as scripts/make_test300.py (outputs/agentic_ct_iqa/test300/key.csv).

  ne16  single request per image, image only                   (setup 16 protocol; it never had the expert)
  ne17  single request per image, image + tool measurements    (setup 17 minus the expert)
  ne18  OpenCode agent, image only                             (setup 18 protocol; it never had the expert)
  ne19  OpenCode agent, images + tool files                    (setup 19 minus the expert)
  ne26  OpenCode agent, images + tool files + 10 examples      (setup 26 minus the expert)

Agent sandboxes come from experiment_sandboxes/test300/batch_XX (setup 26 layout): ne26 drops the expert, ne19 also
drops examples/ and takes setup 2's TASK.md, ne18 keeps only images/ with setup 1's TASK.md.

    python -m scripts.make_noexpert          # experiment_sandboxes/test300_ne{18,19,26}/, colab/noexp_job/
"""
import json
import shutil
import zipfile

from scripts.make_sandboxes import ROOT, isolate
from scripts.sandbox_llama import ANSWER, CONCEPT
from src.config import PROJECT_ROOT

SOURCE = ROOT / "test300"
JOB = PROJECT_ROOT / "colab/noexp_job"
EXPERT_LINE = "- `consult_experts`:"


def tools_md(folder) -> str:
    return "\n".join(l for l in (folder / "TOOLS.md").read_text(encoding="utf-8").split("\n")
                     if not l.startswith(EXPERT_LINE))


def strip_expert(path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    record["tools"].pop("consult_experts")
    return record


def build(variant: str, batch) -> None:
    out = ROOT / f"test300_{variant}" / batch.name
    shutil.copytree(batch, out, ignore=shutil.ignore_patterns(".git", "scores.json"))
    if variant in ("ne18", "ne19"):
        shutil.rmtree(out / "examples")
        task = {"ne18": "setup1_vision_only", "ne19": "setup2_vision_tools"}[variant]
        shutil.copy(ROOT / task / "TASK.md", out / "TASK.md")
    if variant == "ne18":
        for sub in ("tool_results", "tool_views"):
            shutil.rmtree(out / sub)
        (out / "TOOLS.md").unlink()
    else:
        (out / "TOOLS.md").write_text(tools_md(batch), encoding="utf-8")
        for path in (out / "tool_results").glob("*.json"):
            path.write_text(json.dumps(strip_expert(path), indent=1), encoding="utf-8")
    isolate(out)


def single_prompts(batch) -> list[dict]:
    glossary = tools_md(batch).split("## Views")[0].replace("# Tools", "").strip()
    prompts = []
    for image in sorted((batch / "images").glob("img_*.png")):
        name = f"{batch.name}_{image.stem}"
        results = strip_expert(batch / "tool_results" / f"{image.stem}.json")["tools"]
        measures = ("## Measurements\n\nAn automatic measurement pipeline has already run a set of image analysis tools "
                    "on this image. Use the measurements as support for your own reading; the final score is your "
                    f"judgment.\n\n{glossary}\n\n### Tool outputs for this image\n\n```json\n"
                    f"{json.dumps(results, indent=1)}\n```\n")
        prompts += [{"setup": "ne16", "image_id": name, "prompt": CONCEPT + "\n" + ANSWER},
                    {"setup": "ne17", "image_id": name, "prompt": CONCEPT + "\n" + measures + "\n" + ANSWER}]
    return prompts


def main() -> None:
    for variant in ("ne18", "ne19", "ne26"):
        if (ROOT / f"test300_{variant}").exists():
            raise SystemExit(f"test300_{variant} already exists; delete it first to rebuild")
    JOB.mkdir(parents=True, exist_ok=True)
    prompts = []
    with zipfile.ZipFile(JOB / "images300.zip", "w") as images:
        for batch in sorted(SOURCE.glob("batch_*")):
            for variant in ("ne18", "ne19", "ne26"):
                build(variant, batch)
            prompts += single_prompts(batch)
            for image in sorted((batch / "images").glob("img_*.png")):
                images.write(image, f"{batch.name}_{image.stem}.png")
    (JOB / "prompts300.json").write_text(json.dumps(prompts, indent=1), encoding="utf-8")
    for variant in ("ne18", "ne19", "ne26"):  # one zip per batch: large zips time out in `colab upload`
        for batch in sorted((ROOT / f"test300_{variant}").glob("batch_*")):
            with zipfile.ZipFile(JOB / f"{variant}_{batch.name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
                for path in sorted(batch.rglob("*")):
                    if path.is_file():
                        z.write(path, path.relative_to(ROOT).as_posix())
    print(f"{len(prompts)} single prompts, {len(list(JOB.glob('ne*_batch_*.zip')))} sandbox zips -> {JOB}")


if __name__ == "__main__":
    main()
