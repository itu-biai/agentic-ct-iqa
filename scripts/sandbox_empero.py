"""Empero's Qwen3.8-4B-Distill (empero-ai/Qwen3.8-4B-Distill: Qwen3.5-4B fine-tuned on text traces of Qwen3.8 2.4T,
not an official Qwen release) on the 20 sandbox images, served by vLLM on a Colab T4, in both protocols.

Setups 20-23 repeat setups 12-15 (Qwen3.5-4B) with this model and nothing else changed: the same prompts, sandboxes,
permissions, server flags and runners. Setups 24 (one request per image) and 25 (OpenCode agent) are setups 21 and 23
plus 10 LDCT-IQA training slices with their MOS as reference examples (scripts/make_sandboxes.py examples): the agent
gets them as examples/ in the setup2b sandbox, the single request gets them attached before the image to rate.

    python -m scripts.sandbox_empero prepare               # writes colab/empero_job/
    python -m scripts.sandbox_empero ingest <raw.jsonl>    # setups 20/21/24 -> results/<setup>/run_<n>/
    python -m scripts.sandbox_empero agent <agent.zip>     # setups 22/23/25 -> results/<setup>/run_<n>/
"""
import json
import sys
import zipfile
from pathlib import Path

from scripts import sandbox_qwen35
from scripts.sandbox_llama import ANSWER, CONCEPT, IMAGES, RESULTS, SANDBOXES, measurements
from src.config import PROJECT_ROOT

MODEL = "empero-ai/Qwen3.8-4B-Distill"
JOB = PROJECT_ROOT / "colab/empero_job"
EXAMPLES = SANDBOXES / "setup2b_vision_tools_examples/examples"
SINGLE = {"setup20_empero_4b_vision_only": (False, False), "setup21_empero_4b_vision_tools": (True, False),
          "setup24_empero_4b_vision_tools_examples": (True, True)}  # (tool outputs, examples)
AGENT_SETUPS = ("setup22_empero_4b_agent_vision_only", "setup23_empero_4b_agent_vision_tools",
                "setup25_empero_4b_agent_vision_tools_examples")
# The single-request counterpart of the setup2b TASK.md section; the VM attaches the examples after the prompt text.
EXAMPLES_SECTION = """## Reference examples

Before the image to rate, 10 other slices of the same kind are attached that radiologists rated on the same scale;
the line before each gives its score (the mean of the radiologists' ratings). Look at the examples and their scores
to see how the scale is used. The last attached image is the one to rate.
"""


def prepare() -> None:
    JOB.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.stem for p in IMAGES.glob("img_*.png"))
    prompts = [{"setup": setup, "image_id": image_id, "examples": examples,
                "prompt": CONCEPT + ("\n" + measurements(image_id) if tools else "") +
                          ("\n" + EXAMPLES_SECTION if examples else "") + "\n" + ANSWER}
               for setup, (tools, examples) in SINGLE.items() for image_id in ids]
    (JOB / "prompts.json").write_text(json.dumps(prompts, indent=1), encoding="utf-8")
    (JOB / "examples.json").write_text((EXAMPLES / "labels.json").read_text(encoding="utf-8"), encoding="utf-8")
    with zipfile.ZipFile(JOB / "images.zip", "w") as archive:
        for image_id in ids:
            archive.write(IMAGES / f"{image_id}.png", f"{image_id}.png")
        for path in sorted(EXAMPLES.glob("ex_*.png")):
            archive.write(path, f"examples/{path.name}")
    with zipfile.ZipFile(JOB / "sandboxes.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("setup1_vision_only", "setup2_vision_tools", "setup2b_vision_tools_examples"):
            for path in (SANDBOXES / folder).rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(SANDBOXES).as_posix())
    print(f"{len(prompts)} prompts, {len(ids)} images, examples and sandboxes -> {JOB}")


def ingest(raw: Path) -> None:
    sandbox_qwen35.ingest(raw, model=MODEL, provider="Colab Tesla T4, vLLM, fp16")
    for run in sorted((RESULTS / "setup24_empero_4b_vision_tools_examples").glob("run_*")):
        info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
        info["mode"] = ("one independent request per image, no tools, no file system; the 10 rated training "
                        "examples (setup2b) attached before the image")
        (run / "run_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "prepare":
        prepare()
    elif command == "ingest":
        ingest(Path(sys.argv[2]))
    elif command == "agent":
        sandbox_qwen35.agent(Path(sys.argv[2]), setups=AGENT_SETUPS, model=MODEL, server="vLLM on Colab T4, fp16")
