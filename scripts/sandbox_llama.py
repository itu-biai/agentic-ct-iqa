"""Llama 3.2 11B Vision Instruct (NVIDIA hosted) on the 20 sandbox images, the counterpart of setups 1 and 2.

The model takes one image per request and cannot use tools, so there is no agent loop: each image is one independent
request with the same concept text as TASK.md, and the model answers with JSON. It has no file system, code or web
access at all. Setup 5 adds the image's numeric tool outputs (including the expert score fed from setup 3) and the
tool descriptions; the tool view images cannot be sent (one image per request).

    python -m scripts.sandbox_llama setup4_llama_vision_only 1
    python -m scripts.sandbox_llama setup5_llama_vision_tools 1

Writes experiment_sandboxes/results/<setup>/run_<n>/ in the standard format: scores.json, transcript.jsonl,
run_info.json.
"""
import json
import re
import sys
import time
from datetime import datetime

from scripts.nvidia_vlm import api_key, chat
from src.config import PROJECT_ROOT

MODEL = "meta/llama-3.2-11b-vision-instruct"
SANDBOXES = PROJECT_ROOT / "experiment_sandboxes"
RESULTS = SANDBOXES / "results"
IMAGES = SANDBOXES / "setup1_vision_only/images"
TOOLS = SANDBOXES / "setup2_vision_tools"

CONCEPT = """# Task: rate the diagnostic image quality of a CT slice

## Your role

You are a radiologist taking part in an image quality reading study.

## What image quality assessment is

Image quality assessment (IQA) means judging how good an image is for its purpose. For medical images the purpose
is diagnosis, so quality means diagnostic quality: how well a reader can see the anatomy and could detect disease in
the image. This study is no-reference: there is no clean version of the image to compare with; the image is judged
on its own.

## The image

The attached image is an axial slice of abdominal CT, 512 x 512 pixels, grayscale. It comes from a low-dose CT
examination: a CT scan taken with a reduced radiation dose, which can make images noisier or introduce artifacts.
The slice is already shown in a fixed display window.

## The rating scale

Rate the image on this five-level scale. Scores are continuous: use decimals (for example 2.6) when the image falls
between two levels.

- 0 nondiagnostic: noise or artifacts hide anatomy; no diagnosis is possible.
- 1 poor: major structures visible, but noise or streaks make interpretation unreliable.
- 2 fair: diagnosable with effort; noise or streaks clearly visible in soft tissue.
- 3 good: small noise or artifacts that do not interfere with diagnosis.
- 4 excellent: anatomy crisp, soft-tissue texture clean, no disturbing artifacts.
"""

ANSWER = """## Answer

Look at the image and rate it. Answer with one JSON object and nothing else:

{"score": <number from 0 to 4>, "confidence": <number from 0 to 1>, "rationale": "<one or two sentences>"}
"""


def measurements(image_id: str) -> str:
    glossary = (TOOLS / "TOOLS.md").read_text(encoding="utf-8").split("## Views")[0].replace("# Tools", "").strip()
    results = json.loads((TOOLS / "tool_results" / f"{image_id}.json").read_text(encoding="utf-8"))["tools"]
    return ("## Measurements\n\nAn automatic measurement pipeline has already run a set of image analysis tools on "
            "this image. Use the measurements as support for your own reading; the final score is your judgment.\n\n"
            f"{glossary}\n\n### Tool outputs for this image\n\n```json\n{json.dumps(results, indent=1)}\n```\n")


def parse(text: str) -> dict | None:
    """The JSON object if the model wrote one; otherwise a score stated in plain words ("Score: 3", "rated as 2.5"),
    which this model often writes instead of JSON. Scores outside 0-4 count as no answer."""
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            answer = json.loads(match.group(0))
            score = float(answer["score"])
            if 0 <= score <= 4:
                return {"score": score, "confidence": answer.get("confidence"),
                        "rationale": str(answer.get("rationale", "")), "format": "json"}
        except (ValueError, KeyError, TypeError):
            pass
    found = re.search(r"\bscore\b[^0-9\n]{0,15}(\d+(?:\.\d+)?)|\brated as\D{0,5}(\d+(?:\.\d+)?)", text, re.I)
    if not found:
        return None
    score = float(found.group(1) or found.group(2))
    if not 0 <= score <= 4:
        return None
    conf = re.search(r"\bconfidence\b[^0-9\n]{0,15}(\d*\.?\d+)", text, re.I)
    why = re.search(r"\brationale\b\W{0,6}(.+)", text, re.I)
    return {"score": score, "confidence": float(conf.group(1)) if conf else None,
            "rationale": (why.group(1) if why else text).strip()[:400], "format": "text"}


def run(setup: str, run_id: int) -> None:
    out = RESULTS / setup / f"run_{run_id}"
    if out.exists():
        raise SystemExit(f"{out} already exists")
    out.mkdir(parents=True)
    key = api_key()
    ids = sorted(p.stem for p in IMAGES.glob("img_*.png"))
    scores, transcript, start = [], [], datetime.now()
    for image_id in ids:
        prompt = CONCEPT + ("\n" + measurements(image_id) if setup == "setup5_llama_vision_tools" else "") + "\n" + ANSWER
        answer, attempts = None, 0
        while answer is None and attempts < 3:
            attempts += 1
            began = time.time()
            try:
                response = chat(key, MODEL, prompt, IMAGES / f"{image_id}.png", max_tokens=700)
                text = response["choices"][0]["message"]["content"]
                usage = response.get("usage")
            except RuntimeError as error:
                text, usage = f"ERROR {error}", None
            answer = parse(text)
            transcript.append({"image_id": image_id, "attempt": attempts, "seconds": round(time.time() - began, 1),
                               "prompt_chars": len(prompt), "response": text, "usage": usage})
        if answer is not None:
            scores.append({"image_id": image_id, **answer})
        print(image_id, answer["score"] if answer else "FAILED", flush=True)
    (out / "scores.json").write_text(json.dumps({"scores": scores}, indent=1), encoding="utf-8")
    (out / "transcript.jsonl").write_text("\n".join(json.dumps(t) for t in transcript), encoding="utf-8")
    (out / "run_info.json").write_text(json.dumps({
        "setup": setup, "run": run_id, "model": MODEL, "provider": "NVIDIA build.nvidia.com (free tier)",
        "mode": "one independent request per image, no tools, no file system", "start": start.isoformat(),
        "end": datetime.now().isoformat(), "requests": len(transcript)}, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
