"""Runs MedGemma 1.5 4B on the sandbox prompts (scripts/sandbox_medgemma.py prepare) on the Colab VM.

Expects /content/mg/prompts.json and /content/mg/images/. Appends one JSON line per generation to /content/mg/raw.jsonl
and skips work already in it, so it can be restarted. Three runs; up to three attempts per image when the answer has
no readable score (same rule as the Llama runs). Prints MEDGEMMA_DONE at the end.
"""
import json
import os
import re
import time
from datetime import datetime

import torch
import transformers
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "google/medgemma-1.5-4b-it"
JOB = "/content/mg"
OUT = f"{JOB}/raw.jsonl"
RUNS = (1, 2, 3)


def has_score(text: str) -> bool:
    """Same acceptance rule as scripts/sandbox_llama.parse: a JSON score or a stated score, in [0, 4]."""
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            if 0 <= float(json.loads(match.group(0))["score"]) <= 4:
                return True
        except (ValueError, KeyError, TypeError):
            pass
    found = re.search(r"\bscore\b[^0-9\n]{0,15}(\d+(?:\.\d+)?)|\brated as\D{0,5}(\d+(?:\.\d+)?)", text, re.I)
    return bool(found) and 0 <= float(found.group(1) or found.group(2)) <= 4


prompts = json.load(open(f"{JOB}/prompts.json", encoding="utf-8"))
done = set()
if os.path.exists(OUT):
    for line in open(OUT, encoding="utf-8"):
        r = json.loads(line)
        if r.get("type") == "generation" and (r["final"]):
            done.add((r["setup"], r["run"], r["image_id"]))

processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
generation = {k: v for k, v in model.generation_config.to_dict().items()
              if k in ("do_sample", "temperature", "top_k", "top_p", "repetition_penalty")}
generation.update(do_sample=True, max_new_tokens=2048)  # room for the thinking before the answer
with open(OUT, "a", encoding="utf-8") as out:
    out.write(json.dumps({"type": "meta", "model": MODEL, "gpu": torch.cuda.get_device_name(0),
                          "transformers": transformers.__version__, "dtype": "bfloat16", "generation": generation,
                          "time": datetime.now().isoformat()}) + "\n")

for run in RUNS:
    for index, item in enumerate(prompts):
        key = (item["setup"], run, item["image_id"])
        if key in done:
            continue
        image = Image.open(f"{JOB}/images/{item['image_id']}.png").convert("RGB")
        messages = [{"role": "user", "content": [{"type": "text", "text": item["prompt"]},
                                                 {"type": "image", "image": image}]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt").to(model.device, dtype=torch.bfloat16)
        length = inputs["input_ids"].shape[-1]
        for attempt in (1, 2, 3):
            torch.manual_seed(run * 100000 + index * 10 + attempt)
            start = datetime.now()
            began = time.time()
            with torch.inference_mode():
                output = model.generate(**inputs, **generation)
            text = processor.decode(output[0][length:], skip_special_tokens=True)
            final = has_score(text) or attempt == 3
            with open(OUT, "a", encoding="utf-8") as out:
                out.write(json.dumps({"type": "generation", "setup": item["setup"], "run": run,
                                      "image_id": item["image_id"], "attempt": attempt, "final": final,
                                      "seconds": round(time.time() - began, 1), "prompt_tokens": length,
                                      "new_tokens": int(output.shape[-1] - length), "start": start.isoformat(),
                                      "end": datetime.now().isoformat(), "response": text}) + "\n")
            if final:
                break
        print(run, item["setup"], item["image_id"], attempt, flush=True)
print("MEDGEMMA_DONE", flush=True)
