"""Builds the browser version of the simulation flow (outputs/agentic_ct_iqa/simulation/simulasyon.html) from the
outputs of scripts/simulate_five.py and the case texts in scripts/make_simulation_schema.py.

    python -m scripts.make_simulation_page
"""
import base64
import io
import json
from pathlib import Path

from PIL import Image

from scripts.make_simulation_schema import (CASES, MEASURES, PROFILE_CARD, QUESTION, RECORDS, SCORES, SIM,
                                            SYSTEM_PROMPT_TAIL, TASK_PROMPT, TITLES, expert, raw, reference,
                                            skill_rules)

TEMPLATE = Path(__file__).with_name("simulation_page_template.html")
OUT = SIM / "simulasyon.html"
VIEWS = {"A_0914": "soft_tissue", "B_0791": "soft_tissue", "C_0529": "narrow", "D_0302_blur2": "soft_tissue",
         "E_redcnn": "narrow"}
STATUS = {"A_0914": ("uzmana yakın", "ok"), "B_0791": ("uzmana yakın", "ok"), "C_0529": ("streak düzeltmesi", "ok"),
          "D_0302_blur2": ("uzman reddedildi", "override"), "E_redcnn": ("ajan da kanıyor", "fail")}


def thumbnail(path: Path, size: int = 360) -> str:
    image = Image.open(path).convert("L")
    image.thumbnail((size, size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def measures(record: dict, control: dict | None) -> list[dict]:
    pcts, text = reference(record)
    rows = []
    for key, label, source in MEASURES:
        pct = pcts.get(f"{key}_pct")
        if pct is None:
            continue
        group, name = label.split(maxsplit=1)
        outside = f"{key}=" in text and "outside the range" in text.split(f"{key}=")[1].split(";")[0]
        rows.append({"key": key, "group": group, "name": name.strip(), "value": raw(record, source),
                     "control": raw(control, source) if control else None, "pct": pct, "outside": outside})
    return rows


def case_data(case_id: str) -> dict:
    c, rec, sim = CASES[case_id], RECORDS[case_id], SCORES[case_id]
    control = RECORDS[c["control"]] if c["control"] else None
    preset = VIEWS[case_id]
    return {
        "id": c["letter"], "title": TITLES[case_id], "truth": rec["truth"], "profile": rec["profile"],
        "expert": expert(rec), "score": sim["score"], "confidence": sim["confidence"], "mos": sim.get("mos"),
        "trusted": sim["trusted_experts"], "verdict": sim["verdict"],
        "status": STATUS[case_id][0], "statusKind": STATUS[case_id][1],
        "control": {"label": c["control_label"], "expert": expert(control)} if control else None,
        "measures": measures(rec, control), "decision": c["decision"], "chosen": c["chosen"],
        "step5": c["skill5"], "step6": c["view"], "see": c["see"], "evidence": c["evidence"],
        "image": thumbnail(SIM / f"{case_id}_window_{preset}.png"), "preset": preset,
        "rules": {name: skill_rules(name) for name in ("expert_models", "reference_check", c["skill5"])},
        "toolText": reference(rec)[1],
    }


def main() -> None:
    data = {"system": {"task": TASK_PROMPT, "systemTail": SYSTEM_PROMPT_TAIL, "card": PROFILE_CARD,
                       "question": QUESTION},
            "cases": [case_data(case_id) for case_id in CASES]}
    page = TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
