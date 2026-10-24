"""Builds simülasyon_şema.md: box-and-arrow flow diagrams (plain Unicode, any viewer) from the outputs of
scripts/simulate_five.py (tool_outputs.json, claude_sim.json).

    python -m scripts.make_simulation_schema
"""
import json
import re
import textwrap

from src.agent import AgentConfig, load_skill
from src.config import CONFIG, PROJECT_ROOT as ROOT
from src.skill_tools import SKILLS_DIR
SIM = ROOT / "outputs/agentic_ct_iqa/simulation"
RECORDS = {r["case_id"]: r for r in json.loads((SIM / "tool_outputs.json").read_text(encoding="utf-8"))}
SCORES = json.loads((SIM / "claude_sim.json").read_text(encoding="utf-8"))

W = 100
MID = W // 2
STYLES = {"single": "─│┌┐└┘├┤", "double": "═║╔╗╚╝╠╣", "dashed": "╌╎┌┐└┘├┤"}


def wrap(text: str, width: int) -> list[str]:
    if len(text) <= width:
        return [text]
    indent = " " * (len(text) - len(text.lstrip()) + 2)
    return textwrap.wrap(text, width, subsequent_indent=indent, break_long_words=False)


def box(title: str | list[str], rows: list[str] = (), w: int = W, style: str = "single") -> list[str]:
    h, v, tl, tr, bl, br, ml, mr = STYLES[style]
    inner = w - 4
    titles = [title] if isinstance(title, str) else title
    out = [tl + h * (w - 2) + tr]
    out += [f"{v} {t:<{inner}} {v}" for line in titles for t in wrap(line, inner)]
    if rows:
        out.append(ml + h * (w - 2) + mr)
        out += [f"{v} {t:<{inner}} {v}" for row in rows for t in wrap(row, inner)]
    out.append(bl + h * (w - 2) + br)
    return out


def center(lines: list[str], col: int = MID) -> list[str]:
    width = max(len(line) for line in lines)
    left = max(col - width // 2, 0)
    return [" " * left + line for line in lines]


def arrow(label: str = "", col: int = MID) -> list[str]:
    lead = " " * col
    return [lead + "│" + (f"   {label}" if label else ""), lead + "▼"]


def diamond(question: str) -> list[str]:
    w = len(question) + 6
    return center([" /" + "‾" * (w - 4) + "\\ ", "<  " + question + "  >", " \\" + "_" * (w - 4) + "/ "])


def hjoin(left: list[str], right: list[str], gap: int) -> list[str]:
    lw, rw = max(map(len, left)), max(map(len, right))
    rows = max(len(left), len(right))
    left = left + [" " * lw] * (rows - len(left))
    right = right + [" " * rw] * (rows - len(right))
    return [f"{a:<{lw}}{' ' * gap}{b}" for a, b in zip(left, right)]


BW, GAP = 48, 4
CL, CR = BW // 2, BW + GAP + BW // 2  # centres of the two branch columns


def split(left_label: str, right_label: str) -> list[str]:
    line = " " * CL + "┌" + "─" * (MID - CL - 1) + "┴" + "─" * (CR - MID - 1) + "┐"
    labels = f"{' ' * (CL - 1)}{left_label:<{CR - CL}}{right_label}"
    return [" " * MID + "│", line, labels, " " * CL + "▼" + " " * (CR - CL - 1) + "▼"]


def merge(chosen: str | None) -> list[str]:
    """Both branches (chosen None) or only the chosen one continue to the next step."""
    if chosen is None:
        return [" " * CL + "│" + " " * (CR - CL - 1) + "│",
                " " * CL + "└" + "─" * (MID - CL - 1) + "┬" + "─" * (CR - MID - 1) + "┘", " " * MID + "▼"]
    if chosen == "left":
        return [" " * CL + "│", " " * CL + "└" + "─" * (MID - CL - 1) + "┐", " " * MID + "▼"]
    return [" " * CR + "│", " " * MID + "┌" + "─" * (CR - MID - 1) + "┘", " " * MID + "▼"]


def bar(score: float, scale: int = 40) -> str:
    n = round(score / 4 * scale)
    return "█" * n + "·" * (scale - n)


# --- measurement table -------------------------------------------------------------------------------------------

MEASURES = [  # (reference key, label, raw-value source)
    ("high_pass_std", "gürültü   high_pass_std", ("noise_measure", "high_pass_energy", "high_pass_std")),
    ("immerkaer_sigma", "gürültü   Immerkær σ", ("noise_measure", "noise_sigma_estimate", "immerkaer_sigma")),
    ("nlm_residual_std", "gürültü   NLM residual", ("noise_measure", "noise_residual_map", "residual_std")),
    ("laplacian_variance", "gürültü   Laplacian varyansı", None),
    ("fourier_high_band_fraction", "doku      Fourier yüksek bant", ("texture_sharpness", "fourier_high_band_fraction", "fourier_high_band_fraction")),
    ("nps_high_band_ratio", "doku      NPS yüksek bant", ("texture_sharpness", "radial_nps_summary", "nps_high_band_ratio")),
    ("body_coherence", "doku      tutarlılık", ("texture_sharpness", "texture_coherence", "body_coherence")),
    ("edge_width_px", "keskinlik kenar genişliği (px)", ("texture_sharpness", "edge_width", "edge_width_px")),
    ("air_hp_std", "artefakt  hava enerjisi", ("artifact_check", "air_artifact_energy", "air_hp_std")),
]


def raw(record: dict, source) -> str:
    if source is None:
        return "–"
    skill, tool, key = source
    value = record["skills"][skill][tool].get("values", {}).get(key)
    return "–" if value is None else ("≈0" if value == 0 else f"{value:.4g}")


def reference(record: dict) -> tuple[dict, str]:
    ref = record["skills"]["reference_check"]["compare_to_reference"]
    return ref["values"], ref.get("text", "")


def marker(pct: float, outside: bool) -> str:
    cells = ["."] * 20
    cells[min(int(pct / 5), 19)] = "o"
    lo = "<" if outside and pct == 0 else " "
    hi = ">" if outside and pct == 100 else " "
    return f"{lo}[{''.join(cells)}]{hi}"


def flag(pct: float, outside: bool) -> str:
    if outside:
        return "DIŞARIDA"
    if pct < 1:
        return "%1 altı (zayıf)"
    if pct > 99:
        return "%99 üstü"
    return ""


def measure_rows(record: dict, control: dict | None) -> list[str]:
    pcts, text = reference(record)
    head = f"{'ölçüm':<31}{'değer':>9}" + (f"{'kontrol':>9}" if control else "") + f"{'pers.':>7}   0      50     100   not"
    rows = [head, "─" * len(head)]
    for key, label, source in MEASURES:
        pct = pcts.get(f"{key}_pct")
        if pct is None:
            continue
        outside = f"{key}=" in text and "outside the range" in text.split(f"{key}=")[1].split(";")[0]
        ctrl = f"{raw(control, source):>9}" if control else ""
        rows.append(f"{label:<31}{raw(record, source):>9}{ctrl}{pct:>7g} {marker(pct, outside)} {flag(pct, outside)}")
    return rows


# --- per-case flow -------------------------------------------------------------------------------------------------

# Text the system itself produces or reads stays in English, exactly as the pipeline uses it: task prompt, system
# prompt, skill instructions (quoted from skills/*/SKILL.md), tool output, and the simulated agent's own reasoning,
# observation and evidence. Explanations for the reader are Turkish.
TASK_PROMPT = AgentConfig().task_prompt
PROFILE_CARD = (CONFIG.expert_dir / "profile_card.txt").read_text(encoding="utf-8").strip()
SKILLS = {s.name: s for s in (load_skill(d, ()) for d in sorted(SKILLS_DIR.iterdir()) if (d / "SKILL.md").exists())}
SYSTEM_PROMPT_TAIL = (["Skills available (call read_skill before using a skill's tools):"] +
                      [f"- {s.name}: {s.description}" for name, s in SKILLS.items() if name != "rubric"] +
                      ["Finish by calling submit_score exactly once."])
QUOTED_RULES = {"expert_models": (2, 3), "reference_check": (2, 3), "ct_view": (1, 2), "artifact_check": (1, 2)}
QUESTION = "Strong evidence of smoothing?"


def skill_rules(name: str) -> list[str]:
    """The numbered rules of a skill that decide this step, verbatim."""
    items = dict(re.findall(r"^(\d+)\. (.+?)(?=^\d+\. |\Z)", SKILLS[name].instructions, re.M | re.S))
    return [f"{n}. " + " ".join(items[str(n)].split()) for n in QUOTED_RULES[name]]


CASES = {
    "A_0914": dict(
        letter="A", control=None, view='window_view(preset="soft_tissue")', skill5="ct_view", chosen="right",
        decision=["Noise measures are low (3rd-10th percentile), but edge_width sits at the 49th percentile.",
                  "Only one weak sign: nps_high_band_ratio at the 0.8th percentile; a single measure below 1% is "
                  "weak evidence.",
                  "-> No smoothing: the low noise is real quality. Stay close to the expert."],
        see="Kidneys, vessels and bowel wall are crisp; soft-tissue texture is clean.",
        evidence=["consult_experts 3.95; no smoothing sign, so the expert is reliable here",
                  "noise at 3-10th percentile of unprocessed scans; edge_width 2.03 px (49th percentile) is normal",
                  "nps_high_band_ratio at 0.8th percentile is a single weak signal, contradicted by normal edges",
                  "soft-tissue view: crisp kidneys and vessels, clean texture"]),
    "B_0791": dict(
        letter="B", control=None, view='window_view(preset="soft_tissue")', skill5="ct_view", chosen="right",
        decision=["All noise measures are at the 99.4-99.6th percentile; air_hp_std at the 99.8th.",
                  "The opposite of smoothing: noisier than almost every unprocessed scan.",
                  "-> No smoothing: the expert (0.22) agrees with the measurements."],
        see="Heavy grain; bone and kidneys still recognisable, small structures lost.",
        evidence=["high_pass_std 0.077, above the MOS-0 median 0.067; all noise measures at 99.4-99.6th percentile",
                  "air_hp_std at 99.8th percentile: noise also fills the air around the body",
                  "consult_experts 0.22 agrees with the measurements",
                  "soft-tissue view: heavy grain; bone and kidneys still recognisable, small structures lost"]),
    "C_0529": dict(
        letter="C", control=None, view='window_view(preset="narrow")', skill5="artifact_check", chosen="right",
        decision=["Noise is moderate (31st-38th percentile); edges and texture are within the normal range.",
                  "But air_hp_std is at the 67th percentile, far above the noise percentile (36th).",
                  "-> No smoothing; energy in the air that noise does not explain suggests streaks."],
        see="Fine radiating line pattern through fat and air (sparse-view streaks).",
        evidence=["high_pass_std 0.042 (36th percentile) alone suggests a score between 2 and 3",
                  "air_hp_std at 67th percentile, well above the noise percentile: extra energy in the air",
                  "narrow window: fine radiating line pattern through fat and air (sparse-view streaks)",
                  "consult_experts 2.29; lowered slightly for streaks crossing soft tissue"]),
    "D_0302_blur2": dict(
        letter="D", control="D_control_0302", view='window_view(preset="soft_tissue")', skill5="ct_view",
        chosen="left", control_label="orijinal 0302",
        decision=["8 measures lie outside the range of all 950 unprocessed slices.",
                  "Noise is below the cleanest real scan; edge_width 5.41 px exceeds the widest real scan (3.97 px).",
                  "-> Strong smoothing evidence: the expert's high score is not evidence of quality. Override it."],
        see="Vessel and bowel-wall edges washed out; fine texture gone.",
        evidence=["8 measures outside the range of all 950 unprocessed LDCT-IQA slices",
                  "noise measures below the lowest real scan (high_pass_std 0.0136 vs minimum 0.0302): smoothed",
                  "edge_width 5.41 px, above the widest real scan (3.97 px); matches Gaussian blur sigma 2",
                  "expert score 3.87 overridden: its profile card says it rewards blur",
                  "soft-tissue view: vessel and bowel-wall edges washed out, fine texture gone"]),
    "E_redcnn": dict(
        letter="E", control="E_control_gt", view='window_view(preset="narrow")', skill5="ct_view",
        chosen="right", control_label="aynı grubun NDCT'si",
        decision=["No measure lies outside the range of unprocessed scans.",
                  "Weak signs only: body_coherence 99.7th, immerkaer_sigma 4.8th, nps_high_band_ratio 9.9th "
                  "percentile; edge_width normal (55th).",
                  "-> No strong evidence, but the texture is suspicious; discount the expert."],
        see="Waxy, plastic liver texture without the fine grain of real CT.",
        evidence=["all measures inside the range of unprocessed IQA-2026v1 slices, so no strong smoothing evidence",
                  "body_coherence at 99.7th percentile and immerkaer_sigma at 4.8th: texture smoother than almost "
                  "all real scans",
                  "edge_width 1.80 px is normal: edges preserved while texture is removed (typical of supervised "
                  "denoisers)",
                  "narrow view: waxy, plastic liver texture without CT grain",
                  "expert 3.69 discounted: its card says it rewards smoothing"]),
}

TITLES = {"A_0914": "Temiz kesit (dev 0914)", "B_0791": "Çok gürültülü kesit (dev 0791)",
          "C_0529": "Streak'li kesit (dev 0529)", "D_0302_blur2": "Bulanıklaştırılmış kesit (dev 0302, Gauss σ=2)",
          "E_redcnn": "Denoiser çıktısı (IQA-2026 val, RED-CNN)"}


def expert(record: dict) -> float:
    return record["skills"]["expert_models"]["consult_experts"]["values"]["medsiglip_film"]


def case_flow(case_id: str) -> list[str]:
    c, rec, sim = CASES[case_id], RECORDS[case_id], SCORES[case_id]
    control = RECORDS[c["control"]] if c["control"] else None
    lines = box([f"GÖRSEL {c['letter']} — {TITLES[case_id]}", f"Bilinen kalite: {rec['truth']}",
                 f"Referans bankası: {rec['profile']} işlenmemiş eğitim taramaları"], style="double")
    lines += arrow()
    lines += center(box('[1] read_skill(skill="expert_models")  -> instructions', skill_rules("expert_models"), w=90))
    lines += arrow("bütçe 1/6")
    rows = [f"uzman (MedSigLIP-FiLM)  {expert(rec):.2f}   {bar(expert(rec))}"]
    if control:
        rows.append(f"{c['control_label']:<22}  {expert(control):.2f}   {bar(expert(control))}")
        rows.append("(ikinci satır karşılaştırma için; ajana gösterilmez)")
    rows.append("profile card: " + PROFILE_CARD)
    lines += center(box("[2] consult_experts", rows, w=90))
    lines += arrow("bütçe 2/6")
    lines += center(box('[3] read_skill(skill="reference_check")  -> instructions', skill_rules("reference_check"), w=90))
    lines += arrow("bütçe 3/6")
    note = ["kontrol sütunu = karşılaştırma görseli, sadece bilgi için; ajan görmez"] if control else []
    lines += box("[4] compare_to_reference — her ölçümün işlenmemiş taramalar arasındaki persentili",
                 measure_rows(rec, control) + ["", "tool text: " + reference(rec)[1], ""] + note + ["o = persentil konumu;  < > ve DIŞARIDA = bütün gerçek taramaların dışında;",
                                                "– = ayrı bir araçla ölçülmüyor, sadece persentili var"])
    lines += arrow("bütçe 4/6")
    lines += diamond(QUESTION)
    lines += split("YES", "NO / weak")
    chosen = c["chosen"]
    left = box(["SEÇİLDİ" if chosen == "left" else "(bu görselde değil)", "Override the expert;",
                "score by what the smoothing destroyed"], w=BW, style="double" if chosen == "left" else "dashed")
    right = box(["SEÇİLDİ" if chosen == "right" else "(bu görselde değil)", "Stay close to the expert",
                 "unless your own evidence clearly disagrees"], w=BW, style="double" if chosen == "right" else "dashed")
    lines += hjoin(left, right, GAP)
    lines += merge(chosen)
    lines += center(box("Ajanın muhakemesi (agent reasoning)", c["decision"], w=90))
    lines += arrow()
    lines += center(box(f'[5] read_skill(skill="{c['skill5']}")  -> instructions', skill_rules(c["skill5"]), w=90))
    lines += arrow("bütçe 5/6")
    lines += center(box(f"[6] {c['view']}  -> image", [f"Agent observation: {c['see']}"], w=90))
    lines += arrow("bütçe 6/6 — bütçe doldu")
    trusted = json.dumps(sim["trusted_experts"])
    lines += center(box("submit_score", [f"score: {sim['score']:.2f}   confidence: {sim['confidence']:.2f}   "
                                         f"trusted_experts: {trusted}", "evidence:"] +
                        [f"  • {e}" for e in c["evidence"]], w=90, style="double"))
    lines += arrow()
    result = []
    if sim.get("mos") is not None:
        result.append(f"MOS (radyolog)   {sim['mos']:.2f}  {bar(sim['mos'])}")
    result.append(f"Uzman            {expert(rec):.2f}  {bar(expert(rec))}")
    result.append(f"Claude (sim.)    {sim['score']:.2f}  {bar(sim['score'])}")
    if control:
        result.append(f"{c['control_label']:<16} {expert(control):.2f}  {bar(expert(control))}  (uzman puanı)")
    result.append("                       0         1         2         3         4")
    lines += center(box("SONUÇ", result, w=90))
    return lines


def overview() -> list[str]:
    lines = box(["GİRDİ: tek bir BT kesiti (512×512) + görev metni"], ["user: " + TASK_PROMPT], style="double")
    lines += arrow()
    lines += box("Sistem prompt'u", ["[rubric: skills/rubric/SKILL.md, 0 nondiagnostic ... 4 excellent]", ""]
                 + SYSTEM_PROMPT_TAIL)
    lines += arrow()
    lines += center(box("[1-2] UZMANA SOR", ['read_skill(skill="expert_models") -> consult_experts()',
                                             'returns: {"medsiglip_film": <0-4>} + profile card'], w=80))
    lines += arrow()
    lines += center(box("[3-4] GERÇEK TARAMALARLA KARŞILAŞTIR", ['read_skill(skill="reference_check") -> compare_to_reference()',
                                                                'returns: {"<measure>_pct": ...} + out-of-range notes'],
                        w=80))
    lines += arrow()
    lines += diamond(QUESTION)
    lines += split("YES", "NO / weak")
    left = box(["Override the expert;", "score by what the smoothing destroyed;",
                "leave it out of trusted_experts", "örnek: D (bulanık)"], w=BW)
    right = box(["Stay close to the expert", "unless your own evidence clearly disagrees;",
                 "high air_hp_std -> artifact_check", "otherwise -> ct_view",
                 "örnek: A, B, C, E"], w=BW)
    lines += hjoin(left, right, GAP)
    lines += merge(None)
    lines += center(box("[5-6] GÖRSEL DOĞRULAMA", ["read_skill(...) -> window_view(preset) / zoom_view(row, col)",
                                                   "returns: PNG image of the slice"], w=80))
    lines += arrow("6 çağrılık bütçe burada doluyor")
    lines += center(box("submit_score", ['{"score": 0-4, "confidence": 0-1, "evidence": [...], "trusted_experts": [...]}'],
                        w=80, style="double"))
    lines += arrow()
    lines += center(box("KALİBRASYON g_β", ["100 kalibrasyon görüntüsünde fit edilen 4 parametreli logistic "
                                            "(Plan 2.3)"], w=80))
    lines += arrow()
    lines += center(box("ÇIKTI: final kalite puanı", [], w=80, style="double"))
    return lines


def comparison() -> list[str]:
    rows = []
    for case_id in CASES:
        rec, sim = RECORDS[case_id], SCORES[case_id]
        letter = CASES[case_id]["letter"]
        if sim.get("mos") is not None:
            rows.append(f"{letter}  MOS     {sim['mos']:.2f}  {bar(sim['mos'])}")
        else:
            rows.append(f"{letter}  MOS        –  ({rec['truth']})")
        rows.append(f"   Uzman   {expert(rec):.2f}  {bar(expert(rec))}")
        rows.append(f"   Claude  {sim['score']:.2f}  {bar(sim['score'])}")
        control = CASES[case_id]["control"]
        if control:
            rows.append(f"   ({CASES[case_id]['control_label']}: uzman {expert(RECORDS[control]):.2f})")
        rows.append("")
    rows.append("                0         1         2         3         4")
    return box("BEŞ GÖRSEL TEK BAKIŞTA", rows)


def fence(lines: list[str]) -> str:
    return "```text\n" + "\n".join(line.rstrip() for line in lines) + "\n```\n"


INTRO = r"""# Simülasyon şeması: kutular ve oklarla ajan akışı

`simülasyon.md`'deki 5 görselin akışı burada kutu-ok şeması olarak çizildi. Şemalar düz metin; her editörde ve
terminalde aynı görünür (eşit genişlikli yazı tipi gerekir).

- Araç değerleri **gerçek**: `outputs/agentic_ct_iqa/simulation/tool_outputs.json` dosyasından okunuyor.
- Claude'un kararları ve puanları **simülasyon**: API çağrısı yok; puanları araç çıktılarına ve pencere görüntülerine
  bakarak ben verdim (`claude_sim.json`). Görselleri seçerken MOS'u gördüm; gerçek ajan görmez.
- Bu dosya `python -m scripts.make_simulation_schema` ile `simulation/` çıktılarından üretiliyor; elle düzenlemeyin.

**Gösterim**

```text
╔══════╗  çift çizgi = girdi, seçilen yol, final cevap
┌──────┐  tek çizgi  = adım (araç çağrısı ya da talimat okuma)
┌╌╌╌╌╌╌┐  kesik çizgi = bu görselde seçilmeyen yol
 /‾‾‾\
<  ?  >   karar noktası
 \___/
"bütçe k/6" = o ana kadar harcanan araç çağrısı; read_skill de bütçeden düşüyor
[....o.....] = persentil konumu (0 solda, 100 sağda);  < > = bütün gerçek taramaların dışında
█ çubuğu = 0–4 puan ölçeği
```
"""


def main() -> None:
    parts = [INTRO, "\n## 1. Genel pipeline\n\n", fence(overview()),
             "\n## 2. Beş görsel tek bakışta\n\n", fence(comparison())]
    for n, case_id in enumerate(CASES, start=3):
        parts += [f"\n## {n}. Görsel {CASES[case_id]['letter']} — {TITLES[case_id]}\n\n", fence(case_flow(case_id))]
    (ROOT / "simülasyon_şema.md").write_text("".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
