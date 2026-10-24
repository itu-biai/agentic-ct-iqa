"""Tool pipeline walkthrough on five dev cases (simülasyon.md): every tool's output, the views the agent would see,
and a summary figure with the simulated agent scores from simulation/claude_sim.json (written by hand, no API call).
All cases come from dev splits (training images, IQA 2026v1 val); no test image is used.

    python -m scripts.simulate_five            # tool outputs + views
    python -m scripts.simulate_five --figure   # summary figure, after claude_sim.json exists
"""
import json
import sys
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from src.agent import Case
from src.config import CONFIG
from src.data import index_images, load_labels, read_image
from src.experts import expert_scores, load_fold_predictions, strip_ext
from src.skill_tools import IQA2026, LDCTIQA_DEV, DatasetProfile, build_tools
from src.stress_sets import denoiser_group

OUT = CONFIG.output_dir / "simulation"
VIEW_TOOLS = {"window_view", "zoom_view", "orientation_coherence_map", "fourier_spectrum", "mscn_map",
              "gradient_deviation_map", "noise_residual_map"}


@dataclass(frozen=True)
class SimCase:
    case_id: str
    title: str
    image: np.ndarray
    profile: DatasetProfile
    expert_raw: float
    truth: str  # what is known about its quality (MOS or the stress-set ordering)


def load_cases() -> tuple[list[SimCase], dict[str, SimCase]]:
    index = index_images(CONFIG.train_images)
    mos = {k.split(".")[0]: v for k, v in load_labels(CONFIG.train_labels).items()}
    experts = expert_scores(load_fold_predictions(CONFIG.expert_dir))
    train = experts["train"].copy()
    train.index = strip_ext(train.index)

    def ldct(image_id: str) -> np.ndarray:
        return read_image(index[image_id])

    group = {name: image / 255.0 for name, image in denoiser_group(CONFIG.iqa2026_root / "val" / "000002_01_01_162").items()}
    cases = [
        SimCase("A_0914", "Temiz kesit (dev 0914)", ldct("0914"), LDCTIQA_DEV, train["0914"], f"MOS {mos['0914']}"),
        SimCase("B_0791", "Çok gürültülü kesit (dev 0791)", ldct("0791"), LDCTIQA_DEV, train["0791"], f"MOS {mos['0791']}"),
        SimCase("C_0529", "Streak'li kesit (dev 0529)", ldct("0529"), LDCTIQA_DEV, train["0529"], f"MOS {mos['0529']}"),
        SimCase("D_0302_blur2", "Bulanıklaştırılmış kesit (0302, σ=2)", ndimage.gaussian_filter(ldct("0302"), 2.0),
                LDCTIQA_DEV, experts["s1_dev"]["0302|2.0"], f"orijinal MOS {mos['0302']}; bulanığı daha kötü olmalı"),
        SimCase("E_redcnn", "Denoiser çıktısı (IQA-2026 val, RED-CNN)", group["redcnn"], IQA2026,
                experts["s2_dev"]["000002_01_01_162|redcnn"], "etiket yok; NDCT'den yüksek olmamalı"),
    ]
    # Comparison partners, measured but not scored: the unblurred original and the NDCT of the denoiser group.
    controls = {
        "D_0302_blur2": SimCase("D_control_0302", "0302 orijinal", ldct("0302"), LDCTIQA_DEV,
                                experts["s1_dev"]["0302|0.0"], f"MOS {mos['0302']}"),
        "E_redcnn": SimCase("E_control_gt", "Aynı grubun NDCT'si (gt)", group["gt"], IQA2026,
                            experts["s2_dev"]["000002_01_01_162|gt"], "NDCT referansı"),
    }
    return cases, controls


def run_all_tools(case: SimCase) -> dict:
    """Every numeric tool of every skill, plus the reference comparison text and the expert card output."""
    tools = build_tools(case.profile)
    agent_case = Case(case.case_id, case.image, {"medsiglip_film": float(case.expert_raw)})
    record = {"case_id": case.case_id, "title": case.title, "truth": case.truth, "profile": case.profile.name,
              "expert_raw": round(float(case.expert_raw), 3), "skills": {}}
    for skill, specs in tools.items():
        for spec in specs:
            if spec.name in VIEW_TOOLS - {"noise_residual_map"}:
                continue
            try:
                result = spec.run(agent_case)
                entry = {"values": {k: round(v, 5) for k, v in result.values.items()}}
                if result.text and spec.name != "consult_experts":
                    entry["text"] = result.text
            except Exception as error:  # report, as the agent loop would
                entry = {"error": f"{type(error).__name__}: {error}"}
            record["skills"].setdefault(skill, {})[spec.name] = entry
    return record


def save_views(case: SimCase) -> None:
    """The views the agent can request: window presets and, where the profile allows it, a zoom."""
    tools = {spec.name: spec for specs in build_tools(case.profile).values() for spec in specs}
    agent_case = Case(case.case_id, case.image)
    for preset in case.profile.windows:
        (OUT / f"{case.case_id}_window_{preset}.png").write_bytes(tools["window_view"].run(agent_case, preset=preset).image_png)
    (OUT / f"{case.case_id}_residual.png").write_bytes(tools["noise_residual_map"].run(agent_case).image_png)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases, controls = load_cases()
    records = []
    for case in cases + list(controls.values()):
        print("measuring", case.case_id, flush=True)
        records.append(run_all_tools(case))
        save_views(case)
    (OUT / "tool_outputs.json").write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT / "tool_outputs.json")


def figure() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases, _ = load_cases()
    records = {r["case_id"]: r for r in json.loads((OUT / "tool_outputs.json").read_text(encoding="utf-8"))}
    sim = json.loads((OUT / "claude_sim.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, len(cases), figsize=(3.6 * len(cases), 7.2), layout="constrained",
                             gridspec_kw={"height_ratios": [1.6, 1]})
    for col, case in enumerate(cases):
        rec, s = records[case.case_id], sim[case.case_id]
        level, width = case.profile.windows.get("soft_tissue", case.profile.windows["full"])
        axes[0, col].imshow(np.clip(case.image, level - width / 2, level + width / 2), cmap="gray")
        axes[0, col].set_title(f"{case.title}\n{case.truth}", fontsize=9)
        axes[0, col].axis("off")
        ref = rec["skills"].get("reference_check", {}).get("compare_to_reference", {}).get("values", {})
        expert = rec["skills"]["expert_models"]["consult_experts"]["values"]["medsiglip_film"]
        labels = ["Uzman", "Claude (sim.)"]
        scores = [expert, s["score"]]
        colours = ["#8c8c8c", "#d1603d"]
        if s.get("mos") is not None:
            labels.insert(0, "MOS")
            scores.insert(0, s["mos"])
            colours.insert(0, "#3b7dd8")
        ax = axes[1, col]
        bars = ax.barh(labels, scores, color=colours)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_xlim(0, 4.6)
        ax.invert_yaxis()
        ax.axvline(4, color="#cccccc", lw=0.8)
        ax.set_title(f"persentil: gürültü {ref.get('high_pass_std_pct', np.nan):g}, kenar "
                     f"{ref.get('edge_width_px_pct', np.nan):g}, hava {ref.get('air_hp_std_pct', np.nan):g}", fontsize=8)
    fig.suptitle("Beş görsel: uzman puanı, simüle Claude puanı ve (varsa) radyolog MOS'u", fontsize=12)
    fig.savefig(OUT / "simulation_summary.png", dpi=110)
    print("wrote", OUT / "simulation_summary.png")

    # Reference-bank percentiles: one row per case, one column per measure; 0 and 100 = outside every real scan.
    measures = ["high_pass_std", "immerkaer_sigma", "nlm_residual_std", "laplacian_variance",
                "fourier_high_band_fraction", "nps_high_band_ratio", "edge_width_px", "air_hp_std", "body_coherence"]
    table = np.array([[records[c.case_id]["skills"]["reference_check"]["compare_to_reference"]["values"]
                       .get(f"{m}_pct", np.nan) for m in measures] for c in cases])
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.imshow(table, cmap="coolwarm", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(measures)), measures, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(cases)), [c.title for c in cases], fontsize=9)
    for (row, col), value in np.ndenumerate(table):
        ax.text(col, row, "–" if np.isnan(value) else f"{value:g}", ha="center", va="center", fontsize=9,
                fontweight="bold" if value in (0, 100) else "normal")
    ax.set_title("compare_to_reference: işlenmemiş kesitler arasındaki persentil (0 / 100 = hepsinin dışında)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "simulation_percentiles.png", dpi=110)
    print("wrote", OUT / "simulation_percentiles.png")


if __name__ == "__main__":
    figure() if "--figure" in sys.argv else main()
