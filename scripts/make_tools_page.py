"""Builds the visual tool catalogue (outputs/agentic_ct_iqa/tools/araclar.html): every agent tool with its purpose,
method and screening numbers, plus small charts and real outputs computed here. Training and dev data only.

    python -m scripts.make_tools_page
"""
import base64
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr

from scripts.simulate_five import load_cases
from src.config import CONFIG
from src.experts import expert_scores, load_fold_predictions, strip_ext
from src.skill_tools import LDCTIQA
from src.tools import body_and_air_masks, nlm_denoise, noise_residual_map, window_view, zoom_view
from src.tools_iqat1 import (fourier_magnitude_spectrum, gradient_orientation_coherence_map,
                             local_gradient_deviation_map, mscn_map)

OUT = CONFIG.output_dir / "tools"
SCREEN = CONFIG.output_dir / "screening"
TEMPLATE = Path(__file__).with_name("tools_page_template.html")
EXAMPLES = {"A_0914": "Temiz (MOS 4.0)", "C_0529": "Streak'li (MOS 2.0)", "D_0302_blur2": "Bulanık (σ=2)"}

# Measures that the agent's tools return, with a short Turkish label and the tool they come from.
MEASURES = {
    "high_pass_std": ("high-pass gürültü", "high_pass_energy"),
    "immerkaer_sigma": ("Immerkær σ", "noise_sigma_estimate"),
    "flat_roi_sigma": ("düz bölge σ", "flat_roi_sigma"),
    "nlm_residual_std": ("NLM residual", "noise_residual_map"),
    "nlm_residual_edge_corr": ("residual–kenar korelasyonu", "noise_residual_map"),
    "nps_high_band_ratio": ("NPS yüksek bant", "radial_nps_summary"),
    "edge_width_px": ("kenar genişliği", "edge_width"),
    "fourier_high_band_fraction": ("Fourier yüksek bant", "fourier_high_band_fraction"),
    "laplacian_variance": ("Laplacian varyansı", "compare_to_reference"),
    "body_coherence": ("doku tutarlılığı", "texture_coherence"),
    "air_hp_std": ("hava enerjisi", "air_artifact_energy"),
    "streak_anisotropy": ("streak anizotropisi", "streak_anisotropy"),
    "orientation_coherence_mean": ("yön tutarlılığı haritası", "orientation_coherence_map"),
    "log_spectrum_mean": ("Fourier spektrumu", "fourier_spectrum"),
    "mscn_kurtosis": ("MSCN basıklığı", "mscn_map"),
    "gradient_deviation_mean": ("gradyan sapması", "gradient_deviation_map"),
}
REFERENCE = ["high_pass_std", "immerkaer_sigma", "nlm_residual_std", "laplacian_variance", "fourier_high_band_fraction",
             "nps_high_band_ratio", "edge_width_px", "air_hp_std", "body_coherence"]


def screening_table() -> pd.DataFrame:
    """The markdown table written by scripts/screen_tools.py."""
    rows = [line.strip().strip("|").split("|") for line in (CONFIG.output_dir / "tool_screening.md").read_text(
        encoding="utf-8").splitlines() if line.startswith("|") and not line.startswith("|:")]
    frame = pd.DataFrame([[c.strip() for c in r] for r in rows[1:]], columns=[c.strip() for c in rows[0]])
    return frame.set_index("metric").apply(pd.to_numeric, errors="coerce")


def level_chart(train: pd.DataFrame, blur: pd.DataFrame, key: str) -> dict:
    """Median per radiologist level, the 1-99% band of unprocessed slices, and where the blurred dev slices land."""
    level = train["mos"].round().clip(0, 4).astype(int)
    values = train[key]
    return {"key": key, "medians": [float(values[level == k].median()) for k in range(5)],
            "p1": float(values.quantile(0.01)), "p99": float(values.quantile(0.99)),
            "min": float(values.min()), "max": float(values.max()),
            "orig": float(blur.loc[blur["sigma"] == 0.0, key].median()),
            "blur1": float(blur.loc[blur["sigma"] == 1.0, key].median()),
            "blur2": float(blur.loc[blur["sigma"] == 2.0, key].median())}


def jpeg(png_or_array, size: int = 240) -> str:
    image = (Image.open(io.BytesIO(png_or_array)) if isinstance(png_or_array, bytes)
             else Image.fromarray(png_or_array)).convert("L")
    image.thumbnail((size, size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def body_centre(image: np.ndarray) -> tuple[int, int]:
    body, _ = body_and_air_masks(image, LDCTIQA.body_min)
    row, col = ndimage.center_of_mass(body)
    return int(row) - 64, int(col) - 64


def example_outputs() -> dict:
    """Real outputs of the image-producing tools on three dev slices: clean, streaks, blurred."""
    cases = {c.case_id: c for c in load_cases()[0] if c.case_id in EXAMPLES}
    soft, narrow = LDCTIQA.windows["soft_tissue"], LDCTIQA.windows["narrow"]
    tools = {
        "window_view": lambda x: window_view(x, *soft),
        "window_view_narrow": lambda x: window_view(x, *narrow),
        "zoom_view": lambda x: zoom_view(x, *body_centre(x), *soft),
        "noise_residual_map": lambda x: noise_residual_map(x, nlm_denoise),
        "orientation_coherence_map": gradient_orientation_coherence_map,
        "fourier_spectrum": fourier_magnitude_spectrum,
        "mscn_map": mscn_map,
        "gradient_deviation_map": local_gradient_deviation_map,
    }
    out = {}
    for name, tool in tools.items():
        out[name] = []
        for case_id, label in EXAMPLES.items():
            result = tool(cases[case_id].image)
            values = {k: round(v, 4) for k, v in result.values.items() if k not in ("level", "width", "row", "col", "size")}
            out[name].append({"label": label, "image": jpeg(result.image_png), "values": values})
    return out


def expert_stats() -> dict:
    train = pd.read_csv(SCREEN / "train.csv", dtype={"image_id": str}).set_index("image_id")
    scores = expert_scores(load_fold_predictions(CONFIG.expert_dir))["train"]
    scores.index = strip_ext(scores.index)
    rho = spearmanr(scores.reindex(train.index), train["mos"]).statistic
    card = (CONFIG.expert_dir / "profile_card.txt").read_text(encoding="utf-8").strip()
    errors = [float(v) for v in re.findall(r"MOS \d: ([+-]\d\.\d+)", card)]
    shift = float(re.search(r"scores ([+-]\d\.\d+) MOS relative", card).group(1))
    preferred = float(re.search(r"prefers the original in (\d+)%", card).group(1))
    table1 = pd.read_csv(CONFIG.output_dir / "tables/table1_baselines.csv").set_index("method")
    table2 = pd.read_csv(CONFIG.output_dir / "tables/table2_baselines.csv", index_col=0)
    return {"rho": float(rho), "errors": errors, "shift": shift, "preferred": preferred, "card": card,
            "overall": float(table1.loc["medsiglip_film", "overall_raw"]), "a2": float(table2.loc["medsiglip_film", "A2"]),
            "r_up": float(table2.loc["medsiglip_film", "R_up"]), "tau5": float(table2.loc["medsiglip_film", "tau5"])}


def reference_strip() -> list[dict]:
    """Percentiles of the clean and blurred example in the reference bank, from the simulation outputs."""
    records = {r["case_id"]: r for r in json.loads((CONFIG.output_dir / "simulation/tool_outputs.json").read_text(
        encoding="utf-8"))}
    rows = []
    for key in REFERENCE:
        row = {"key": key, "label": MEASURES[key][0]}
        for case_id, name in (("A_0914", "clean"), ("D_0302_blur2", "blur")):
            ref = records[case_id]["skills"]["reference_check"]["compare_to_reference"]
            row[name] = ref["values"].get(f"{key}_pct")
            row[f"{name}_out"] = f"{key}=" in ref["text"] and "outside the range" in ref["text"].split(f"{key}=")[1].split(";")[0]
        rows.append(row)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    screen = screening_table()
    detect = pd.read_csv(SCREEN / "smoothing_detection.csv").set_index("metric")
    train = pd.read_csv(SCREEN / "train.csv", dtype={"image_id": str})
    blur = pd.read_csv(SCREEN / "s1_dev.csv", dtype={"image_id": str})
    expert = expert_stats()

    stats = {}
    for key in MEASURES:
        s, d = screen.loc[key], detect.loc[key]
        stats[key] = {"rho": float(s["rho_mos"]), "partial": None if pd.isna(s["partial_hp"]) else float(s["partial_hp"]),
                      "a1": float(s["A1"]), "a2": float(s["A2"]), "r_up": float(s["R_up"]), "tau5": float(s["tau5"]),
                      "auc": float(d["auc_blur"]), "ood": float(d["blur_ood"])}
    scatter = [{"key": k, "label": MEASURES[k][0], "tool": MEASURES[k][1], "rho": abs(v["rho"]), "a2": v["a2"]}
               for k, v in stats.items()]
    scatter.append({"key": "medsiglip_film", "label": "uzman (MedSigLIP-FiLM)", "tool": "consult_experts",
                    "rho": abs(expert["rho"]), "a2": expert["a2"]})
    data = {
        "stats": stats, "scatter": scatter, "expert": expert, "reference": reference_strip(),
        "ood": [{"key": k, "label": MEASURES[k][0], "ood": stats[k]["ood"], "auc": stats[k]["auc"]} for k in REFERENCE],
        "levels": {k: level_chart(train, blur, k) for k in MEASURES if k in train.columns},
        "examples": example_outputs(),
    }
    page = TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
    (OUT / "araclar.html").write_text(page, encoding="utf-8")
    print(f"wrote {OUT / 'araclar.html'} ({(OUT / 'araclar.html').stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
