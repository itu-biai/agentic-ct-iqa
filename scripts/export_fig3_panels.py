"""Fig 3 panels (issue #329, E18): for each of the 4 cases in colab/vft_job/lab329.npz the slice in the radiologists'
soft-tissue window (350 / 40) and its Stage B occlusion map (outputs/agentic_ct_iqa/i329/occlusion_case<n>.npy, written
by scripts/issue329_eval.py) as an overlay, PNG for TikZ; plus fig3_cases.csv with MOS, Stage B, MedSigLIP, the agent's
score, the tool files it read and its rationale (test cases).

    python -m scripts.export_fig3_panels     # -> outputs/agentic_ct_iqa/i329/fig3/
"""
import numpy as np
import pandas as pd
from matplotlib import colormaps
from PIL import Image

from src.config import CONFIG, PROJECT_ROOT

I329 = CONFIG.output_dir / "i329"
OUT = I329 / "fig3"


def soft_tissue(image: np.ndarray) -> np.ndarray:
    hu = image.astype(np.float32) / 255 * 1400 - 1000
    return (np.clip((hu - (40 - 175)) / 350, 0, 1) * 255).astype(np.uint8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lab = np.load(PROJECT_ROOT / "colab/vft_job/lab329.npz")
    agent = pd.read_csv(I329 / "agent_per_image.csv").set_index("test_id")
    reads = pd.read_csv(I329 / "agent_reads.csv")
    test = pd.read_csv(I329 / "test_predictions.csv", index_col=0)
    rows = []
    for n, name in enumerate(lab["occl_ids"]):
        image = soft_tissue(lab["occl_images"][n])
        Image.fromarray(image).save(OUT / f"case{n + 1}_slice.png")
        drop = np.load(I329 / f"occlusion_case{n + 1}.npy")
        scale = np.abs(drop).max() or 1
        heat = np.asarray(Image.fromarray(((drop / scale + 1) / 2 * 255).astype(np.uint8)).resize((512, 512), Image.BILINEAR))
        rgb = (colormaps["RdBu_r"](heat / 255)[..., :3] * 255).astype(np.uint8)
        overlay = (0.55 * np.repeat(image[..., None], 3, 2) + 0.45 * rgb).astype(np.uint8)
        Image.fromarray(overlay).save(OUT / f"case{n + 1}_occlusion.png")
        key = str(name).split(": ")[1]
        row = {"case": str(name), "occlusion |max|": scale}
        if key in test.index:
            row.update({"MOS": test.loc[key, "mos"], "Stage B": test.loc[key, "T6"], "MedSigLIP": test.loc[key, "MedSigLIP"],
                        "agent": test.loc[key, "Z3"], "agent read": ", ".join(sorted(set(reads.loc[reads["test_id"] == key, "tool"]))),
                        "rationale": agent.loc[key, "rationale"]})
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "fig3_cases.csv", index=False)
    print(pd.DataFrame(rows).drop(columns="rationale", errors="ignore").to_string())


if __name__ == "__main__":
    main()
