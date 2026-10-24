import numpy as np
import pandas as pd
import pytest

from src.experts import assert_out_of_fold, expert_scores, profile_card


def fake_predictions(n_train: int = 50, n_folds: int = 5, leak: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    keys = [f"{i:04d}.tif" for i in range(n_train)]
    rows = []
    for fold in range(1, n_folds + 1):
        held = keys[(fold - 1) * n_train // n_folds: fold * n_train // n_folds]
        rows += [("train", k, fold, rng.uniform(0, 4)) for k in keys]
        rows += [("oof", k, fold, rng.uniform(0, 4)) for k in held]
        rows += [("test", f"test{j:03d}.tiff", fold, float(fold)) for j in range(3)]
    if leak:
        rows.append(("oof", keys[0], 2, 1.0))
    return pd.DataFrame(rows, columns=["set", "key", "fold", "pred"])


def test_out_of_fold_check_passes_and_catches_leaks():
    assert_out_of_fold(fake_predictions(), n_train=50)
    with pytest.raises(AssertionError):
        assert_out_of_fold(fake_predictions(leak=True), n_train=50)


def test_scores_are_oof_for_train_and_fold_mean_elsewhere():
    preds = fake_predictions()
    scores = expert_scores(preds)
    assert len(scores["train"]) == 50
    assert scores["train"].loc["0000.tif"] == preds[(preds.set == "oof") & (preds.key == "0000.tif")].pred.item()
    assert scores["test"].loc["test000.tiff"] == pytest.approx(3.0)  # mean of folds 1..5


def test_profile_card_reports_blur_preference_from_data():
    index = [f"{i}" for i in range(30)]
    mos = pd.Series(np.linspace(0, 4, 30), index=index)
    pred = mos + 0.1
    noise = pd.Series(np.linspace(1, 0, 30), index=index)
    original = pd.Series([2.0, 3.0], index=["a", "b"])
    rewards = profile_card(pred, mos, noise, original, original + 1.0, np.array([4.0, 0.0, 2.0, 1.0]))
    penalises = profile_card(pred, mos, noise, original, original - 1.0, np.array([4.0, 0.0, 2.0, 1.0]))
    assert "rewards smoothing" in rewards and "prefers the original in 0%" in rewards
    assert "does not reward smoothing" in penalises


def test_expert_tool_maps_score_and_returns_card():
    from src.agent import Case
    from src.skill_tools import expert_tool
    tool = expert_tool("card text", np.array([4.0, 0.0, 2.0, 1.0]))
    result = tool.run(Case("x", np.zeros((4, 4)), {"medsiglip_film": 2.0}))
    assert result.values["medsiglip_film"] == pytest.approx(2.0) and result.text == "card text"
    assert tool.run(Case("y", np.zeros((4, 4)))).values == {}
