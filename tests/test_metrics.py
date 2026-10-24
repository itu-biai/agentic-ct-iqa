import numpy as np
import pandas as pd
import pytest

from src.metrics import challenge_scores, dose_ladder_tau, fit_logistic, logistic4, ndct_above_rate, preference_accuracy


def test_perfect_prediction_gives_overall_three():
    mos = np.linspace(0, 4, 50)
    assert challenge_scores(mos, mos)["overall"] == pytest.approx(3.0)


def test_logistic_recovers_monotone_map():
    rng = np.random.default_rng(0)
    raw = rng.uniform(-3, 3, 200)
    mos = logistic4(raw, 4.0, 0.0, 0.5, 0.8)
    params = fit_logistic(raw, mos)
    assert np.allclose(logistic4(raw, *params), mos, atol=1e-3)
    assert challenge_scores(raw, mos, params)["plcc"] == pytest.approx(1.0, abs=1e-6)


def test_perfect_dose_ladder_gives_tau_one():
    frame = pd.DataFrame({"slice_id": np.repeat(["a", "b"], 6), "dose_rank": np.tile(np.arange(6), 2),
                          "score": np.tile(np.arange(6) * 0.5, 2)})
    assert dose_ladder_tau(frame) == pytest.approx(1.0)


def test_ties_count_as_wrong():
    original = pd.Series([3.0, 2.0, 1.0], index=["x", "y", "z"])
    degraded = pd.Series([2.0, 2.0, 1.5], index=["x", "y", "z"])
    assert preference_accuracy(original, degraded) == pytest.approx(1 / 3)


def test_ndct_above_rate_excludes_noisy():
    frame = pd.DataFrame({"group_id": ["g"] * 4, "method": ["gt", "noisy", "tv", "redcnn"],
                          "score": [3.0, 3.5, 2.0, 3.5]})
    assert ndct_above_rate(frame) == pytest.approx(0.5)
