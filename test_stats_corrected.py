"""Unit tests for stats_corrected.py.   Run:  python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from stats_corrected import corrected_resampled_ttest, table6_from_wide


def test_matches_hand_calculation():
    d = np.array([0.02, 0.01, 0.03, -0.01, 0.02, 0.00, 0.04, 0.01])
    n = len(d)
    ratio = 0.25
    se = np.sqrt((1 / n + ratio) * d.var(ddof=1))
    t = d.mean() / se
    p = 2 * stats.t.sf(abs(t), n - 1)
    r = corrected_resampled_ttest(d, ratio)
    assert r["t"] == pytest.approx(t)
    assert r["p"] == pytest.approx(p)
    assert r["df"] == n - 1
    half = stats.t.ppf(0.975, n - 1) * se
    assert r["ci_low"] == pytest.approx(d.mean() - half)
    assert r["ci_high"] == pytest.approx(d.mean() + half)


def test_ci_excludes_zero_iff_p_below_alpha():
    rng = np.random.default_rng(1)
    for _ in range(50):
        d = rng.normal(rng.uniform(-0.05, 0.05), 0.03, 100)
        r = corrected_resampled_ttest(d, 0.25)
        assert (r["ci_low"] > 0 or r["ci_high"] < 0) == (r["p"] < 0.05)


def test_correction_is_more_conservative_than_naive():
    rng = np.random.default_rng(2)
    d = rng.normal(0.02, 0.03, 100)
    naive = stats.ttest_1samp(d, 0).pvalue
    corr = corrected_resampled_ttest(d, 0.25)["p"]
    assert corr > naive
    # ratio 0 reduces to the ordinary one-sample t-test
    assert corrected_resampled_ttest(d, 0.0)["p"] == pytest.approx(naive)


def test_sign_and_counts():
    d = np.array([0.1, 0.2, -0.05, 0.3])
    r = corrected_resampled_ttest(d, 0.25)
    assert r["mean_diff"] > 0 and r["seeds_first_higher"] == 3


def test_zero_variance_and_bad_input():
    assert corrected_resampled_ttest(np.full(10, 0.01), 0.25)["p"] < 1e-12
    with pytest.raises(ValueError):
        corrected_resampled_ttest([0.1, 0.2], 0.25)
    with pytest.raises(ValueError):
        corrected_resampled_ttest(np.ones(10), -1)


def test_table6_has_six_rows_and_right_direction():
    rng = np.random.default_rng(3)
    base = rng.normal(0.77, 0.02, 100)
    data = {}
    for m in ["auroc", "balanced_accuracy", "f1_malignant"]:
        data[f"{m}_image"] = base
        data[f"{m}_inferred"] = base - 0.01 + rng.normal(0, 0.03, 100)
        data[f"{m}_patient"] = base - 0.02 + rng.normal(0, 0.03, 100)
    tab = table6_from_wide(pd.DataFrame(data), test_size=0.2)
    assert len(tab) == 6
    assert (tab["test_train_ratio"] == 0.25).all()
    assert (tab["mean_diff"] > 0).all()


def test_reproduces_manuscript_table6_from_reported_intervals():
    """Consistency check of the numbers printed in the manuscript's Table 6: the reported
    corrected CI implies a standard error, which must give the reported t and p."""
    rows = [  # mean, ci_low, ci_high, t, p  (manuscript Table 6)
        (0.0225, -0.0112, 0.0562, 1.33, 0.188),
        (0.0181, -0.0189, 0.0551, 0.97, 0.334),
        (0.0287, -0.0260, 0.0834, 1.04, 0.301),
        (0.0099, -0.0267, 0.0466, 0.54, 0.592),
        (0.0059, -0.0306, 0.0424, 0.32, 0.748),
        (0.0096, -0.0441, 0.0633, 0.35, 0.724),
    ]
    tcrit = stats.t.ppf(0.975, 99)
    for m, lo, hi, t_rep, p_rep in rows:
        se = (hi - lo) / 2 / tcrit
        assert m / se == pytest.approx(t_rep, abs=0.015)
        assert 2 * stats.t.sf(abs(m / se), 99) == pytest.approx(p_rep, abs=0.002)
