import numpy as np
import pandas as pd

from proxyscore import (
    Status,
    check_downstream,
    check_stability,
    downstream_validity,
    lift_table,
    psi,
    psi_over_time,
)

# --- PSI ---------------------------------------------------------------


def test_psi_same_distribution_near_zero():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
    assert psi(a, b) < 0.02


def test_psi_shifted_distribution_large():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 5000), rng.normal(1.0, 1, 5000)
    assert psi(a, b) > 0.25


def test_psi_over_time_rows():
    rng = np.random.default_rng(1)
    score = rng.normal(0, 1, 900)
    period = np.repeat(["m1", "m2", "m3"], 300)
    table = psi_over_time(score, period)
    assert list(table["period"]) == ["m2", "m3"]


def test_check_stability_pass_and_fail():
    rng = np.random.default_rng(2)
    stable = np.concatenate([rng.normal(0, 1, 500), rng.normal(0, 1, 500)])
    period = np.repeat(["m1", "m2"], 500)
    assert check_stability(stable, period).status is Status.PASS

    drifted = np.concatenate([rng.normal(0, 1, 500), rng.normal(1.5, 1, 500)])
    res = check_stability(drifted, period)
    assert res.status is Status.FAIL
    assert res.metrics["max_psi"] > 0.25


def test_check_stability_skips_single_period():
    res = check_stability(np.arange(100.0), ["m1"] * 100)
    assert res.status is Status.SKIP


# --- downstream validation ----------------------------------------------


def make_scored_outcome(n=2000, signal=2.0, seed=3, invert=False):
    rng = np.random.default_rng(seed)
    score = rng.normal(0, 1, n)
    logit = signal * score * (-1 if invert else 1)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.Series(score), pd.Series(y)


def test_downstream_strong_binary():
    score, y = make_scored_outcome()
    res = check_downstream(score, y)
    assert res.status is Status.PASS
    assert res.metrics["auc_oriented"] > 0.65


def test_downstream_detects_negative_polarity():
    score, y = make_scored_outcome(invert=True)
    m = downstream_validity(score, y)
    assert m["polarity"] == -1
    assert m["auc_oriented"] > 0.65
    assert check_downstream(score, y).status is Status.PASS


def test_downstream_fails_on_noise():
    rng = np.random.default_rng(4)
    score = pd.Series(rng.normal(0, 1, 2000))
    y = pd.Series(rng.integers(0, 2, 2000))
    assert check_downstream(score, y).status is Status.FAIL


def test_downstream_continuous_outcome():
    rng = np.random.default_rng(5)
    score = pd.Series(rng.normal(0, 1, 1000))
    y = 0.5 * score + rng.normal(0, 1, 1000)
    res = check_downstream(score, y)
    assert res.metrics["outcome_type"] == "continuous"
    assert res.status is Status.PASS


def test_downstream_skips_tiny_sample():
    assert check_downstream(pd.Series([1.0, 2.0]), pd.Series([0, 1])).status is Status.SKIP


def test_lift_table_monotone_signal():
    score, y = make_scored_outcome(signal=3.0)
    lt = lift_table(score, y, n_bands=10)
    assert len(lt) == 10
    # band 1 = highest scores = highest outcome rate for positive signal
    assert lt.iloc[0]["outcome_rate"] > lt.iloc[-1]["outcome_rate"]
    assert lt.iloc[0]["lift"] > 1.5
