import numpy as np
import pandas as pd

from proxyscore import Status, check_leakage, check_segments, leakage_scan, segment_summary


def test_segment_summary_levels():
    rng = np.random.default_rng(0)
    score = np.concatenate([rng.normal(0, 1, 500), rng.normal(1.0, 1, 500)])
    seg = np.repeat(["a", "b"], 500)
    table = segment_summary(score, seg)
    assert table.loc["b", "score_mean"] > table.loc["a", "score_mean"]
    assert abs(table.loc["b", "smd_vs_rest"]) > 0.5


def test_check_segments_warns_on_level_gap():
    rng = np.random.default_rng(1)
    score = np.concatenate([rng.normal(0, 1, 500), rng.normal(1.0, 1, 500)])
    seg = np.repeat(["a", "b"], 500)
    res = check_segments(score, seg)
    assert res.status is Status.WARN


def test_check_segments_fails_when_score_useless_in_segment():
    rng = np.random.default_rng(2)
    n = 1000
    score = rng.normal(0, 1, n)
    seg = np.repeat(["works", "broken"], n // 2)
    y = np.empty(n)
    # score predicts outcome in segment "works", pure noise in "broken"
    logit = 2.5 * score[: n // 2]
    y[: n // 2] = (rng.uniform(size=n // 2) < 1 / (1 + np.exp(-logit))).astype(int)
    y[n // 2 :] = rng.integers(0, 2, n // 2)
    res = check_segments(score, seg, outcome=y)
    assert res.status in (Status.WARN, Status.FAIL)


def test_check_segments_passes_consistent():
    rng = np.random.default_rng(3)
    n = 1200
    score = rng.normal(0, 1, n)
    seg = rng.choice(["a", "b", "c"], n)
    logit = 2.0 * score
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    res = check_segments(score, seg, outcome=y)
    assert res.status is Status.PASS


def test_check_segments_skips_tiny_segments():
    res = check_segments([1.0, 2.0, 3.0, 4.0], ["a", "a", "b", "b"])
    assert res.status is Status.SKIP


def test_leakage_scan_flags_leaky_indicator():
    rng = np.random.default_rng(4)
    n = 1000
    y = rng.integers(0, 2, n)
    X = pd.DataFrame(
        {
            "honest": rng.normal(0, 1, n) + 0.3 * y,
            "leaky": y + rng.normal(0, 0.05, n),
        }
    )
    table = leakage_scan(X, y)
    assert bool(table.loc["leaky", "statistical_flag"])
    assert not bool(table.loc["honest", "statistical_flag"])
    res = check_leakage(X, y)
    assert res.status is Status.FAIL
    assert "leaky" in res.summary


def test_leakage_name_heuristic():
    rng = np.random.default_rng(5)
    n = 500
    y = rng.integers(0, 2, n)
    X = pd.DataFrame(
        {
            "logins": rng.normal(0, 1, n),
            "days_since_renewal_call": rng.normal(0, 1, n),
        }
    )
    res = check_leakage(X, y)
    assert res.status is Status.WARN
    assert "renewal" in res.summary


def test_leakage_clean_passes():
    rng = np.random.default_rng(6)
    n = 500
    y = rng.integers(0, 2, n)
    X = pd.DataFrame({"logins": rng.normal(0, 1, n), "nps": rng.normal(0, 1, n)})
    assert check_leakage(X, y).status is Status.PASS
