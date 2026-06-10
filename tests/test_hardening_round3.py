"""Regression tests for the third-round code review (findings 7-11)."""

import numpy as np
import pandas as pd
import pytest

from proxyscore import (
    Status,
    check_downstream,
    check_leakage,
    check_segments,
    psi,
    segment_summary,
)

# --- finding 7: PSI must see shifts in BOTH directions from a constant baseline


def test_psi_constant_baseline_detects_positive_shift():
    assert psi(np.full(200, 5.0), np.full(200, 6.0)) > 0.25


def test_psi_constant_baseline_detects_negative_shift():
    assert psi(np.full(200, 5.0), np.full(200, 4.0)) > 0.25


def test_psi_constant_baseline_unchanged_is_stable():
    assert psi(np.full(200, 5.0), np.full(200, 5.0)) < 0.01


def test_psi_two_valued_baseline_still_discriminates():
    base = np.array([0.0, 1.0] * 100)
    same = np.array([0.0, 1.0] * 100)
    shifted = np.array([0.0] * 20 + [1.0] * 180)
    assert psi(base, same) < 0.01
    assert psi(base, shifted) > 0.25


# --- finding 8: indicator named 'outcome' must not crash the leakage scan ----


def test_leakage_indicator_named_outcome_does_not_crash():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n)
    X = pd.DataFrame({"outcome": rng.normal(0, 1, n), "logins": rng.normal(0, 1, n)})
    res = check_leakage(X, y)
    # runs, and the suspicious name is still flagged by the name heuristic
    assert res.status is Status.WARN
    assert "outcome" in res.summary


def test_leakage_indicator_named_outcome_statistical_path():
    rng = np.random.default_rng(1)
    n = 500
    y = rng.integers(0, 2, n)
    leaky = y + rng.normal(0, 0.05, n)
    X = pd.DataFrame({"outcome": leaky})
    res = check_leakage(X, y)
    assert res.status is Status.FAIL  # statistical flag computed, not crashed


# --- finding 9: SMD uses sample-size-weighted pooled variance -----------------


def test_smd_uses_weighted_pooled_variance():
    rng = np.random.default_rng(2)
    small = rng.normal(1.0, 3.0, 100)  # small, high-variance segment
    large = rng.normal(0.0, 1.0, 900)
    score = np.concatenate([small, large])
    seg = np.array(["small"] * 100 + ["large"] * 900)
    table = segment_summary(score, seg)

    n1, n2 = 100, 900
    var1 = pd.Series(small).var(ddof=1)
    var2 = pd.Series(large).var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    expected = (small.mean() - large.mean()) / pooled
    assert table.loc["small", "smd_vs_rest"] == pytest.approx(expected)
    # the unweighted average would understate this SMD noticeably
    unweighted = (small.mean() - large.mean()) / np.sqrt((var1 + var2) / 2)
    assert abs(expected) > abs(unweighted)


# --- finding 10: NaN validity is reported as unassessed, not dropped ----------


def test_constant_score_segment_is_reported_not_dropped():
    rng = np.random.default_rng(3)
    n = 200
    score = np.concatenate([rng.normal(0, 1, n // 2), np.full(n // 2, 3.0)])
    seg = np.repeat(["varied", "flat"], n // 2)
    outcome = 0.5 * score + rng.normal(0, 0.5, n)  # continuous outcome
    res = check_segments(score, seg, outcome=outcome)
    assert res.status is Status.WARN
    assert "flat" in res.summary
    assert "could not be assessed" in res.summary


# --- finding 11: constant outcome SKIPs downstream with a clear message --------


def test_downstream_constant_outcome_skips():
    rng = np.random.default_rng(4)
    score = pd.Series(rng.normal(0, 1, 200))
    res = check_downstream(score, np.zeros(200))
    assert res.status is Status.SKIP
    assert "no variation" in res.summary
