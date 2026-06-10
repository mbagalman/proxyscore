"""Regression tests for the fourth-round code review (commit dcd781a)."""

import numpy as np
import pandas as pd
import pytest

from proxyscore import (
    CompositeScore,
    ProxyAudit,
    Status,
    Verdict,
    check_downstream,
    check_indicators,
    check_leakage,
    check_segments,
    check_stability,
    psi,
    psi_over_time,
)

# --- finding 1: excluded small segments must cap the verdict -----------------


def test_excluded_small_segment_warns():
    rng = np.random.default_rng(0)
    score = rng.normal(0, 1, 500)
    seg = np.array(["a"] * 240 + ["b"] * 240 + ["tiny"] * 20)
    res = check_segments(score, seg)
    assert res.status is Status.WARN
    assert "tiny" in res.summary
    assert "excluded" in res.summary


def test_excluded_small_segment_blocks_decision_grade():
    rng = np.random.default_rng(1)
    n = 500
    score = rng.normal(0, 1, n)
    X = pd.DataFrame(
        {"a": score + rng.normal(0, 0.6, n), "b": score + rng.normal(0, 0.6, n)}
    )
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-2.5 * score))).astype(int)
    seg = np.array(["big1"] * 240 + ["big2"] * 240 + ["tiny"] * 20)
    report = ProxyAudit(indicators=X, score=score, outcome=y, segments=seg).run()
    assert report["segments"].status is Status.WARN
    assert report.verdict is not Verdict.DECISION_GRADE
    assert "segments" in report.verdict_reason


# --- finding 2: rank scaling must not invent values for unfitted columns -----


def test_rank_transform_of_unfitted_column_is_nan():
    train = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [np.nan] * 4})
    new = pd.DataFrame({"a": [2.5, 3.5], "b": [10.0, 20.0]})
    cs = CompositeScore(scaling="rank", min_coverage=1.0).fit(train)
    s = cs.transform(new)
    assert s.isna().all()  # column b has no learned reference distribution


def test_all_missing_indicator_fails_quality_check():
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"a": rng.normal(0, 1, 100), "empty": [np.nan] * 100})
    res = check_indicators(X)
    assert res.status is Status.FAIL
    assert "empty" in res.summary


# --- finding 3: leakage validates inputs before the constant-outcome return ---


def test_leakage_constant_outcome_wrong_length_raises():
    X = pd.DataFrame({"a": np.arange(8.0)})
    with pytest.raises(ValueError, match="length"):
        check_leakage(X, [0.0])


def test_leakage_constant_outcome_mismatched_index_raises():
    X = pd.DataFrame({"a": np.arange(8.0)}, index=range(8))
    y = pd.Series(np.zeros(8), index=range(100, 108))
    with pytest.raises(ValueError, match="index does not match"):
        check_leakage(X, y)


def test_leakage_infinite_outcome_raises():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"a": rng.normal(0, 1, 50)})
    y = rng.normal(0, 1, 50)
    y[0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        check_leakage(X, y)


# --- finding 4: scores must be numeric at every public boundary ---------------


def test_string_score_rejected_everywhere():
    n = 60
    rng = np.random.default_rng(4)
    str_score = pd.Series(["high", "low"] * (n // 2))
    y = rng.integers(0, 2, n)
    X = pd.DataFrame({"a": rng.normal(0, 1, n)})
    with pytest.raises(TypeError, match="numeric"):
        check_downstream(str_score, y)
    with pytest.raises(TypeError, match="numeric"):
        check_segments(str_score, ["a", "b"] * (n // 2), outcome=y)
    with pytest.raises(TypeError, match="numeric"):
        check_stability(str_score, ["m1", "m2"] * (n // 2))
    with pytest.raises(TypeError, match="numeric"):
        ProxyAudit(indicators=X, score=str_score)
    with pytest.raises(TypeError, match="numeric"):
        check_indicators(X, score=str_score)


# --- finding 5: mixed-type binary labels rejected with a clear error ----------


def test_mixed_type_binary_labels_rejected():
    rng = np.random.default_rng(5)
    score = pd.Series(rng.normal(0, 1, 100))
    y = pd.Series([1, "yes"] * 50)
    with pytest.raises(TypeError, match="orderable"):
        check_downstream(score, y)


def test_homogeneous_string_labels_still_work():
    rng = np.random.default_rng(6)
    n = 200
    score = rng.normal(0, 1, n)
    y = np.where(score > 0, "yes", "no")
    res = check_downstream(pd.Series(score), pd.Series(y))
    assert res.status is not Status.FAIL


# --- finding 6: bins must be an integer across all PSI entry points -----------


def test_psi_rejects_non_integer_bins():
    a = np.arange(100.0)
    with pytest.raises(ValueError, match="integer"):
        psi(a, a, bins=2.5)
    with pytest.raises(ValueError, match="integer"):
        psi_over_time(a, ["m1", "m2"] * 50, bins=2.5)
    with pytest.raises(ValueError, match="integer"):
        check_stability(a, ["m1", "m2"] * 50, bins=2.5)
    with pytest.raises(ValueError, match="integer"):
        psi(a, a, bins=True)


def test_numpy_integer_bins_accepted():
    rng = np.random.default_rng(7)
    a, b = rng.normal(0, 1, 500), rng.normal(0, 1, 500)
    assert psi(a, b, bins=np.int64(10)) < 0.05
