"""Regression tests for the second-round code review (commit 0f6f988)."""

import numpy as np
import pandas as pd
import pytest

from proxyscore import (
    ProxyAudit,
    Status,
    Verdict,
    check_downstream,
    check_leakage,
    check_segments,
    check_stability,
    psi,
)


def make_strong_pair(n=600, seed=0):
    """Score, clean indicators, and a binary outcome strongly driven by the score."""
    rng = np.random.default_rng(seed)
    score = rng.normal(0, 1, n)
    X = pd.DataFrame(
        {
            "a": score + rng.normal(0, 0.6, n),
            "b": score + rng.normal(0, 0.6, n),
        }
    )
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-2.5 * score))).astype(int)
    return X, pd.Series(score), pd.Series(y)


# --- finding 1: supplied-but-skipped checks cap the verdict -----------------


def test_underpowered_stability_blocks_decision_grade():
    X, score, y = make_strong_pair()
    period = np.array(["m1"] * 20 + ["m2"] * (len(X) - 20))  # baseline too small
    report = ProxyAudit(indicators=X, score=score, outcome=y, period=period).run()
    assert report["downstream"].status is Status.PASS
    assert report["stability"].status is Status.SKIP
    assert report.verdict is not Verdict.DECISION_GRADE
    assert "stability" in report.verdict_reason


def test_unsupplied_inputs_do_not_block_decision_grade():
    X, score, y = make_strong_pair()
    report = ProxyAudit(indicators=X, score=score, outcome=y).run()
    # stability and segments were never supplied: their SKIPs are
    # "not applicable", not missing evidence
    assert report["stability"].status is Status.SKIP
    assert report["segments"].status is Status.SKIP
    assert report.verdict is Verdict.DECISION_GRADE


def test_underpowered_segments_block_decision_grade():
    X, score, y = make_strong_pair()
    segments = np.array(["a"] * 10 + ["b"] * (len(X) - 10))  # one tiny segment
    segments[10:20] = "c"  # second tiny segment, so only one is evaluable
    report = ProxyAudit(indicators=X, score=score, outcome=y, segments=segments).run()
    assert report["segments"].status is Status.SKIP
    assert report.verdict is not Verdict.DECISION_GRADE


# --- finding 2: leakage assessability requires a computable metric ----------


def test_leakage_skips_on_constant_outcome():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(0, 1, 100)})
    res = check_leakage(X, np.zeros(100))
    assert res.status is Status.SKIP
    assert "variation" in res.summary


def test_leakage_single_class_overlap_is_unassessed():
    rng = np.random.default_rng(2)
    n = 200
    y = np.array([1] * 100 + [0] * 100)
    onlypos = np.full(n, np.nan)
    onlypos[:100] = rng.normal(0, 1, 100)  # populated only where y == 1
    X = pd.DataFrame({"good": rng.normal(0, 1, n), "onlypos": onlypos})
    res = check_leakage(X, y)
    assert res.status is Status.WARN
    assert "onlypos" in res.summary
    assert res.metrics["n_unassessed"] == 1


def test_leakage_constant_indicator_is_unassessed():
    rng = np.random.default_rng(3)
    n = 200
    y = rng.integers(0, 2, n)
    X = pd.DataFrame({"good": rng.normal(0, 1, n), "constant": np.ones(n)})
    res = check_leakage(X, y)
    table = res.details.set_index("indicator")
    assert not table.loc["constant", "assessed"]
    assert res.status is Status.WARN


# --- finding 3: unknown stability baseline raises ----------------------------


def test_check_stability_unknown_baseline_raises():
    rng = np.random.default_rng(4)
    score = rng.normal(0, 1, 400)
    period = np.repeat(["m1", "m2"], 200)
    with pytest.raises(ValueError, match="not found"):
        check_stability(score, period, baseline_period="missing")


# --- finding 4: check_downstream validates n_bands ----------------------------


def test_check_downstream_rejects_bad_n_bands():
    _, score, y = make_strong_pair()
    with pytest.raises(ValueError, match="n_bands"):
        check_downstream(score, y, n_bands=1)


def test_check_downstream_always_returns_lift_table_when_graded():
    _, score, y = make_strong_pair()
    res = check_downstream(score, y)
    assert res.details is not None


# --- finding 5: consistent rejection of nonnumeric multiclass outcomes --------


def test_multiclass_string_outcome_rejected_consistently():
    rng = np.random.default_rng(5)
    n = 300
    score = pd.Series(rng.normal(0, 1, n))
    y = pd.Series(rng.choice(["red", "green", "blue"], n))
    seg = pd.Series(rng.choice(["a", "b"], n))
    X = pd.DataFrame({"a": rng.normal(0, 1, n)})
    with pytest.raises(TypeError, match="two-valued"):
        check_downstream(score, y)
    with pytest.raises(TypeError, match="two-valued"):
        check_segments(score, seg, outcome=y)
    with pytest.raises(TypeError, match="two-valued"):
        check_leakage(X, y)


# --- finding 6: infinite scores rejected in stability / standalone APIs --------


def test_psi_rejects_infinite_values():
    with pytest.raises(ValueError, match="infinite"):
        psi([1.0, 2.0, np.inf], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="infinite"):
        psi([1.0, 2.0, 3.0], [1.0, np.inf, 3.0])


def test_check_stability_rejects_infinite_scores():
    score = np.concatenate([np.random.default_rng(6).normal(0, 1, 199), [np.inf]])
    period = np.repeat(["m1", "m2"], 100)
    with pytest.raises(ValueError, match="infinite"):
        check_stability(score, period)


def test_audit_rejects_infinite_score():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="infinite"):
        ProxyAudit(indicators=X, score=[0.1, np.inf, 0.3, 0.4])


def test_check_downstream_rejects_infinite_score():
    _, score, y = make_strong_pair()
    score.iloc[0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        check_downstream(score, y)
