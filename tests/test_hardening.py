"""Regression tests for the issues found in the 0.1.0 code review."""

import numpy as np
import pandas as pd
import pytest

from proxyscore import (
    CompositeScore,
    PCAScore,
    ProxyAudit,
    Status,
    Thresholds,
    check_downstream,
    check_leakage,
    check_segments,
    check_stability,
    cronbach_alpha,
    lift_table,
    psi,
)

# --- finding 1: index / length alignment ---------------------------------


def test_audit_rejects_mismatched_series_index():
    X = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 3, 4, 5]}, index=[10, 11, 12, 13])
    score = pd.Series([0.1, 0.2, 0.3, 0.4])  # RangeIndex, different labels
    with pytest.raises(ValueError, match="index does not match"):
        ProxyAudit(indicators=X, score=score)


def test_audit_rejects_reordered_series_index():
    X = pd.DataFrame({"a": [1.0, 2, 3]}, index=[0, 1, 2])
    score = pd.Series([0.1, 0.2, 0.3], index=[2, 1, 0])
    with pytest.raises(ValueError, match="index does not match"):
        ProxyAudit(indicators=X, score=score)


def test_audit_rejects_wrong_length_array():
    X = pd.DataFrame({"a": [1.0, 2, 3, 4]})
    with pytest.raises(ValueError, match="length"):
        ProxyAudit(indicators=X, outcome=[0, 1])


def test_audit_rejects_duplicate_index():
    X = pd.DataFrame({"a": [1.0, 2, 3]}, index=[0, 0, 1])
    with pytest.raises(ValueError, match="duplicate"):
        ProxyAudit(indicators=X)


def test_standalone_check_rejects_mismatched_outcome():
    score = pd.Series(np.arange(100.0))
    outcome = pd.Series(np.arange(100) % 2, index=range(100, 200))
    with pytest.raises(ValueError, match="index does not match"):
        check_downstream(score, outcome)


def test_array_inputs_of_matching_length_still_work():
    rng = np.random.default_rng(0)
    score = rng.normal(0, 1, 200)
    y = (rng.uniform(size=200) < 1 / (1 + np.exp(-2 * score))).astype(int)
    res = check_downstream(score, y)
    assert res.status in (Status.PASS, Status.WARN)


# --- finding 2: missing data must not become valid scores ------------------


def test_rank_scaling_missing_stays_missing():
    X = pd.DataFrame({"a": [1.0, 2, 3, 4, np.nan], "b": [1.0, 2, 3, 4, 5]})
    s = CompositeScore(scaling="rank", min_coverage=1.0).fit_transform(X)
    assert np.isnan(s.iloc[4])
    assert s.iloc[:4].notna().all()


def test_composite_all_missing_row_is_nan():
    X = pd.DataFrame({"a": [1.0, 2, np.nan], "b": [1.0, 2, np.nan]})
    s = CompositeScore().fit_transform(X)
    assert np.isnan(s.iloc[2])


def test_composite_below_coverage_is_nan():
    X = pd.DataFrame({"a": [1.0, 2, 3, np.nan], "b": [1.0, 2, 3, 4]})
    s = CompositeScore(min_coverage=0.8).fit_transform(X)
    assert np.isnan(s.iloc[3])  # only 50% of weight observed


def test_composite_partial_row_renormalized():
    X = pd.DataFrame({"a": [0.0, 1, 2, 4], "b": [0.0, 1, 2, np.nan]})
    s = CompositeScore(min_coverage=0.5).fit_transform(X)
    # row 3 scored from 'a' alone, renormalized: equals a's z-score
    expected = (4 - X["a"].mean()) / X["a"].std(ddof=0)
    assert s.iloc[3] == pytest.approx(expected)


def test_pca_incomplete_row_is_nan():
    rng = np.random.default_rng(1)
    latent = rng.normal(0, 1, 100)
    X = pd.DataFrame(
        {
            "a": latent + rng.normal(0, 0.3, 100),
            "b": latent + rng.normal(0, 0.3, 100),
            "c": latent + rng.normal(0, 0.3, 100),
        }
    )
    X.iloc[0, 0] = np.nan
    s = PCAScore().fit(X).transform(X)
    assert np.isnan(s.iloc[0])
    assert s.iloc[1:].notna().all()


# --- finding 3: segment validity needs outcome evidence --------------------


def test_segment_without_outcomes_is_not_consistent():
    rng = np.random.default_rng(2)
    n = 200
    score = rng.normal(0, 1, n)
    seg = np.repeat(["covered", "uncovered"], n // 2)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-2 * score))).astype(float)
    y[n // 2 :] = np.nan  # no outcomes at all in the second segment
    res = check_segments(score, seg, outcome=y)
    assert res.status is Status.WARN
    assert "uncovered" in res.summary


def test_segment_single_class_outcome_does_not_crash():
    rng = np.random.default_rng(3)
    n = 200
    score = rng.normal(0, 1, n)
    seg = np.repeat(["a", "b"], n // 2)
    y = (rng.uniform(size=n) < 0.4).astype(float)
    y[n // 2 :] = 0.0  # segment b: only negatives
    res = check_segments(score, seg, outcome=y)
    assert res.status in (Status.WARN, Status.FAIL)


# --- finding 4: binary validation needs events ------------------------------


def test_downstream_skips_single_event():
    score = pd.Series(np.linspace(0, 1, 100))
    y = pd.Series([0] * 99 + [1])  # one positive, placed at the top score
    res = check_downstream(score, y)
    assert res.status is Status.SKIP
    assert "negative" in res.summary


def test_downstream_class_minimum_configurable():
    rng = np.random.default_rng(4)
    score = pd.Series(rng.normal(0, 1, 200))
    y = pd.Series([0] * 185 + [1] * 15)
    assert check_downstream(score, y).status is not Status.SKIP
    strict = Thresholds(min_class_count=20)
    assert check_downstream(score, y, strict).status is Status.SKIP


# --- finding 5: diagnostics are honestly listwise ---------------------------


def test_cronbach_alpha_is_listwise_not_imputed():
    rng = np.random.default_rng(5)
    latent = rng.normal(0, 1, 400)
    X = pd.DataFrame({f"x{i}": latent + rng.normal(0, 0.6, 400) for i in range(4)})
    X_missing = X.copy()
    X_missing.iloc[:150, 0] = np.nan  # heavy missingness in one item
    assert cronbach_alpha(X_missing) == pytest.approx(
        cronbach_alpha(X.iloc[150:]), abs=1e-12
    )


# --- finding 6: string-labeled binary outcomes ------------------------------


def test_downstream_string_binary_outcome():
    rng = np.random.default_rng(6)
    n = 600
    score = rng.normal(0, 1, n)
    y = np.where(rng.uniform(size=n) < 1 / (1 + np.exp(-2 * score)), "yes", "no")
    res = check_downstream(pd.Series(score), pd.Series(y))
    assert res.status is Status.PASS
    assert res.details is not None  # lift table computed, not crashed
    assert res.metrics["auc_oriented"] > 0.65


def test_lift_table_string_binary_outcome():
    score = list(range(100))
    y = ["yes" if i >= 50 else "no" for i in range(100)]
    lt = lift_table(score, y, n_bands=4)
    assert lt.iloc[0]["outcome_rate"] == 1.0  # top band all "yes"


def test_lift_table_rejects_multivalued_strings():
    with pytest.raises(TypeError):
        lift_table(list(range(30)), ["a", "b", "c"] * 10, n_bands=2)


# --- finding 7: unassessed leakage is not a clean leakage result -------------


def test_leakage_skips_when_nothing_assessable():
    X = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, 6, 7, 8]})
    y = [0, 1, 0, 1, 0, 1, 0, 1]
    res = check_leakage(X, y)
    assert res.status is Status.SKIP


def test_leakage_warns_on_unassessable_indicator():
    rng = np.random.default_rng(7)
    n = 200
    y = rng.integers(0, 2, n)
    sparse = np.full(n, np.nan)
    sparse[:5] = rng.normal(0, 1, 5)
    X = pd.DataFrame({"good": rng.normal(0, 1, n), "sparse": sparse})
    res = check_leakage(X, y)
    assert res.status is Status.WARN
    assert "sparse" in res.summary
    assert res.metrics["n_unassessed"] == 1


# --- finding 8: weight validation --------------------------------------------


def test_unknown_weight_key_rejected():
    X = pd.DataFrame({"a": [1.0, 2, 3]})
    with pytest.raises(ValueError, match="not in the indicators"):
        CompositeScore(weights={"typo": 1.0}).fit(X)


def test_zero_total_weight_rejected():
    X = pd.DataFrame({"a": [1.0, 2, 3], "b": [3.0, 2, 1]})
    with pytest.raises(ValueError, match="zero"):
        CompositeScore(weights={"a": 0.0, "b": 0.0}).fit(X)


def test_non_finite_weight_rejected():
    X = pd.DataFrame({"a": [1.0, 2, 3], "b": [3.0, 2, 1]})
    with pytest.raises(ValueError, match="finite"):
        CompositeScore(weights={"a": float("nan")}).fit(X)


# --- finding 9: thresholds and parameter validation ---------------------------


def test_thresholds_reject_equal_psi_bands():
    with pytest.raises(ValueError):
        Thresholds(psi_stable=0.25, psi_unstable=0.25)


def test_thresholds_reject_reversed_auc():
    with pytest.raises(ValueError):
        Thresholds(min_auc_weak=0.8, min_auc_strong=0.6)


def test_psi_rejects_one_bin():
    with pytest.raises(ValueError):
        psi([1.0, 2.0], [1.0, 2.0], bins=1)


def test_lift_table_rejects_one_band():
    with pytest.raises(ValueError):
        lift_table(list(range(50)), [0, 1] * 25, n_bands=1)


# --- finding 10: PSI sample-size guard ----------------------------------------


def test_stability_skips_tiny_baseline():
    rng = np.random.default_rng(8)
    score = np.concatenate([rng.normal(0, 1, 20), rng.normal(0, 1, 500)])
    period = np.array(["m1"] * 20 + ["m2"] * 500)
    res = check_stability(score, period)
    assert res.status is Status.SKIP


def test_stability_excludes_underpowered_period_from_grading():
    rng = np.random.default_rng(9)
    score = np.concatenate(
        [rng.normal(0, 1, 500), rng.normal(0, 1, 500), rng.normal(5.0, 1, 20)]
    )
    period = np.array(["m1"] * 500 + ["m2"] * 500 + ["m3"] * 20)
    res = check_stability(score, period)
    assert res.status is Status.PASS  # the shifted period is too small to grade
    assert any("m3" in n for n in res.notes)


# --- finding 11: infinity rejected at the boundary ------------------------------


def test_infinite_indicator_rejected():
    X = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
    with pytest.raises(ValueError, match="infinite"):
        ProxyAudit(indicators=X)


def test_infinite_indicator_rejected_in_constructors():
    X = pd.DataFrame({"a": [1.0, -np.inf, 3.0], "b": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="infinite"):
        CompositeScore().fit(X)
