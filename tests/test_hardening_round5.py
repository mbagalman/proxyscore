"""Regression tests for the fifth-round code review (commit 99a2c9a)."""

import numpy as np
import pandas as pd
import pytest

from proxyscore import (
    PCAScore,
    ProxyAudit,
    Thresholds,
    check_downstream,
    check_leakage,
    psi,
)

# --- finding 1: complex values rejected across all quantitative inputs --------


def test_complex_indicators_rejected():
    X = pd.DataFrame({"a": np.array([1 + 2j, 3 + 0j, 4 + 1j])})
    with pytest.raises(TypeError, match="real-valued"):
        ProxyAudit(indicators=X)


def test_complex_outcome_rejected():
    rng = np.random.default_rng(0)
    score = pd.Series(rng.normal(0, 1, 50))
    y = pd.Series(rng.normal(0, 1, 50) + 1j * rng.normal(0, 1, 50))
    with pytest.raises(TypeError, match="real-valued"):
        check_downstream(score, y)
    X = pd.DataFrame({"a": rng.normal(0, 1, 50)})
    with pytest.raises(TypeError, match="real-valued"):
        check_leakage(X, y)


def test_complex_psi_samples_rejected():
    real = np.arange(100.0)
    cplx = np.arange(100) + 1j * np.arange(100)
    with pytest.raises(TypeError, match="real-valued"):
        psi(cplx, real)
    with pytest.raises(TypeError, match="real-valued"):
        psi(real, cplx)


# --- finding 2: Thresholds rejects malformed custom values --------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_vif": float("nan")},
        {"max_segment_smd": float("nan")},
        {"max_vif": float("inf")},
        {"psi_unstable": float("nan")},
        {"max_segment_auc_gap": 2.0},
        {"max_segment_corr_gap": 3.0},
        {"min_class_count": True},
        {"min_segment_size": 2.5},
    ],
)
def test_thresholds_reject_malformed_numbers(kwargs):
    with pytest.raises(ValueError):
        Thresholds(**kwargs)


@pytest.mark.parametrize(
    "patterns",
    [None, "churn", [1, 2], ["ok", ""], [b"churn"]],
)
def test_thresholds_reject_bad_leak_patterns(patterns):
    with pytest.raises(TypeError):
        Thresholds(leak_name_patterns=patterns)


def test_thresholds_copies_leak_patterns_defensively():
    patterns = ["churn", "renew"]
    t = Thresholds(leak_name_patterns=patterns)
    patterns.append("mutated")
    assert "mutated" not in t.leak_name_patterns


# --- finding 3: PCA refuses to fit zero-variance data --------------------------


def test_pca_all_constant_fit_raises():
    X = pd.DataFrame({"a": [2.0] * 10, "b": [5.0] * 10, "c": [-1.0] * 10})
    with pytest.raises(ValueError, match="varies"):
        PCAScore().fit(X)


def test_pca_partially_constant_fit_works():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"varies": rng.normal(0, 1, 100), "flat": np.ones(100)})
    ps = PCAScore().fit(X)
    # the single varying indicator carries the component
    assert abs(ps.loadings_["varies"]) > 0.99
    assert ps.fit_transform(X).notna().all()
