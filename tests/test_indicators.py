import numpy as np
import pandas as pd

from proxyscore import (
    Status,
    check_indicators,
    cronbach_alpha,
    indicator_summary,
    redundant_pairs,
    vif,
)


def make_reflective(n=400, seed=1):
    rng = np.random.default_rng(seed)
    latent = rng.normal(0, 1, n)
    return pd.DataFrame(
        {f"x{i}": latent + rng.normal(0, 0.6, n) for i in range(4)}
    ), latent


def test_cronbach_alpha_high_for_reflective():
    X, _ = make_reflective()
    assert cronbach_alpha(X) > 0.7


def test_cronbach_alpha_low_for_unrelated():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(0, 1, (400, 4)), columns=list("abcd"))
    assert cronbach_alpha(X) < 0.3


def test_indicator_summary_columns():
    X, _ = make_reflective()
    s = indicator_summary(X)
    assert {"missing_rate", "n_unique", "std", "item_rest_corr"} <= set(s.columns)
    assert (s["item_rest_corr"] > 0.5).all()


def test_vif_flags_collinear():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 300)
    X = pd.DataFrame({"a": a, "dup": a + rng.normal(0, 0.01, 300), "c": rng.normal(0, 1, 300)})
    v = vif(X)
    assert v["a"] > 10
    assert v["c"] < 5


def test_redundant_pairs_found():
    rng = np.random.default_rng(4)
    a = rng.normal(0, 1, 300)
    X = pd.DataFrame({"a": a, "dup": a * 1.01 + 0.001, "c": rng.normal(0, 1, 300)})
    pairs = redundant_pairs(X, 0.9)
    assert len(pairs) == 1
    assert {pairs.iloc[0]["indicator_a"], pairs.iloc[0]["indicator_b"]} == {"a", "dup"}


def test_check_indicators_passes_clean_data():
    X, _ = make_reflective()
    res = check_indicators(X)
    assert res.status is Status.PASS


def test_check_indicators_fails_zero_variance():
    X, _ = make_reflective()
    X["constant"] = 1.0
    res = check_indicators(X)
    assert res.status is Status.FAIL
    assert "zero variance" in res.summary


def test_check_indicators_warns_on_dominance():
    X, _ = make_reflective()
    score = X["x0"] * 2 + 0.5  # score is just one indicator
    res = check_indicators(X, score=score)
    assert res.status in (Status.WARN, Status.FAIL)
    assert "single indicator" in res.summary


def test_check_indicators_warns_on_missingness():
    X, _ = make_reflective()
    X.loc[X.index[:200], "x1"] = np.nan
    res = check_indicators(X)
    assert res.status is Status.WARN
    assert "missing" in res.summary
