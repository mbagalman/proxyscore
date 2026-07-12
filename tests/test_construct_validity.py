from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from proxyscore import ConstructValidityAssessment, assess_construct_validity


def _exact_correlated_sample(correlation: np.ndarray, n: int = 400) -> pd.DataFrame:
    """Create a centered sample whose sample correlation equals the input matrix."""
    rng = np.random.default_rng(31)
    raw = rng.normal(size=(n, correlation.shape[0]))
    centered = raw - raw.mean(axis=0)
    q, _ = np.linalg.qr(centered)
    values = q @ np.linalg.cholesky(correlation).T * np.sqrt(n - 1)
    return pd.DataFrame(values, columns=[f"x{i}" for i in range(correlation.shape[0])])


def test_reference_ave_and_htmt_match_known_correlation_structure():
    correlation = np.full((6, 6), 0.16)
    correlation[:3, :3] = 0.80
    correlation[3:, 3:] = 0.80
    np.fill_diagonal(correlation, 1.0)
    data = _exact_correlated_sample(correlation)

    result = assess_construct_validity(
        data,
        {"trust": ["x0", "x1", "x2"], "value": ["x3", "x4", "x5"]},
        n_bootstrap=80,
        random_state=7,
    )

    assert isinstance(result, ConstructValidityAssessment)
    assert result.ave.set_index("construct").loc["trust", "ave"] == pytest.approx(13 / 15)
    assert result.ave.set_index("construct").loc["value", "ave"] == pytest.approx(13 / 15)
    assert result.htmt.iloc[0]["htmt"] == pytest.approx(0.20)
    assert result.ave["meets_threshold"].all()
    assert result.htmt["below_threshold"].all()
    assert result.polarity["aligned"].all()
    assert (result.ave["valid_bootstrap_samples"] == 80).all()
    assert result.htmt.iloc[0]["valid_bootstrap_samples"] == 80


def test_overlapping_constructs_have_high_htmt_and_are_flagged():
    correlation = np.full((6, 6), 0.72)
    correlation[:3, :3] = 0.80
    correlation[3:, 3:] = 0.80
    np.fill_diagonal(correlation, 1.0)
    data = _exact_correlated_sample(correlation)

    result = assess_construct_validity(
        data,
        {"trust": ["x0", "x1", "x2"], "value": ["x3", "x4", "x5"]},
        n_bootstrap=0,
    )

    assert result.htmt.iloc[0]["htmt"] == pytest.approx(0.90)
    assert not bool(result.htmt.iloc[0]["below_threshold"])
    assert pd.isna(result.htmt.iloc[0]["ci_lower"])
    assert any("No bootstrap" in warning for warning in result.warnings)


def test_negative_within_construct_correlations_gate_favorable_flags():
    signs = np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    correlation = np.full((6, 6), 0.16)
    correlation[:3, :3] = 0.80
    correlation[3:, 3:] = 0.80
    np.fill_diagonal(correlation, 1.0)
    correlation *= np.outer(signs, signs)
    data = _exact_correlated_sample(correlation)

    result = assess_construct_validity(
        data,
        {"trust": ["x0", "x1", "x2"], "value": ["x3", "x4", "x5"]},
        n_bootstrap=0,
    )

    trust_polarity = result.polarity[result.polarity["construct"] == "trust"]
    assert (~trust_polarity["aligned"]).sum() == 2
    assert trust_polarity["correlation"].min() == pytest.approx(-0.80)
    assert result.htmt.iloc[0]["htmt"] == pytest.approx(0.20)
    assert bool(result.htmt.iloc[0]["estimate_below_threshold"])
    assert not bool(result.htmt.iloc[0]["polarity_aligned"])
    assert not bool(result.htmt.iloc[0]["below_threshold"])
    trust_ave = result.ave.set_index("construct").loc["trust"]
    assert bool(trust_ave["estimate_meets_threshold"])
    assert not bool(trust_ave["meets_threshold"])
    assert any("x0/x1" in warning and "withheld" in warning for warning in result.warnings)


def test_within_correlation_floor_is_configurable():
    correlation = np.full((6, 6), 0.16)
    correlation[:3, :3] = 0.10
    correlation[3:, 3:] = 0.80
    np.fill_diagonal(correlation, 1.0)
    data = _exact_correlated_sample(correlation)

    strict = assess_construct_validity(
        data,
        {"trust": ["x0", "x1", "x2"], "value": ["x3", "x4", "x5"]},
        min_within_correlation=0.20,
        n_bootstrap=0,
    )
    permissive = assess_construct_validity(
        data,
        {"trust": ["x0", "x1", "x2"], "value": ["x3", "x4", "x5"]},
        min_within_correlation=0.05,
        n_bootstrap=0,
    )

    assert not strict.ave.set_index("construct").loc["trust", "polarity_aligned"]
    assert permissive.ave.set_index("construct").loc["trust", "polarity_aligned"]


def test_shared_complete_case_sample_and_two_indicator_warning_are_explicit():
    rng = np.random.default_rng(9)
    data = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    data.loc[0, "a"] = np.nan
    data.loc[1, "d"] = np.nan

    result = assess_construct_validity(
        data,
        {"first": ["a", "b"], "second": ["c", "d"]},
        min_sample_size=100,
        n_bootstrap=10,
    )

    assert result.input_rows == 120
    assert result.complete_rows == 118
    assert result.dropped_rows == 2
    assert any("complete-case" in warning for warning in result.warnings)
    assert any("Two-indicator" in warning for warning in result.warnings)


def test_markdown_and_tables_preserve_separate_construct_results():
    rng = np.random.default_rng(2)
    latent = rng.normal(size=(150, 2))
    data = pd.DataFrame(
        {
            "a": latent[:, 0] + rng.normal(scale=0.2, size=150),
            "b": latent[:, 0] + rng.normal(scale=0.2, size=150),
            "c": latent[:, 1] + rng.normal(scale=0.2, size=150),
            "d": latent[:, 1] + rng.normal(scale=0.2, size=150),
        }
    )
    result = assess_construct_validity(
        data, {"first": ["a", "b"], "second": ["c", "d"]}, n_bootstrap=10
    )

    assert set(result.tables()) == {"loadings", "ave", "polarity", "htmt"}
    assert "# Multi-construct validity" in result.to_markdown()
    assert "exploratory one-factor" in result.to_markdown()


@pytest.mark.parametrize(
    ("constructs", "error", "message"),
    [
        ({"one": ["a", "b"]}, ValueError, "at least two"),
        ({"one": ["a"], "two": ["c", "d"]}, ValueError, "at least two indicators"),
        ({"one": ["a", "b"], "two": ["b", "c"]}, ValueError, "assigned to both"),
        ({"one": ["a", "b"], "two": ["c", "missing"]}, KeyError, "missing columns"),
    ],
)
def test_construct_definition_safeguards(constructs, error, message):
    data = pd.DataFrame(np.arange(400).reshape(100, 4), columns=list("abcd"))
    with pytest.raises(error, match=message):
        assess_construct_validity(data, constructs, n_bootstrap=0)


def test_sample_constant_and_configuration_safeguards():
    rng = np.random.default_rng(4)
    data = pd.DataFrame(rng.normal(size=(99, 4)), columns=list("abcd"))
    constructs = {"one": ["a", "b"], "two": ["c", "d"]}

    with pytest.raises(ValueError, match="at least 100 complete rows"):
        assess_construct_validity(data, constructs, n_bootstrap=0)
    with pytest.raises(ValueError, match="must vary"):
        assess_construct_validity(data.assign(a=1), constructs, min_sample_size=90, n_bootstrap=0)
    with pytest.raises(ValueError, match="n_bootstrap"):
        assess_construct_validity(data, constructs, min_sample_size=90, n_bootstrap=-1)
    with pytest.raises(ValueError, match="confidence_level"):
        assess_construct_validity(
            data, constructs, min_sample_size=90, confidence_level=1, n_bootstrap=0
        )
    with pytest.raises(ValueError, match="min_within_correlation"):
        assess_construct_validity(
            data,
            constructs,
            min_sample_size=90,
            min_within_correlation=-1.1,
            n_bootstrap=0,
        )
