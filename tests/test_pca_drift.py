from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from proxyscore import PCALoadingDriftAssessment, PCAScore, assess_pca_loading_drift


def _samples(n: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(130)
    latent = rng.normal(size=n)
    baseline = pd.DataFrame(
        {
            "usage": latent + rng.normal(scale=0.2, size=n),
            "depth": latent + rng.normal(scale=0.2, size=n),
            "sentiment": latent + rng.normal(scale=0.2, size=n),
        }
    )
    current_latent = rng.normal(size=n)
    changed = pd.DataFrame(
        {
            "usage": current_latent + rng.normal(scale=0.2, size=n),
            "depth": -current_latent + rng.normal(scale=0.2, size=n),
            "sentiment": rng.normal(size=n),
        }
    )
    return baseline, changed


def test_stable_sample_has_sign_aligned_similar_loadings_and_intervals():
    baseline_data, _ = _samples()
    constructor = PCAScore().fit(baseline_data)
    original_loadings = constructor.loadings_.copy()

    result = assess_pca_loading_drift(
        constructor,
        baseline_data.sample(frac=1, random_state=2),
        n_bootstrap=80,
        random_state=4,
    )

    assert isinstance(result, PCALoadingDriftAssessment)
    assert result.cosine_similarity == pytest.approx(1.0)
    assert result.max_abs_loading_delta == pytest.approx(0.0, abs=1e-12)
    assert result.explained_variance_delta == pytest.approx(0.0, abs=1e-12)
    assert result.valid_bootstrap_samples == 80
    assert result.cosine_ci_lower > 0.99
    assert set(result.tables()) == {"loadings"}
    pd.testing.assert_series_equal(constructor.loadings_, original_loadings)


def test_changed_covariance_structure_has_large_loading_drift():
    baseline_data, changed = _samples()
    constructor = PCAScore().fit(baseline_data)

    result = assess_pca_loading_drift(
        constructor, changed, n_bootstrap=60, random_state=8
    )

    assert result.cosine_similarity < 0.80
    assert result.max_abs_loading_delta > 0.50
    assert result.explained_variance_delta < -0.15
    indexed = result.loadings.set_index("indicator")
    assert indexed.loc["usage", "current_loading"] * indexed.loc[
        "depth", "current_loading"
    ] < 0


def test_global_pca_sign_is_aligned_before_comparison():
    baseline_data, _ = _samples()
    constructor = PCAScore().fit(baseline_data)
    constructor.loadings_ = -constructor.loadings_

    result = assess_pca_loading_drift(
        constructor, baseline_data, n_bootstrap=20, random_state=3
    )

    assert result.cosine_similarity == pytest.approx(1.0)
    assert result.max_abs_loading_delta == pytest.approx(0.0, abs=1e-12)


def test_missing_rows_and_no_bootstrap_are_explicit():
    baseline_data, _ = _samples(n=150)
    constructor = PCAScore().fit(baseline_data)
    current = baseline_data.copy()
    current.loc[:4, "usage"] = np.nan

    result = assess_pca_loading_drift(
        constructor, current, min_sample_size=100, n_bootstrap=0
    )

    assert result.input_rows == 150
    assert result.complete_rows == 145
    assert result.dropped_rows == 5
    assert np.isnan(result.cosine_ci_lower)
    assert any("complete rows" in warning for warning in result.warnings)
    assert any("No bootstrap" in warning for warning in result.warnings)
    assert "never refitted" in result.to_markdown()


def test_input_and_configuration_safeguards():
    baseline_data, _ = _samples(n=30)
    fitted = PCAScore().fit(baseline_data)

    with pytest.raises(ValueError, match="fitted PCAScore"):
        assess_pca_loading_drift(PCAScore(), baseline_data, min_sample_size=20)
    with pytest.raises(KeyError, match="missing baseline columns"):
        assess_pca_loading_drift(
            fitted, baseline_data.drop(columns="usage"), min_sample_size=20
        )
    with pytest.raises(ValueError, match="at least 30 complete rows"):
        assess_pca_loading_drift(
            fitted, baseline_data.iloc[:29], min_sample_size=30
        )
    with pytest.raises(ValueError, match="n_bootstrap"):
        assess_pca_loading_drift(
            fitted, baseline_data, min_sample_size=20, n_bootstrap=-1
        )
    with pytest.raises(ValueError, match="confidence_level"):
        assess_pca_loading_drift(
            fitted, baseline_data, min_sample_size=20, confidence_level=1
        )
