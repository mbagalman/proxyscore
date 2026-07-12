import numpy as np
import pandas as pd
import pytest

from proxyscore import (
    CalibrationModel,
    assess_calibration,
    fit_and_assess_calibration,
    fit_calibrator,
)


def exact_probability_sample(repeats: int = 100) -> tuple[pd.Series, pd.Series]:
    probabilities: list[float] = []
    outcomes: list[int] = []
    for numerator in range(1, 10):
        probability = numerator / 10
        probabilities.extend([probability] * repeats)
        outcomes.extend([1] * numerator * (repeats // 10))
        outcomes.extend([0] * (10 - numerator) * (repeats // 10))
    return pd.Series(probabilities, dtype=float), pd.Series(outcomes, dtype=int)


def test_arbitrary_scores_require_mapping_or_explicit_probability_opt_in():
    with pytest.raises(ValueError, match="arbitrary scores are not probabilities"):
        assess_calibration([10, 20, 30, 40], [0, 0, 1, 1], n_bootstrap=0)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        assess_calibration(
            [10, 20, 30, 40],
            [0, 0, 1, 1],
            assume_probabilities=True,
            n_bootstrap=0,
        )


def test_perfect_calibration_reports_expected_metrics_and_curve_uncertainty():
    probability, outcome = exact_probability_sample()
    result = assess_calibration(
        probability,
        outcome,
        assume_probabilities=True,
        bins=9,
        min_bin_size=30,
        n_bootstrap=100,
        random_state=9,
    )

    assert result.metrics["brier_score"] == pytest.approx(11 / 60)
    assert result.metrics["expected_calibration_error"] == pytest.approx(0.0)
    assert result.metrics["calibration_intercept"] == pytest.approx(0.0, abs=1e-7)
    assert result.metrics["calibration_in_the_large"] == pytest.approx(0.0, abs=1e-7)
    assert result.metrics["calibration_model_intercept"] == pytest.approx(0.0, abs=1e-7)
    assert result.metrics["calibration_slope"] == pytest.approx(1.0, abs=1e-7)
    assert not result.curve["sparse"].any()
    assert (result.curve["observed_rate_ci_lower"] <= result.curve["observed_rate"]).all()
    assert (result.curve["observed_rate_ci_upper"] >= result.curve["observed_rate"]).all()
    assert result.metrics["brier_ci_lower"] <= result.metrics["brier_score"]
    assert result.metrics["brier_ci_upper"] >= result.metrics["brier_score"]


def test_overconfidence_has_worse_ece_and_calibration_slope_below_one():
    probability, outcome = exact_probability_sample(repeats=200)
    overconfident = pd.Series(1 / (1 + np.exp(-2 * np.log(probability / (1 - probability)))))

    calibrated = assess_calibration(
        probability, outcome, assume_probabilities=True, bins=9, n_bootstrap=0
    )
    result = assess_calibration(
        overconfident, outcome, assume_probabilities=True, bins=9, n_bootstrap=0
    )

    assert result.metrics["expected_calibration_error"] > calibrated.metrics[
        "expected_calibration_error"
    ]
    assert result.metrics["calibration_slope"] == pytest.approx(0.5, abs=1e-6)


def test_calibration_in_the_large_fixes_slope_at_one():
    probability, outcome = exact_probability_sample(repeats=200)
    logit = np.log(probability / (1 - probability))
    shifted_overconfident = pd.Series(1 / (1 + np.exp(-(0.5 + 2 * logit))))

    result = assess_calibration(
        shifted_overconfident,
        outcome,
        assume_probabilities=True,
        bins=9,
        n_bootstrap=0,
    )

    calibration_in_the_large = float(result.metrics["calibration_in_the_large"])
    recalibrated = 1 / (1 + np.exp(-(np.log(
        shifted_overconfident / (1 - shifted_overconfident)
    ) + calibration_in_the_large)))
    assert recalibrated.mean() == pytest.approx(outcome.mean())
    assert result.metrics["calibration_intercept"] == pytest.approx(
        calibration_in_the_large
    )
    assert result.metrics["calibration_model_intercept"] == pytest.approx(-0.25, abs=1e-6)
    assert result.metrics["calibration_slope"] == pytest.approx(0.5, abs=1e-6)
    assert abs(
        calibration_in_the_large - float(result.metrics["calibration_model_intercept"])
    ) > 0.05


@pytest.mark.parametrize("method", ["logistic", "isotonic"])
def test_fitted_mapping_round_trips_and_predicts_probabilities(method):
    rng = np.random.default_rng(42)
    score = pd.Series(rng.normal(size=600))
    outcome = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-score))))

    model = fit_calibrator(score, outcome, method=method)
    restored = CalibrationModel.from_json(model.to_json())
    predicted = restored.predict(pd.Series([-100.0, 0.0, 100.0, np.nan]))

    assert restored.to_dict() == model.to_dict()
    assert predicted.iloc[:3].between(0, 1).all()
    assert np.isnan(predicted.iloc[3])


def test_fit_and_assess_uses_disjoint_stratified_holdout_by_default():
    rng = np.random.default_rng(5)
    score = pd.Series(rng.normal(size=400), index=np.arange(1000, 1400))
    outcome = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-score))), index=score.index)

    result = fit_and_assess_calibration(score, outcome, n_bootstrap=0)

    assert result.model is not None
    assert result.metrics["split_method"] == "stratified_random_holdout"
    assert result.metrics["fit_sample_size"] + result.metrics["evaluation_sample_size"] == 400
    assert result.model.fit_sample_size == result.metrics["fit_sample_size"]
    assert "disjoint stratified holdout" in result.notes[-1]


def test_constant_scores_are_supported_with_identifiability_warning():
    outcome = pd.Series([0] * 80 + [1] * 20)
    model = fit_calibrator(pd.Series([7.0] * 100), outcome, method="logistic")
    result = assess_calibration(
        pd.Series([9.0] * 100), outcome, model=model, bins=5, n_bootstrap=0
    )

    assert model.parameters[1] == 0
    assert np.isnan(result.metrics["calibration_model_intercept"])
    assert result.metrics["calibration_slope"] != result.metrics["calibration_slope"]
    assert any("constant" in warning for warning in result.warnings)


def test_severe_imbalance_is_reported_without_crashing():
    probability = pd.Series([0.01] * 199 + [0.8])
    outcome = pd.Series([0] * 199 + [1])
    result = assess_calibration(
        probability,
        outcome,
        assume_probabilities=True,
        bins=10,
        min_bin_size=30,
        n_bootstrap=50,
    )

    assert result.metrics["positive_count"] == 1
    assert len(result.curve) == 2
    assert any("severely imbalanced" in warning for warning in result.warnings)
    assert any("quantile bins" in warning for warning in result.warnings)


def test_tied_probability_bins_and_ece_are_row_order_invariant():
    probability = pd.Series([0.2] * 100)
    outcome = pd.Series([1] * 20 + [0] * 80)
    shuffled = outcome.sample(frac=1, random_state=7).reset_index(drop=True)

    ordered = assess_calibration(
        probability, outcome, assume_probabilities=True, bins=10, n_bootstrap=0
    )
    reordered = assess_calibration(
        probability, shuffled, assume_probabilities=True, bins=10, n_bootstrap=0
    )

    assert len(ordered.curve) == len(reordered.curve) == 1
    assert ordered.metrics["effective_bins"] == 1
    assert ordered.metrics["expected_calibration_error"] == pytest.approx(0.0)
    assert reordered.metrics["expected_calibration_error"] == pytest.approx(0.0)
    assert ordered.curve.iloc[0]["observed_rate"] == pytest.approx(0.2)


def test_sparse_bins_and_sample_sizes_are_explicit():
    probability = pd.Series(np.linspace(0.05, 0.95, 40))
    outcome = pd.Series([0, 1] * 20)
    result = assess_calibration(
        probability,
        outcome,
        assume_probabilities=True,
        bins=10,
        min_bin_size=10,
        n_bootstrap=0,
    )

    assert result.metrics["evaluation_sample_size"] == 40
    assert result.curve["sparse"].all()
    assert result.warnings


def test_invalid_options_and_nonbinary_outcome_are_rejected():
    with pytest.raises(ValueError, match="binary outcome"):
        fit_calibrator([1, 2, 3], [0, 1, 2])
    with pytest.raises(ValueError, match="method"):
        fit_calibrator([1, 2, 3, 4], [0, 0, 1, 1], method="spline")
    with pytest.raises(ValueError, match="evaluation_fraction"):
        fit_and_assess_calibration([1, 2, 3, 4], [0, 0, 1, 1], evaluation_fraction=1)
    with pytest.raises(ValueError, match="not both"):
        model = fit_calibrator([1, 2, 3, 4], [0, 0, 1, 1])
        assess_calibration(
            [1, 2, 3, 4],
            [0, 0, 1, 1],
            model=model,
            assume_probabilities=True,
        )
