from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from proxyscore import SurvivalValidationAssessment, assess_survival_validation

pytest.importorskip("sksurv")


def _survival_sample(n: int, seed: int) -> tuple[pd.Series, pd.Series, pd.Series, np.ndarray]:
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)
    rate = 0.035 * np.exp(0.9 * risk)
    event_time = rng.exponential(1 / rate)
    censor_time = rng.uniform(15, 70, size=n)
    duration = np.minimum(event_time, censor_time)
    event = event_time <= censor_time
    index = pd.Index([f"row-{seed}-{value}" for value in range(n)])
    return (
        pd.Series(risk, index=index),
        pd.Series(duration, index=index),
        pd.Series(event, index=index),
        rate,
    )


def test_survival_ranking_and_probability_metrics_are_separate():
    score, duration, event, rate = _survival_sample(800, 12)
    _, reference_duration, reference_event, _ = _survival_sample(1200, 13)
    horizons = [10.0, 20.0, 30.0]
    survival = pd.DataFrame(
        np.exp(-rate[:, None] * np.asarray(horizons)),
        index=score.index,
        columns=horizons,
    )

    result = assess_survival_validation(
        score,
        duration,
        event,
        horizons=horizons,
        reference_duration=reference_duration,
        reference_event_observed=reference_event,
        survival_probabilities=survival,
    )

    assert isinstance(result, SurvivalValidationAssessment)
    assert result.calibration_assessed
    assert result.event_count + result.censored_count == result.complete_rows
    assert result.ranking_summary.set_index("metric").loc["ipcw_concordance", "value"] > 0.68
    assert (result.ranking_by_horizon["cumulative_dynamic_auc"] > 0.7).all()
    assert result.calibration_by_horizon["horizon"].tolist() == horizons
    assert result.calibration_by_horizon["brier_score"].between(0, 0.25).all()
    assert set(result.tables()) == {
        "ranking_summary",
        "ranking_by_horizon",
        "calibration_by_horizon",
    }


def test_lower_risk_direction_orients_ranking_without_requiring_probabilities():
    score, duration, event, _ = _survival_sample(600, 20)
    _, reference_duration, reference_event, _ = _survival_sample(900, 21)

    result = assess_survival_validation(
        -score,
        duration,
        event,
        horizons=[10, 25],
        reference_duration=reference_duration,
        reference_event_observed=reference_event,
        risk_direction="lower",
    )

    assert not result.calibration_assessed
    assert result.calibration_by_horizon.empty
    assert result.ranking_summary.iloc[0]["value"] > 0.7
    assert "Not assessed" in result.to_markdown()


def test_missing_evaluation_and_reference_rows_are_dropped_with_warnings():
    score, duration, event, _ = _survival_sample(300, 30)
    _, reference_duration, reference_event, _ = _survival_sample(500, 31)
    score.iloc[0] = np.nan
    duration.iloc[1] = np.nan
    event = event.astype("boolean")
    event.iloc[2] = pd.NA
    reference_duration.iloc[0] = np.nan

    result = assess_survival_validation(
        score,
        duration,
        event,
        horizons=[10, 20],
        reference_duration=reference_duration,
        reference_event_observed=reference_event,
    )

    assert result.dropped_rows == 3
    assert result.reference_rows == 499
    assert any("Dropped 3 evaluation" in warning for warning in result.warnings)
    assert any("Dropped 1 reference" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ("bad_event", "boolean or 0/1"),
        ("bad_duration", "positive follow-up"),
        ("bad_horizon", "earlier than the maximum follow-up"),
        ("increasing_probability", "must not increase"),
    ],
)
def test_invalid_survival_inputs_fail_clearly(change, match):
    score, duration, event, rate = _survival_sample(300, 40)
    _, reference_duration, reference_event, _ = _survival_sample(500, 41)
    horizons = [10.0, 20.0]
    survival = pd.DataFrame(
        np.exp(-rate[:, None] * np.asarray(horizons)),
        index=score.index,
        columns=horizons,
    )
    if change == "bad_event":
        event = event.astype(object)
        event.iloc[0] = "unknown"
    elif change == "bad_duration":
        duration.iloc[0] = 0
    elif change == "bad_horizon":
        horizons = [10.0, float(min(duration.max(), reference_duration.max()))]
        survival = None
    else:
        survival.iloc[0, 1] = survival.iloc[0, 0] + 0.1

    with pytest.raises(ValueError, match=match):
        assess_survival_validation(
            score,
            duration,
            event,
            horizons=horizons,
            reference_duration=reference_duration,
            reference_event_observed=reference_event,
            survival_probabilities=survival,
        )


def test_probability_columns_and_row_alignment_are_strict():
    score, duration, event, rate = _survival_sample(300, 50)
    _, reference_duration, reference_event, _ = _survival_sample(500, 51)
    survival = pd.DataFrame(
        np.exp(-rate[:, None] * np.asarray([10.0, 20.0])),
        index=score.index[::-1],
        columns=[10.0, 20.0],
    )

    with pytest.raises(ValueError, match="index must match"):
        assess_survival_validation(
            score,
            duration,
            event,
            horizons=[10, 20],
            reference_duration=reference_duration,
            reference_event_observed=reference_event,
            survival_probabilities=survival,
        )


def test_sample_and_event_safeguards_precede_estimation():
    score, duration, event, _ = _survival_sample(40, 60)
    _, reference_duration, reference_event, _ = _survival_sample(120, 61)

    with pytest.raises(ValueError, match="at least 100 complete evaluation rows"):
        assess_survival_validation(
            score,
            duration,
            event,
            horizons=[10],
            reference_duration=reference_duration,
            reference_event_observed=reference_event,
        )

    score, duration, event, _ = _survival_sample(200, 62)
    event[:] = False
    with pytest.raises(ValueError, match="at least 20 observed evaluation events"):
        assess_survival_validation(
            score,
            duration,
            event,
            horizons=[10],
            reference_duration=reference_duration,
            reference_event_observed=reference_event,
        )
