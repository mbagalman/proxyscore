from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from proxyscore import (
    ArtifactVersionError,
    CompositeScore,
    MonitoringBaseline,
    MonitoringLimits,
    MonitorStatus,
    PCAScore,
    Thresholds,
    create_monitoring_baseline,
    monitor_batch,
)

FIXED_TIME = datetime(2026, 7, 11, 16, 0, tzinfo=timezone.utc)


def baseline_data(n=1000, seed=51):
    rng = np.random.default_rng(seed)
    indicators = pd.DataFrame(
        {
            "usage": rng.normal(size=n),
            "depth": rng.normal(size=n),
            "tickets": rng.normal(size=n),
        }
    )
    constructor = CompositeScore(
        weights={"usage": 1, "depth": 1, "tickets": -0.5},
        scaling="zscore",
    ).fit(indicators)
    score = constructor.transform(indicators)
    probability = 1 / (1 + np.exp(-2 * score.to_numpy()))
    outcome = pd.Series((rng.uniform(size=n) < probability).astype(int))
    return indicators, constructor, score, outcome


def make_baseline(**overrides):
    indicators, constructor, score, outcome = baseline_data()
    kwargs = {
        "score_id": "customer-health",
        "score_version": "v1",
        "score": score,
        "score_constructor": constructor,
        "outcome": outcome,
        "created_at": FIXED_TIME,
        "metadata": {"owner": "analytics"},
    }
    kwargs.update(overrides)
    return create_monitoring_baseline(indicators, **kwargs), indicators, score, outcome


def check(result, name):
    return next(item for item in result.checks if item.name == name)


def test_baseline_persists_required_reference_state():
    baseline, indicators, _, _ = make_baseline()

    assert baseline.format_version == "1.0"
    assert baseline.score_id == "customer-health"
    assert baseline.score_version == "v1"
    assert baseline.created_at == "2026-07-11T16:00:00Z"
    assert baseline.package_version
    assert baseline.indicator_columns == list(indicators.columns)
    assert baseline.baseline_rows == 1000
    assert baseline.score_bin_cuts
    assert set(baseline.indicator_bin_cuts) == set(indicators.columns)
    assert baseline.thresholds["psi_unstable"] == 0.25
    assert baseline.construction_state["type"] == "CompositeScore"
    assert baseline.metadata == {"owner": "analytics"}
    assert baseline.baseline_outcome_performance["outcome_type"] == "binary"


def test_baseline_json_and_file_round_trip_are_stable(tmp_path):
    baseline, _, _, _ = make_baseline()
    document = baseline.to_json()
    restored = MonitoringBaseline.from_json(document)
    path = tmp_path / "baseline.json"
    written = baseline.save(path)
    loaded = MonitoringBaseline.load(path)

    assert restored.to_dict() == baseline.to_dict()
    assert loaded.to_dict() == baseline.to_dict()
    assert written == path.resolve()
    assert json.loads(document)["format_version"] == "1.0"
    assert "NaN" not in document


def test_composite_constructor_round_trip_scores_without_refitting():
    baseline, indicators, _, _ = make_baseline()
    restored = baseline.constructor()
    assert isinstance(restored, CompositeScore)
    expected = restored.transform(indicators)

    result = monitor_batch(
        baseline,
        indicators,
        outcome=baseline_data()[3],
        batch_id="same",
        observed_at=FIXED_TIME,
    )
    assert expected.notna().all()
    assert check(result, "score_drift").metrics["psi"] == pytest.approx(0)


def test_rank_constructor_state_round_trip():
    indicators, _, _, _ = baseline_data(n=100)
    constructor = CompositeScore(scaling="rank").fit(indicators)
    baseline = create_monitoring_baseline(
        indicators,
        score_id="rank-score",
        score_version="1",
        score_constructor=constructor,
    )
    restored = baseline.constructor()
    assert isinstance(restored, CompositeScore)
    pd.testing.assert_series_equal(
        restored.transform(indicators),
        constructor.transform(indicators),
    )


def test_pca_constructor_state_round_trip():
    indicators, _, _, _ = baseline_data(n=200)
    constructor = PCAScore().fit(indicators)
    baseline = create_monitoring_baseline(
        indicators,
        score_id="pca-score",
        score_version="1",
        score_constructor=constructor,
    )
    restored = baseline.constructor()
    assert isinstance(restored, PCAScore)
    pd.testing.assert_series_equal(
        restored.transform(indicators),
        constructor.transform(indicators),
    )


def test_monitoring_same_batch_with_mature_outcomes_is_informational():
    baseline, indicators, score, outcome = make_baseline()
    result = monitor_batch(
        baseline,
        indicators,
        score=score,
        outcome=outcome,
        score_version="v1",
        batch_id="2026-07",
        observed_at=FIXED_TIME,
    )

    assert result.alert_state is MonitorStatus.INFORMATIONAL
    assert result.exit_code == 0
    assert all(item.status is MonitorStatus.INFORMATIONAL for item in result.checks)
    assert result.metrics["score_psi"] == pytest.approx(0)
    assert check(result, "outcome_performance").metrics["drop"] == pytest.approx(0)
    assert set(result.details) == {"indicator_drift", "missingness"}


def test_shifted_batch_fails_score_and_indicator_drift():
    baseline, indicators, _, _ = make_baseline()
    shifted = indicators + 3
    constructor = baseline.constructor()
    assert constructor is not None
    result = monitor_batch(
        baseline,
        shifted,
        score=constructor.transform(shifted),
        outcome=pd.Series([0, 1] * 500),
    )

    assert result.alert_state is MonitorStatus.FAILURE
    assert result.exit_code == 2
    assert check(result, "score_drift").status is MonitorStatus.FAILURE
    assert check(result, "indicator_drift").status is MonitorStatus.FAILURE
    assert result.metrics["max_indicator_psi"] >= 0.25


def test_missingness_increase_is_reported_and_can_fail():
    baseline, indicators, score, _ = make_baseline()
    degraded = indicators.copy()
    degraded.loc[:300, "usage"] = np.nan
    result = monitor_batch(baseline, degraded, score=score)

    assert check(result, "missingness").status is MonitorStatus.FAILURE
    row = result.details["missingness"].set_index("indicator").loc["usage"]
    assert row["batch_missing_rate"] > 0.20
    assert row["status"] == "failure"


@pytest.mark.parametrize(
    ("rows", "expected"),
    [(400, MonitorStatus.WARNING), (50, MonitorStatus.FAILURE)],
)
def test_volume_change_uses_persisted_limits(rows, expected):
    baseline, indicators, score, _ = make_baseline()
    result = monitor_batch(baseline, indicators.iloc[:rows], score=score.iloc[:rows])
    assert check(result, "volume").status is expected


def test_missing_required_column_stops_before_metrics():
    baseline, indicators, _, _ = make_baseline()
    result = monitor_batch(baseline, indicators.drop(columns="usage"))

    assert result.alert_state is MonitorStatus.FAILURE
    assert result.metrics == {"validation_stopped_before_metrics": True}
    assert result.details == {}
    assert len(result.checks) == 1
    assert "usage" in check(result, "schema").metrics["missing_columns"]


def test_extra_column_is_warning_but_metrics_still_run():
    baseline, indicators, score, outcome = make_baseline()
    batch = indicators.assign(extra=1)
    result = monitor_batch(baseline, batch, score=score, outcome=outcome)

    assert check(result, "schema").status is MonitorStatus.WARNING
    assert result.alert_state is MonitorStatus.WARNING
    assert "extra" in check(result, "schema").metrics["extra_columns"]
    assert "score_psi" in result.metrics


def test_nonnumeric_and_infinite_indicators_stop_before_metrics():
    baseline, indicators, _, _ = make_baseline()
    text = indicators.copy()
    text["usage"] = "bad"
    nonnumeric = monitor_batch(baseline, text)
    assert nonnumeric.alert_state is MonitorStatus.FAILURE
    assert "non-numeric" in check(nonnumeric, "schema").summary

    infinite = indicators.copy()
    infinite.loc[0, "usage"] = np.inf
    invalid = monitor_batch(baseline, infinite)
    assert invalid.alert_state is MonitorStatus.FAILURE
    assert "infinite" in check(invalid, "schema").summary


def test_score_version_mismatch_stops_before_metrics():
    baseline, indicators, _, _ = make_baseline()
    result = monitor_batch(baseline, indicators, score_version="v2")
    assert result.alert_state is MonitorStatus.FAILURE
    assert check(result, "schema").metrics["expected_score_version"] == "v1"


def test_external_score_baseline_requires_score_on_each_batch():
    indicators, _, score, _ = baseline_data()
    baseline = create_monitoring_baseline(
        indicators,
        score_id="external",
        score_version="1",
        score=score,
    )
    result = monitor_batch(baseline, indicators)
    assert result.alert_state is MonitorStatus.FAILURE
    assert "no fitted constructor" in check(result, "schema").summary.lower()


def test_missing_outcomes_are_not_assessable_with_documented_exit_code():
    baseline, indicators, score, _ = make_baseline()
    result = monitor_batch(baseline, indicators, score=score)

    assert check(result, "outcome_performance").status is MonitorStatus.NOT_ASSESSABLE
    assert result.alert_state is MonitorStatus.NOT_ASSESSABLE
    assert result.exit_code == 3


def test_immature_or_underpowered_outcomes_are_not_assessable():
    baseline, indicators, score, _ = make_baseline()
    outcome = pd.Series(np.nan, index=indicators.index)
    outcome.iloc[:3] = [0, 0, 1]
    result = monitor_batch(baseline, indicators, score=score, outcome=outcome)
    assert check(result, "outcome_performance").status is MonitorStatus.NOT_ASSESSABLE
    assert "at least 10" in check(result, "outcome_performance").summary


def test_degraded_matured_outcome_performance_fails():
    baseline, indicators, _, _ = make_baseline()
    rng = np.random.default_rng(99)
    random_score = pd.Series(rng.normal(size=len(indicators)))
    random_outcome = pd.Series(rng.integers(0, 2, size=len(indicators)))
    result = monitor_batch(
        baseline,
        indicators,
        score=random_score,
        outcome=random_outcome,
    )
    performance = check(result, "outcome_performance")
    assert performance.status is MonitorStatus.FAILURE
    assert performance.metrics["drop"] > 0.10


def test_thresholds_and_monitoring_limits_round_trip_and_control_results():
    thresholds = Thresholds(psi_stable=0.01, psi_unstable=0.02)
    limits = MonitoringLimits(missing_rate_warning_delta=0.01)
    baseline, indicators, score, _ = make_baseline(
        thresholds=thresholds,
        monitoring_limits=limits,
    )
    assert baseline.thresholds["psi_unstable"] == 0.02
    assert baseline.monitoring_limits["missing_rate_warning_delta"] == 0.01

    changed = indicators.copy()
    changed.loc[:20, "usage"] = np.nan
    result = monitor_batch(baseline, changed, score=score)
    assert check(result, "missingness").status is MonitorStatus.WARNING


def test_result_json_markdown_and_html_are_operator_ready(tmp_path):
    baseline, indicators, score, _ = make_baseline()
    result = monitor_batch(
        baseline,
        indicators,
        score=score,
        batch_id="<script>alert(1)</script>",
        observed_at=FIXED_TIME,
    )
    document = result.to_json()
    markdown = result.to_markdown(max_rows=2)
    html_document = result.to_html(max_rows=2)
    json_path = result.write_json(tmp_path / "run.json")
    html_path = result.write_html(tmp_path / "run.html", max_rows=2)

    parsed = json.loads(document)
    assert parsed["alert_state"] == "not_assessable"
    assert parsed["exit_code"] == 3
    assert "NaN" not in document
    assert "# Proxy score monitoring" in markdown
    assert "Showing first 2" in markdown
    assert "<!doctype html>" in html_document
    assert "<script>alert(1)</script>" not in html_document
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_document
    assert json_path.exists()
    assert html_path.exists()


def test_result_serialization_is_deterministic_for_fixed_inputs():
    baseline, indicators, score, _ = make_baseline()
    first = monitor_batch(
        baseline,
        indicators,
        score=score,
        batch_id="fixed",
        observed_at=FIXED_TIME,
    )
    second = monitor_batch(
        baseline,
        indicators,
        score=score,
        batch_id="fixed",
        observed_at=FIXED_TIME,
    )
    assert first.to_json() == second.to_json()


def test_incompatible_and_malformed_artifacts_fail_clearly():
    baseline, _, _, _ = make_baseline()
    values = baseline.to_dict()
    values["format_version"] = "2.0"
    with pytest.raises(ArtifactVersionError, match="No automatic migration"):
        MonitoringBaseline.from_dict(values)

    malformed = baseline.to_dict()
    malformed["indicator_bin_cuts"] = {}
    with pytest.raises(ArtifactVersionError, match="bin schema"):
        MonitoringBaseline.from_dict(malformed)


def test_unknown_constructor_state_is_rejected():
    baseline, _, _, _ = make_baseline()
    values = baseline.to_dict()
    values["construction_state"]["type"] = "UnknownScore"
    with pytest.raises(ArtifactVersionError, match="unsupported score constructor"):
        MonitoringBaseline.from_dict(values)


def test_baseline_input_validation():
    indicators, constructor, score, _ = baseline_data(n=20)
    with pytest.raises(ValueError, match="provide score"):
        create_monitoring_baseline(
            indicators,
            score_id="x",
            score_version="1",
        )
    with pytest.raises(TypeError, match="column names must be strings"):
        create_monitoring_baseline(
            indicators.rename(columns={"usage": 1}),
            score_id="x",
            score_version="1",
            score=score,
        )
    unfitted = CompositeScore()
    with pytest.raises(ValueError, match="must be fitted"):
        create_monitoring_baseline(
            indicators,
            score_id="x",
            score_version="1",
            score=score,
            score_constructor=unfitted,
        )
    with pytest.raises(TypeError, match="JSON-serializable"):
        create_monitoring_baseline(
            indicators,
            score_id="x",
            score_version="1",
            score=constructor.transform(indicators),
            metadata={"bad": object()},
        )


def test_monitoring_limit_validation():
    with pytest.raises(ValueError, match="volume_failure_low"):
        MonitoringLimits(volume_failure_low=0.8, volume_warning_low=0.5)
    with pytest.raises(ValueError, match="performance_warning_drop"):
        MonitoringLimits(performance_warning_drop=0.2, performance_failure_drop=0.1)


def test_monitor_status_exit_codes_are_stable():
    assert MonitorStatus.INFORMATIONAL.exit_code == 0
    assert MonitorStatus.WARNING.exit_code == 1
    assert MonitorStatus.FAILURE.exit_code == 2
    assert MonitorStatus.NOT_ASSESSABLE.exit_code == 3
