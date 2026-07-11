from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from proxyscore import ProxyAudit, align_delayed_outcomes


def observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account_id": ["a", "b", "c"],
            "snapshot_at": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "signal_a": [1.0, 2.0, 3.0],
            "signal_b": [3.0, 2.0, 1.0],
            "score": [10.0, 20.0, 30.0],
        }
    )


def test_aligns_first_delayed_outcome_and_reports_diagnostics():
    events = pd.DataFrame(
        {
            "account_id": ["b", "a", "a"],
            "event_at": pd.to_datetime(["2026-01-08", "2026-01-10", "2026-01-05"]),
            "churned": [1, 1, 1],
        }
    )

    aligned = align_delayed_outcomes(
        observations(),
        events,
        entity="account_id",
        score_time="snapshot_at",
        outcome="churned",
        outcome_time="event_at",
        horizon="10d",
        as_of="2026-01-20",
    )

    assert aligned.data["aligned_outcome"].tolist() == [1, 1, 0]
    assert aligned.data["outcome_status"].tolist() == ["matched", "matched", "unmatched"]
    assert aligned.data.loc[0, "aligned_outcome_time"] == pd.Timestamp("2026-01-05")
    assert aligned.diagnostics.matched_rows == 2
    assert aligned.diagnostics.unmatched_rows == 1
    assert aligned.diagnostics.observations_with_multiple_candidates == 1
    assert aligned.diagnostics.duplicate_candidate_rows == 1
    assert aligned.diagnostics.lag_min == pd.Timedelta("4d")
    assert aligned.diagnostics.lag_median == pd.Timedelta("5d")
    assert aligned.diagnostics.lag_max == pd.Timedelta("6d")


def test_repeated_entities_are_matched_to_each_snapshot_window():
    snapshots = pd.DataFrame(
        {
            "account_id": ["a", "a"],
            "snapshot_at": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "signal": [1.0, 2.0],
        }
    )
    events = pd.DataFrame(
        {
            "account_id": ["a", "a"],
            "event_at": pd.to_datetime(["2026-01-15", "2026-02-10"]),
            "value": [10, 20],
        }
    )

    aligned = align_delayed_outcomes(
        snapshots,
        events,
        entity="account_id",
        score_time="snapshot_at",
        outcome="value",
        outcome_time="event_at",
        horizon="20d",
        as_of="2026-03-01",
    )

    assert aligned.data["aligned_outcome"].tolist() == [10, 20]


def test_window_start_is_exclusive_and_end_is_inclusive():
    snapshots = observations().iloc[[0]]
    events = pd.DataFrame(
        {
            "account_id": ["a", "a"],
            "event_at": pd.to_datetime(["2026-01-01", "2026-01-11"]),
            "value": [1, 2],
        }
    )

    aligned = align_delayed_outcomes(
        snapshots,
        events,
        entity="account_id",
        score_time="snapshot_at",
        outcome="value",
        outcome_time="event_at",
        horizon="10d",
        as_of="2026-01-20",
    )

    assert aligned.data.loc[0, "aligned_outcome"] == 2
    assert aligned.diagnostics.observations_with_multiple_candidates == 0


def test_explicit_outcome_windows_and_observation_window_end():
    snapshots = pd.DataFrame(
        {
            "account_id": ["a"],
            "features_end": pd.to_datetime(["2026-01-05"]),
            "forecast_start": pd.to_datetime(["2026-01-07"]),
            "forecast_end": pd.to_datetime(["2026-01-31"]),
            "signal": [1.0],
        }
    )
    events = pd.DataFrame(
        {
            "account_id": ["a"],
            "event_at": pd.to_datetime(["2026-01-08"]),
            "renewed": [1],
        }
    )

    aligned = align_delayed_outcomes(
        snapshots,
        events,
        entity="account_id",
        observation_window_end="features_end",
        outcome_window_start="forecast_start",
        outcome_window_end="forecast_end",
        outcome="renewed",
        outcome_time="event_at",
        as_of="2026-02-01",
    )

    assert aligned.data.loc[0, "aligned_outcome"] == 1
    assert aligned.data.loc[0, "outcome_lag"] == pd.Timedelta("3d")


def test_rejects_outcome_window_that_overlaps_observation_window():
    snapshots = pd.DataFrame(
        {
            "account_id": ["a"],
            "features_end": pd.to_datetime(["2026-01-05"]),
            "forecast_start": pd.to_datetime(["2026-01-04"]),
            "forecast_end": pd.to_datetime(["2026-01-31"]),
        }
    )
    events = pd.DataFrame({"account_id": [], "event_at": [], "value": []})

    with pytest.raises(ValueError, match="start on or after"):
        align_delayed_outcomes(
            snapshots,
            events,
            entity="account_id",
            observation_window_end="features_end",
            outcome_window_start="forecast_start",
            outcome_window_end="forecast_end",
            outcome="value",
            outcome_time="event_at",
            as_of="2026-02-01",
        )


@pytest.mark.parametrize("match, expected", [("first", 1), ("last", 2)])
def test_multiple_outcome_match_policy(match, expected):
    events = pd.DataFrame(
        {
            "account_id": ["a", "a"],
            "event_at": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "value": [1, 2],
        }
    )
    aligned = align_delayed_outcomes(
        observations().iloc[[0]],
        events,
        entity="account_id",
        score_time="snapshot_at",
        outcome="value",
        outcome_time="event_at",
        horizon="10d",
        as_of="2026-01-20",
        match=match,
    )
    assert aligned.data.loc[0, "aligned_outcome"] == expected


def test_multiple_outcome_error_policy_rejects_ambiguity():
    events = pd.DataFrame(
        {
            "account_id": ["a", "a"],
            "event_at": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "value": [1, 2],
        }
    )
    with pytest.raises(ValueError, match="multiple eligible outcomes"):
        align_delayed_outcomes(
            observations().iloc[[0]],
            events,
            entity="account_id",
            score_time="snapshot_at",
            outcome="value",
            outcome_time="event_at",
            horizon="10d",
            as_of="2026-01-20",
            match="error",
        )


def test_censoring_separates_immature_rows_from_mature_non_events():
    aligned = align_delayed_outcomes(
        observations(),
        pd.DataFrame(columns=["account_id", "event_at", "value"]),
        entity="account_id",
        score_time="snapshot_at",
        outcome="value",
        outcome_time="event_at",
        horizon="10d",
        as_of="2026-01-12",
    )

    assert aligned.data["outcome_status"].tolist() == ["unmatched", "unmatched", "censored"]
    assert aligned.data["aligned_outcome"].iloc[:2].tolist() == [0, 0]
    assert pd.isna(aligned.data["aligned_outcome"].iloc[2])
    assert aligned.diagnostics.unmatched_rows == 2
    assert aligned.diagnostics.censored_rows == 1


def test_invalid_and_future_outcomes_are_excluded_and_counted():
    events = pd.DataFrame(
        {
            "account_id": ["a", None, "b", "c"],
            "event_at": ["2026-01-05", "2026-01-06", None, "2026-02-01"],
            "value": [1, 1, 1, 1],
        }
    )
    aligned = align_delayed_outcomes(
        observations(),
        events,
        entity="account_id",
        score_time="snapshot_at",
        outcome="value",
        outcome_time="event_at",
        horizon="10d",
        as_of="2026-01-20",
    )

    assert aligned.diagnostics.invalid_outcome_rows == 2
    assert aligned.diagnostics.future_outcome_rows == 1
    assert aligned.diagnostics.matched_rows == 1


def test_missing_observation_timestamp_is_rejected():
    snapshots = observations()
    snapshots.loc[0, "snapshot_at"] = pd.NaT
    events = pd.DataFrame(columns=["account_id", "event_at", "value"])
    with pytest.raises(ValueError, match="missing timestamps"):
        align_delayed_outcomes(
            snapshots,
            events,
            entity="account_id",
            score_time="snapshot_at",
            outcome="value",
            outcome_time="event_at",
            horizon="10d",
            as_of="2026-01-20",
        )


def test_timezone_aware_inputs_are_normalized_to_utc():
    snapshots = pd.DataFrame(
        {
            "account_id": ["a"],
            "snapshot_at": [pd.Timestamp("2026-01-01 08:00", tz="America/New_York")],
            "signal": [1.0],
        }
    )
    events = pd.DataFrame(
        {
            "account_id": ["a"],
            "event_at": [pd.Timestamp("2026-01-02 14:00", tz="Europe/London")],
            "value": [1],
        }
    )
    aligned = align_delayed_outcomes(
        snapshots,
        events,
        entity="account_id",
        score_time="snapshot_at",
        outcome="value",
        outcome_time="event_at",
        horizon="2d",
        as_of=pd.Timestamp("2026-01-04", tz="UTC"),
    )

    assert str(aligned.data["aligned_outcome_time"].dt.tz) == "UTC"
    assert aligned.data.loc[0, "outcome_lag"] == pd.Timedelta("25h")


def test_mixed_timezone_style_is_rejected():
    snapshots = observations().iloc[[0]]
    events = pd.DataFrame(
        {
            "account_id": ["a"],
            "event_at": [pd.Timestamp("2026-01-02", tz="UTC")],
            "value": [1],
        }
    )
    with pytest.raises(ValueError, match="same timezone style"):
        align_delayed_outcomes(
            snapshots,
            events,
            entity="account_id",
            score_time="snapshot_at",
            outcome="value",
            outcome_time="event_at",
            horizon="10d",
            as_of="2026-01-20",
        )


def test_unsorted_inputs_and_tied_events_use_source_order():
    snapshots = observations().iloc[[1, 0]].reset_index(drop=True)
    events = pd.DataFrame(
        {
            "account_id": ["a", "a", "b"],
            "event_at": pd.to_datetime(["2026-01-05", "2026-01-05", "2026-01-04"]),
            "value": [7, 8, 9],
        }
    )
    aligned = align_delayed_outcomes(
        snapshots,
        events,
        entity="account_id",
        score_time="snapshot_at",
        outcome="value",
        outcome_time="event_at",
        horizon="10d",
        as_of="2026-01-20",
        match="first",
    )
    assert aligned.data["aligned_outcome"].tolist() == [9, 7]


def test_audit_inputs_have_a_shared_fresh_index_and_run_end_to_end():
    n = 400
    rng = np.random.default_rng(42)
    snapshots = pd.DataFrame(
        {
            "account_id": [f"a{i}" for i in range(n)],
            "snapshot_at": pd.Timestamp("2026-01-01"),
            "signal_a": rng.normal(size=n),
            "signal_b": rng.normal(size=n),
        },
        index=np.arange(1000, 1000 + n),
    )
    snapshots["score"] = snapshots["signal_a"] - snapshots["signal_b"]
    event_mask = snapshots["score"].to_numpy() > 0
    events = pd.DataFrame(
        {
            "account_id": snapshots.loc[event_mask, "account_id"].to_numpy(),
            "event_at": pd.Timestamp("2026-01-15"),
            "churned": 1,
        }
    )
    aligned = align_delayed_outcomes(
        snapshots,
        events,
        entity="account_id",
        score_time="snapshot_at",
        outcome="churned",
        outcome_time="event_at",
        horizon="30d",
        as_of="2026-03-01",
    )
    kwargs = aligned.audit_inputs(["signal_a", "signal_b"], score_column="score")

    assert kwargs["indicators"].index.equals(kwargs["outcome"].index)
    report = ProxyAudit(**kwargs).run()
    assert report["downstream"].metrics["auc_oriented"] > 0.95


def test_audit_inputs_exclude_censored_rows_by_default():
    aligned = align_delayed_outcomes(
        observations(),
        pd.DataFrame(columns=["account_id", "event_at", "value"]),
        entity="account_id",
        score_time="snapshot_at",
        outcome="value",
        outcome_time="event_at",
        horizon="10d",
        as_of="2026-01-12",
    )
    kwargs = aligned.audit_inputs(["signal_a", "signal_b"], score_column="score")
    assert len(kwargs["indicators"]) == 2
    assert kwargs["outcome"].tolist() == [0, 0]


def test_configuration_validation():
    events = pd.DataFrame(columns=["account_id", "event_at", "value"])
    with pytest.raises(ValueError, match="exactly one of score_time"):
        align_delayed_outcomes(
            observations(),
            events,
            entity="account_id",
            outcome="value",
            outcome_time="event_at",
            horizon="10d",
            as_of="2026-01-20",
        )
    with pytest.raises(ValueError, match="exactly one of horizon"):
        align_delayed_outcomes(
            observations(),
            events,
            entity="account_id",
            score_time="snapshot_at",
            outcome="value",
            outcome_time="event_at",
            as_of="2026-01-20",
        )
    with pytest.raises(ValueError, match="greater than zero"):
        align_delayed_outcomes(
            observations(),
            events,
            entity="account_id",
            score_time="snapshot_at",
            outcome="value",
            outcome_time="event_at",
            horizon="0d",
            as_of="2026-01-20",
        )
