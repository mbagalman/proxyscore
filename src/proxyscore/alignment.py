"""Point-in-time alignment of score snapshots with delayed outcomes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal

import pandas as pd

MatchPolicy = Literal["first", "last", "error"]


@dataclass(frozen=True)
class AlignmentDiagnostics:
    """Counts and observed lags from a delayed-outcome alignment."""

    input_observations: int
    input_outcomes: int
    aligned_rows: int
    matched_rows: int
    unmatched_rows: int
    censored_rows: int
    invalid_outcome_rows: int
    future_outcome_rows: int
    observations_with_multiple_candidates: int
    duplicate_candidate_rows: int
    lag_min: pd.Timedelta | None
    lag_median: pd.Timedelta | None
    lag_max: pd.Timedelta | None

    def summary(self) -> pd.Series:
        """Return diagnostics as a labeled Series suitable for display or export."""
        return pd.Series(
            {
                "input_observations": self.input_observations,
                "input_outcomes": self.input_outcomes,
                "aligned_rows": self.aligned_rows,
                "matched_rows": self.matched_rows,
                "unmatched_rows": self.unmatched_rows,
                "censored_rows": self.censored_rows,
                "invalid_outcome_rows": self.invalid_outcome_rows,
                "future_outcome_rows": self.future_outcome_rows,
                "observations_with_multiple_candidates": (
                    self.observations_with_multiple_candidates
                ),
                "duplicate_candidate_rows": self.duplicate_candidate_rows,
                "lag_min": self.lag_min,
                "lag_median": self.lag_median,
                "lag_max": self.lag_max,
            },
            name="alignment",
        )


@dataclass(frozen=True)
class AlignmentResult:
    """Aligned rows plus diagnostics and column names needed by an audit."""

    data: pd.DataFrame
    diagnostics: AlignmentDiagnostics
    outcome_column: str = "aligned_outcome"
    outcome_time_column: str = "aligned_outcome_time"
    lag_column: str = "outcome_lag"
    status_column: str = "outcome_status"

    def audit_inputs(
        self,
        indicator_columns: Sequence[str],
        *,
        score_column: str | None = None,
        segment_column: str | None = None,
        period_column: str | None = None,
        include_censored: bool = False,
    ) -> dict[str, Any]:
        """Build keyword arguments that can be passed directly to ``ProxyAudit``.

        Mature matched and unmatched rows are included by default. Censored rows can
        be retained explicitly; their outcome remains missing and downstream checks
        will omit them pairwise.
        """
        required = list(indicator_columns)
        optional = [score_column, segment_column, period_column]
        requested = required + [column for column in optional if column is not None]
        missing = [column for column in requested if column not in self.data.columns]
        if missing:
            raise KeyError(f"alignment result is missing requested columns: {missing}")
        if not required:
            raise ValueError("indicator_columns must contain at least one column")

        frame = self.data
        if not include_censored:
            frame = frame.loc[frame[self.status_column] != "censored"]
        frame = frame.reset_index(drop=True)

        kwargs: dict[str, Any] = {
            "indicators": frame[required],
            "outcome": frame[self.outcome_column],
        }
        if score_column is not None:
            kwargs["score"] = frame[score_column]
        if segment_column is not None:
            kwargs["segments"] = frame[segment_column]
        if period_column is not None:
            kwargs["period"] = frame[period_column]
        return kwargs


def _datetime_series(values: pd.Series, name: str) -> tuple[pd.Series, bool | None]:
    """Parse datetimes while rejecting a mixture of naive and aware values."""
    aware_states: set[bool] = set()
    for value in values.dropna():
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} contains an invalid timestamp: {value!r}") from exc
        aware_states.add(stamp.tzinfo is not None and stamp.utcoffset() is not None)
    if len(aware_states) > 1:
        raise ValueError(f"{name} mixes timezone-aware and timezone-naive timestamps")
    aware = next(iter(aware_states), None)
    try:
        parsed = pd.to_datetime(values, errors="raise", utc=aware is True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains invalid timestamps") from exc
    return pd.Series(parsed, index=values.index, name=name), aware


def _timestamp(value: Any, name: str) -> tuple[pd.Timestamp, bool]:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid timestamp, got {value!r}") from exc
    aware = stamp.tzinfo is not None and stamp.utcoffset() is not None
    if aware:
        stamp = stamp.tz_convert("UTC")
    return stamp, aware


def _timedelta(value: str | timedelta | pd.Timedelta) -> pd.Timedelta:
    try:
        duration = pd.Timedelta(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"horizon must be a valid duration, got {value!r}") from exc
    if pd.isna(duration) or duration <= pd.Timedelta(0):
        raise ValueError(f"horizon must be greater than zero, got {value!r}")
    return duration


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def align_delayed_outcomes(
    observations: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    entity: str,
    outcome: str,
    outcome_time: str,
    score_time: str | None = None,
    observation_window_end: str | None = None,
    outcome_window_start: str | None = None,
    outcome_window_end: str | None = None,
    horizon: str | timedelta | pd.Timedelta | None = None,
    as_of: Any = None,
    match: MatchPolicy = "first",
    no_outcome_value: Any = 0,
) -> AlignmentResult:
    """Align score snapshots to outcomes observed strictly afterward.

    Each observation defines an eligible outcome interval ``(start, end]``. The
    start defaults to the score or observation-window-end timestamp. The end is
    either an observation column supplied through ``outcome_window_end`` or the
    start plus ``horizon``. Rows whose interval has not closed at ``as_of`` are
    censored and receive a missing outcome. Mature rows with no matching event
    receive ``no_outcome_value`` (zero by default for event-table workflows).

    Timezone-aware inputs may use different zones and are normalized to UTC.
    Timezone-aware and timezone-naive values may not be mixed. If ``as_of`` is
    omitted, the latest valid outcome timestamp is used, falling back to the
    latest outcome-window end when the outcome table has no valid timestamps.

    Parameters ``match='first'`` and ``match='last'`` resolve multiple eligible
    outcomes deterministically by timestamp and source order. ``match='error'``
    rejects any observation with more than one eligible outcome.
    """
    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations must be a pandas DataFrame")
    if not isinstance(outcomes, pd.DataFrame):
        raise TypeError("outcomes must be a pandas DataFrame")
    if (score_time is None) == (observation_window_end is None):
        raise ValueError("provide exactly one of score_time or observation_window_end")
    if (horizon is None) == (outcome_window_end is None):
        raise ValueError("provide exactly one of horizon or outcome_window_end")
    if match not in ("first", "last", "error"):
        raise ValueError("match must be 'first', 'last', or 'error'")

    observation_end = score_time if score_time is not None else observation_window_end
    assert observation_end is not None
    observation_columns = [entity, observation_end]
    if outcome_window_start is not None:
        observation_columns.append(outcome_window_start)
    if outcome_window_end is not None:
        observation_columns.append(outcome_window_end)
    _require_columns(observations, observation_columns, "observations")
    _require_columns(outcomes, [entity, outcome, outcome_time], "outcomes")

    reserved = {
        "aligned_outcome",
        "aligned_outcome_time",
        "outcome_lag",
        "outcome_status",
    }
    conflicts = sorted(reserved.intersection(observations.columns))
    if conflicts:
        raise ValueError(f"observations contain reserved output columns: {conflicts}")

    obs = observations.copy().reset_index(drop=True)
    if obs[entity].isna().any():
        raise ValueError(f"observations.{entity} contains missing entity identifiers")
    obs["_observation_id"] = range(len(obs))

    parsed_end, end_aware = _datetime_series(obs[observation_end], observation_end)
    if parsed_end.isna().any():
        raise ValueError(f"observations.{observation_end} contains missing timestamps")
    obs["_observation_end"] = parsed_end

    if outcome_window_start is None:
        obs["_window_start"] = parsed_end
        start_aware = end_aware
    else:
        parsed_start, start_aware = _datetime_series(
            obs[outcome_window_start], outcome_window_start
        )
        if parsed_start.isna().any():
            raise ValueError(f"observations.{outcome_window_start} contains missing timestamps")
        obs["_window_start"] = parsed_start

    if outcome_window_end is None:
        assert horizon is not None
        obs["_window_end"] = obs["_window_start"] + _timedelta(horizon)
        window_end_aware = start_aware
    else:
        parsed_window_end, window_end_aware = _datetime_series(
            obs[outcome_window_end], outcome_window_end
        )
        if parsed_window_end.isna().any():
            raise ValueError(f"observations.{outcome_window_end} contains missing timestamps")
        obs["_window_end"] = parsed_window_end

    temporal_awareness = {
        state for state in (end_aware, start_aware, window_end_aware) if state is not None
    }
    if len(temporal_awareness) > 1:
        raise ValueError("observation time columns mix timezone-aware and timezone-naive values")
    if (obs["_window_start"] < obs["_observation_end"]).any():
        raise ValueError("outcome windows must start on or after the observation window ends")
    if (obs["_window_end"] <= obs["_window_start"]).any():
        raise ValueError("every outcome window must end after it starts")

    events = outcomes[[entity, outcome, outcome_time]].copy().reset_index(drop=True)
    parsed_outcome_time, outcome_aware = _datetime_series(events[outcome_time], outcome_time)
    invalid_mask = events[entity].isna() | events[outcome].isna() | parsed_outcome_time.isna()
    invalid_outcome_rows = int(invalid_mask.sum())
    events = events.loc[~invalid_mask].copy()
    events["_candidate_time"] = parsed_outcome_time.loc[~invalid_mask]
    events["_candidate_value"] = events[outcome]
    events["_candidate_order"] = events.index

    expected_aware = next(iter(temporal_awareness), None)
    if events.empty:
        outcome_aware = expected_aware
    if (
        expected_aware is not None
        and outcome_aware is not None
        and expected_aware != outcome_aware
    ):
        raise ValueError("observation and outcome timestamps must use the same timezone style")

    if as_of is None:
        if events.empty:
            as_of_stamp = obs["_window_end"].max()
        else:
            as_of_stamp = events["_candidate_time"].max()
        as_of_aware = expected_aware if expected_aware is not None else outcome_aware
    else:
        as_of_stamp, as_of_aware = _timestamp(as_of, "as_of")
    if expected_aware is not None and as_of_aware != expected_aware:
        raise ValueError("as_of must use the same timezone style as observation timestamps")

    future_mask = events["_candidate_time"] > as_of_stamp
    future_outcome_rows = int(future_mask.sum())
    events = events.loc[~future_mask]

    candidates = obs[
        ["_observation_id", entity, "_observation_end", "_window_start", "_window_end"]
    ].merge(
        events[[entity, "_candidate_time", "_candidate_value", "_candidate_order"]],
        how="left",
        on=entity,
        sort=False,
    )
    eligible = (
        candidates["_candidate_time"].notna()
        & (candidates["_candidate_time"] > candidates["_window_start"])
        & (candidates["_candidate_time"] <= candidates["_window_end"])
    )
    eligible_candidates = candidates.loc[eligible].copy()
    candidate_counts = eligible_candidates.groupby("_observation_id").size()
    multiple = candidate_counts[candidate_counts > 1]
    observations_with_multiple = int(len(multiple))
    duplicate_candidate_rows = int((multiple - 1).sum())
    if match == "error" and observations_with_multiple:
        ids = multiple.index[:5].tolist()
        raise ValueError(
            f"{observations_with_multiple} observation(s) have multiple eligible outcomes "
            f"(e.g. observation IDs {ids}); choose match='first' or match='last'"
        )

    eligible_candidates = eligible_candidates.sort_values(
        ["_observation_id", "_candidate_time", "_candidate_order"], kind="stable"
    )
    selected = eligible_candidates.drop_duplicates(
        "_observation_id", keep="last" if match == "last" else "first"
    ).set_index("_observation_id")

    selected_value = obs["_observation_id"].map(selected["_candidate_value"])
    selected_time = obs["_observation_id"].map(selected["_candidate_time"])
    matched = selected_time.notna()
    mature = obs["_window_end"] <= as_of_stamp
    censored = ~matched & ~mature
    unmatched = ~matched & mature

    result = observations.copy().reset_index(drop=True)
    result["aligned_outcome"] = selected_value.where(matched, no_outcome_value)
    result.loc[censored, "aligned_outcome"] = pd.NA
    result["aligned_outcome_time"] = selected_time
    result["outcome_lag"] = selected_time - obs["_observation_end"]
    result["outcome_status"] = "matched"
    result.loc[unmatched, "outcome_status"] = "unmatched"
    result.loc[censored, "outcome_status"] = "censored"

    lags = result.loc[matched, "outcome_lag"]
    diagnostics = AlignmentDiagnostics(
        input_observations=len(observations),
        input_outcomes=len(outcomes),
        aligned_rows=len(result),
        matched_rows=int(matched.sum()),
        unmatched_rows=int(unmatched.sum()),
        censored_rows=int(censored.sum()),
        invalid_outcome_rows=invalid_outcome_rows,
        future_outcome_rows=future_outcome_rows,
        observations_with_multiple_candidates=observations_with_multiple,
        duplicate_candidate_rows=duplicate_candidate_rows,
        lag_min=lags.min() if len(lags) else None,
        lag_median=pd.Timedelta(lags.median()) if len(lags) else None,
        lag_max=lags.max() if len(lags) else None,
    )
    return AlignmentResult(result, diagnostics)
