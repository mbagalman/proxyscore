"""Validation for right-censored time-to-event outcomes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

import numpy as np
import pandas as pd

from ._utils import aligned_series, as_series, check_unique_index, ensure_count, validate_score

RiskDirection = Literal["higher", "lower"]


class _SurvivalMetrics(Protocol):
    def concordance_index_ipcw(
        self,
        survival_train: np.ndarray,
        survival_test: np.ndarray,
        estimate: np.ndarray,
        tau: float,
    ) -> tuple[float, int, int, int, int]: ...

    def cumulative_dynamic_auc(
        self,
        survival_train: np.ndarray,
        survival_test: np.ndarray,
        estimate: np.ndarray,
        times: np.ndarray,
    ) -> tuple[np.ndarray, float]: ...

    def brier_score(
        self,
        survival_train: np.ndarray,
        survival_test: np.ndarray,
        estimate: np.ndarray,
        times: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class SurvivalValidationAssessment:
    """IPCW ranking and optional probability evaluation at declared horizons."""

    input_rows: int
    complete_rows: int
    dropped_rows: int
    event_count: int
    censored_count: int
    reference_rows: int
    reference_event_count: int
    risk_direction: RiskDirection
    horizons: tuple[float, ...]
    ranking_summary: pd.DataFrame
    ranking_by_horizon: pd.DataFrame
    calibration_by_horizon: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def calibration_assessed(self) -> bool:
        """Whether horizon-specific survival probabilities were evaluated."""
        return not self.calibration_by_horizon.empty

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return report-ready ranking and calibration tables."""
        return {
            "ranking_summary": self.ranking_summary,
            "ranking_by_horizon": self.ranking_by_horizon,
            "calibration_by_horizon": self.calibration_by_horizon,
        }

    def to_markdown(self) -> str:
        """Render ranking separately from probability calibration."""
        lines = [
            "# Survival validation",
            "",
            f"**Evaluation sample:** {self.complete_rows} complete rows "
            f"({self.event_count} events, {self.censored_count} censored)",
            f"**Censoring reference:** {self.reference_rows} rows "
            f"({self.reference_event_count} events)",
            "",
            "## Ranking",
            "",
            _markdown_table(self.ranking_summary),
            "",
            _markdown_table(self.ranking_by_horizon),
            "",
            "## Probability evaluation",
            "",
        ]
        if self.calibration_assessed:
            lines.append(_markdown_table(self.calibration_by_horizon))
        else:
            lines.append(
                "Not assessed. Supply horizon-specific survival probabilities to compute "
                "IPCW Brier scores."
            )
        if self.warnings:
            lines += ["", "## Warnings", ""] + [f"> {value}" for value in self.warnings]
        if self.notes:
            lines += ["", "## Notes", ""] + [f"> {value}" for value in self.notes]
        return "\n".join(lines) + "\n"


def _markdown_table(table: pd.DataFrame) -> str:
    try:
        return table.to_markdown(index=False, floatfmt=".4g")
    except ImportError:
        return "```\n" + table.to_string(index=False) + "\n```"


def _metrics() -> _SurvivalMetrics:
    try:
        import sksurv.metrics as metrics
    except ImportError as exc:
        raise ImportError(
            "Survival validation requires the optional scikit-survival dependency. "
            "Install it with `pip install proxyscore[survival]`."
        ) from exc
    return cast(_SurvivalMetrics, metrics)


def _event_series(values: Any, name: str, index: pd.Index) -> pd.Series:
    series = aligned_series(values, name, index)
    observed = series.dropna()
    if not observed.isin([False, True, 0, 1]).all():
        raise ValueError(f"{name} must contain only boolean or 0/1 censoring indicators")
    return series.astype("boolean")


def _duration_series(values: Any, name: str, index: pd.Index) -> pd.Series:
    series = aligned_series(values, name, index)
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_complex_dtype(series):
        raise TypeError(f"{name} must be real-valued numeric")
    numeric = series.astype(float)
    finite = numeric.dropna()
    if np.isinf(finite).any():
        raise ValueError(f"{name} must not contain infinite values")
    if (finite <= 0).any():
        raise ValueError(f"{name} must contain positive follow-up durations")
    return numeric


def _normalize_horizons(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError("horizons must be a non-empty sequence of positive finite times")
    normalized: list[float] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("horizons must contain only positive finite times")
        normalized.append(float(value))
    if len(set(normalized)) != len(normalized):
        raise ValueError("horizons must not contain duplicates")
    return tuple(sorted(normalized))


def _survival_array(duration: pd.Series, event: pd.Series) -> np.ndarray:
    values = np.empty(len(duration), dtype=[("event", "?"), ("time", "<f8")])
    values["event"] = event.to_numpy(dtype=bool)
    values["time"] = duration.to_numpy(dtype=float)
    return values


def _survival_probability_frame(
    values: Any, index: pd.Index, horizons: tuple[float, ...]
) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        if not values.index.equals(index):
            raise ValueError(
                "survival_probabilities index must match the risk_score index in the same order"
            )
        try:
            columns = [float(value) for value in values.columns]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "survival_probabilities columns must be the declared horizons"
            ) from exc
        frame = values.copy()
        frame.columns = columns
        missing = [value for value in horizons if value not in columns]
        extra = [value for value in columns if value not in horizons]
        if missing or extra:
            raise ValueError(
                "survival_probabilities columns must exactly match horizons; "
                f"missing={missing}, extra={extra}"
            )
        frame = frame.loc[:, list(horizons)]
    else:
        array = np.asarray(values)
        if array.shape != (len(index), len(horizons)):
            raise ValueError(
                "survival_probabilities must have shape "
                f"({len(index)}, {len(horizons)}), got {array.shape}"
            )
        frame = pd.DataFrame(array, index=index, columns=horizons)
    if any(not pd.api.types.is_numeric_dtype(frame[column]) for column in frame.columns):
        raise TypeError("survival_probabilities must be numeric")
    frame = frame.astype(float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("survival_probabilities must be complete and finite")
    if ((frame < 0) | (frame > 1)).any(axis=None):
        raise ValueError("survival_probabilities must lie in [0, 1]")
    if (np.diff(frame.to_numpy(), axis=1) > 1e-12).any():
        raise ValueError("survival probabilities must not increase across later horizons")
    return frame


def assess_survival_validation(
    risk_score: Any,
    duration: Any,
    event_observed: Any,
    *,
    horizons: Sequence[float],
    reference_duration: Any,
    reference_event_observed: Any,
    survival_probabilities: Any | None = None,
    risk_direction: RiskDirection = "higher",
    min_sample_size: int = 100,
    min_events: int = 20,
) -> SurvivalValidationAssessment:
    """Evaluate a risk ranking against a right-censored outcome.

    ``event_observed`` is true when the event occurred and false when the row was
    right-censored. ``reference_*`` supplies the follow-up sample used to estimate
    inverse censoring weights. Risk scores are evaluated with IPCW concordance through
    the latest declared horizon and cumulative/dynamic AUC at each horizon. Optional
    survival probabilities, shaped ``(rows, horizons)``, are evaluated separately with
    the IPCW Brier score.
    """
    ensure_count(min_sample_size, 2, "min_sample_size")
    ensure_count(min_events, 1, "min_events")
    declared_horizons = _normalize_horizons(horizons)
    if risk_direction not in ("higher", "lower"):
        raise ValueError("risk_direction must be 'higher' or 'lower'")

    score = as_series(risk_score, "risk_score")
    check_unique_index(score.index, "risk_score")
    validate_score(score, "risk_score")
    durations = _duration_series(duration, "duration", score.index)
    events = _event_series(event_observed, "event_observed", score.index)
    probabilities = (
        None
        if survival_probabilities is None
        else _survival_probability_frame(survival_probabilities, score.index, declared_horizons)
    )

    complete_mask = score.notna() & durations.notna() & events.notna()
    complete_score = score.loc[complete_mask].astype(float)
    complete_duration = durations.loc[complete_mask]
    complete_event = events.loc[complete_mask]
    if probabilities is not None:
        probabilities = probabilities.loc[complete_mask]

    reference = as_series(reference_duration, "reference_duration")
    check_unique_index(reference.index, "reference_duration")
    reference_durations = _duration_series(
        reference, "reference_duration", reference.index
    )
    reference_events = _event_series(
        reference_event_observed, "reference_event_observed", reference.index
    )
    reference_mask = reference_durations.notna() & reference_events.notna()
    reference_durations = reference_durations.loc[reference_mask]
    reference_events = reference_events.loc[reference_mask]

    if len(complete_score) < min_sample_size:
        raise ValueError(
            f"survival validation requires at least {min_sample_size} complete evaluation rows; "
            f"found {len(complete_score)}"
        )
    event_count = int(complete_event.sum())
    if event_count < min_events:
        raise ValueError(
            f"survival validation requires at least {min_events} observed evaluation events; "
            f"found {event_count}"
        )
    if len(reference_durations) < min_sample_size:
        raise ValueError(
            f"censoring reference requires at least {min_sample_size} complete rows; "
            f"found {len(reference_durations)}"
        )
    reference_event_count = int(reference_events.sum())
    if reference_event_count < min_events:
        raise ValueError(
            f"censoring reference requires at least {min_events} observed events; "
            f"found {reference_event_count}"
        )
    if complete_score.nunique() < 2:
        raise ValueError("risk_score must contain at least two distinct complete values")

    max_supported = min(float(complete_duration.max()), float(reference_durations.max()))
    if declared_horizons[-1] >= max_supported:
        raise ValueError(
            "every horizon must be earlier than the maximum follow-up in both evaluation "
            f"and reference samples ({max_supported:g})"
        )
    min_evaluation = float(complete_duration.min())
    if declared_horizons[0] < min_evaluation:
        raise ValueError(
            "every horizon must be at or after the earliest evaluation follow-up "
            f"({min_evaluation:g})"
        )
    for horizon in declared_horizons:
        cases = int(((complete_duration <= horizon) & complete_event.astype(bool)).sum())
        controls = int((complete_duration > horizon).sum())
        if cases == 0 or controls == 0:
            raise ValueError(
                f"horizon {horizon:g} needs at least one observed event by the horizon and "
                "one row known event-free beyond it"
            )

    evaluation_y = _survival_array(complete_duration, complete_event)
    reference_y = _survival_array(reference_durations, reference_events)
    oriented_score = complete_score.to_numpy()
    if risk_direction == "lower":
        oriented_score = -oriented_score
    times = np.asarray(declared_horizons)
    metrics = _metrics()
    try:
        concordance = metrics.concordance_index_ipcw(
            reference_y, evaluation_y, oriented_score, tau=declared_horizons[-1]
        )
        auc, mean_auc = metrics.cumulative_dynamic_auc(
            reference_y, evaluation_y, oriented_score, times
        )
    except ValueError as exc:
        raise ValueError(f"survival ranking could not be estimated: {exc}") from exc

    ranking_summary = pd.DataFrame(
        [
            {
                "metric": "ipcw_concordance",
                "value": float(concordance[0]),
                "through_horizon": declared_horizons[-1],
                "concordant_pairs": int(concordance[1]),
                "discordant_pairs": int(concordance[2]),
                "tied_risk_pairs": int(concordance[3]),
                "tied_time_pairs": int(concordance[4]),
            },
            {
                "metric": "mean_cumulative_dynamic_auc",
                "value": float(mean_auc),
                "through_horizon": declared_horizons[-1],
                "concordant_pairs": pd.NA,
                "discordant_pairs": pd.NA,
                "tied_risk_pairs": pd.NA,
                "tied_time_pairs": pd.NA,
            },
        ]
    )
    ranking_by_horizon = pd.DataFrame(
        {
            "horizon": times,
            "cumulative_dynamic_auc": np.atleast_1d(auc).astype(float),
            "cases": [
                int(((complete_duration <= value) & complete_event.astype(bool)).sum())
                for value in times
            ],
            "controls": [int((complete_duration > value).sum()) for value in times],
        }
    )

    calibration = pd.DataFrame(
        columns=["horizon", "brier_score", "mean_predicted_event_probability"]
    )
    if probabilities is not None:
        try:
            returned_times, brier = metrics.brier_score(
                reference_y, evaluation_y, probabilities.to_numpy(), times
            )
        except ValueError as exc:
            raise ValueError(
                f"survival probability evaluation could not be estimated: {exc}"
            ) from exc
        calibration = pd.DataFrame(
            {
                "horizon": returned_times.astype(float),
                "brier_score": brier.astype(float),
                "mean_predicted_event_probability": (1 - probabilities.mean(axis=0)).to_numpy(),
            }
        )

    warnings: list[str] = []
    dropped_rows = len(score) - len(complete_score)
    if dropped_rows:
        warnings.append(
            f"Dropped {dropped_rows} evaluation rows with missing risk, duration, or "
            "censoring data."
        )
    if len(reference) - len(reference_durations):
        warnings.append(
            f"Dropped {len(reference) - len(reference_durations)} reference rows with missing "
            "duration or censoring data."
        )
    if event_count / len(complete_score) < 0.1:
        warnings.append("Fewer than 10% of evaluation rows contain an observed event.")
    notes = [
        "Higher oriented risk means an earlier event; lower-directed inputs were sign-reversed.",
        "Ranking uses IPCW concordance and cumulative/dynamic AUC from scikit-survival.",
    ]
    if probabilities is None:
        notes.append(
            "Probability evaluation was not run because no horizon-specific survival "
            "probabilities were supplied."
        )
    else:
        notes.append(
            "Brier score is a proper probability score (lower is better) and reflects both "
            "calibration and discrimination; it is not merged into the ranking results."
        )

    return SurvivalValidationAssessment(
        input_rows=len(score),
        complete_rows=len(complete_score),
        dropped_rows=dropped_rows,
        event_count=event_count,
        censored_count=len(complete_score) - event_count,
        reference_rows=len(reference_durations),
        reference_event_count=reference_event_count,
        risk_direction=risk_direction,
        horizons=declared_horizons,
        ranking_summary=ranking_summary,
        ranking_by_horizon=ranking_by_horizon,
        calibration_by_horizon=calibration,
        warnings=warnings,
        notes=notes,
    )
