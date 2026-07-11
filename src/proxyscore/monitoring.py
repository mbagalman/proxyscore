"""Versioned baseline artifacts and repeatable batch monitoring."""

from __future__ import annotations

import html
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ._utils import (
    aligned_series,
    as_indicator_frame,
    check_outcome_type,
    check_unique_index,
    ensure_count,
    ensure_finite,
    validate_score,
)
from .config import Thresholds
from .construct import CompositeScore, PCAScore
from .validation import downstream_validity

ARTIFACT_FORMAT_VERSION = "1.0"


class ArtifactVersionError(ValueError):
    """Raised when a monitoring artifact cannot be read by this version."""


class MonitorStatus(str, Enum):
    """Operator state for one monitoring check or an entire run."""

    INFORMATIONAL = "informational"
    WARNING = "warning"
    FAILURE = "failure"
    NOT_ASSESSABLE = "not_assessable"

    @property
    def exit_code(self) -> int:
        """Process status suitable for schedulers and command-line wrappers."""
        return {
            MonitorStatus.INFORMATIONAL: 0,
            MonitorStatus.WARNING: 1,
            MonitorStatus.FAILURE: 2,
            MonitorStatus.NOT_ASSESSABLE: 3,
        }[self]


@dataclass(frozen=True)
class MonitoringLimits:
    """Operational thresholds not covered by the statistical audit config."""

    missing_rate_warning_delta: float = 0.05
    volume_warning_low: float = 0.50
    volume_warning_high: float = 2.00
    volume_failure_low: float = 0.10
    volume_failure_high: float = 10.00
    performance_warning_drop: float = 0.05
    performance_failure_drop: float = 0.10

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")
        if not 0 <= self.volume_failure_low <= self.volume_warning_low <= 1:
            raise ValueError(
                "require 0 <= volume_failure_low <= volume_warning_low <= 1"
            )
        if not 1 <= self.volume_warning_high <= self.volume_failure_high:
            raise ValueError(
                "require 1 <= volume_warning_high <= volume_failure_high"
            )
        if self.performance_warning_drop > self.performance_failure_drop:
            raise ValueError(
                "performance_warning_drop cannot exceed performance_failure_drop"
            )


@dataclass(frozen=True)
class MonitoringCheck:
    """One operator-facing monitoring conclusion."""

    name: str
    status: MonitorStatus
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)


def _package_version() -> str:
    try:
        return importlib_metadata.version("proxyscore")
    except importlib_metadata.PackageNotFoundError:
        return "0.1.0"


def _utc_iso(value: datetime | None = None) -> str:
    stamp = value or datetime.now(timezone.utc)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return _utc_iso(value.to_pydatetime() if isinstance(value, pd.Timestamp) else value)
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return value


def _ensure_json_object(value: dict[str, Any], name: str) -> dict[str, Any]:
    converted = cast(dict[str, Any], _json_value(value))
    try:
        json.dumps(converted, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain only JSON-serializable values") from exc
    return converted


def _series_map(values: pd.Series | None) -> dict[str, float | None] | None:
    if values is None:
        return None
    return {str(key): _json_value(value) for key, value in values.items()}


def _restore_series(values: dict[str, Any] | None, columns: list[str]) -> pd.Series | None:
    if values is None:
        return None
    return pd.Series(
        {column: np.nan if values.get(column) is None else values[column] for column in columns},
        dtype=float,
    )


def _constructor_state(constructor: CompositeScore | PCAScore | None) -> dict[str, Any] | None:
    if constructor is None:
        return None
    if isinstance(constructor, CompositeScore):
        if constructor.columns_ is None:
            raise ValueError("score_constructor must be fitted before creating a baseline")
        state: dict[str, Any] = {
            "type": "CompositeScore",
            "weights": constructor.weights,
            "scaling": constructor.scaling,
            "min_coverage": constructor.min_coverage,
            "columns": constructor.columns_,
            "center": _series_map(constructor.center_),
            "scale": _series_map(constructor.scale_),
            "rank_reference": None,
        }
        if constructor.fit_values_ is not None:
            state["rank_reference"] = {
                column: [float(value) for value in constructor.fit_values_[column].dropna()]
                for column in constructor.columns_
            }
        return _ensure_json_object(state, "score constructor state")
    if isinstance(constructor, PCAScore):
        if constructor.columns_ is None or constructor.loadings_ is None:
            raise ValueError("score_constructor must be fitted before creating a baseline")
        return _ensure_json_object(
            {
                "type": "PCAScore",
                "columns": constructor.columns_,
                "mean": _series_map(constructor.mean_),
                "std": _series_map(constructor.std_),
                "loadings": _series_map(constructor.loadings_),
                "explained_variance_ratio": constructor.explained_variance_ratio_,
            },
            "score constructor state",
        )
    raise TypeError("score_constructor must be a fitted CompositeScore or PCAScore")


def _restore_constructor(state: dict[str, Any]) -> CompositeScore | PCAScore:
    constructor_type = state.get("type")
    columns = [str(column) for column in state.get("columns", [])]
    if not columns:
        raise ArtifactVersionError("constructor state has no fitted columns")
    if constructor_type == "CompositeScore":
        min_coverage = state.get("min_coverage")
        if not isinstance(min_coverage, (int, float)):
            raise ArtifactVersionError("CompositeScore state has invalid min_coverage")
        composite = CompositeScore(
            weights=state.get("weights"),
            scaling=str(state.get("scaling")),
            min_coverage=float(min_coverage),
        )
        composite.columns_ = columns
        composite.center_ = _restore_series(state.get("center"), columns)
        composite.scale_ = _restore_series(state.get("scale"), columns)
        rank_reference = state.get("rank_reference")
        if rank_reference is not None:
            composite.fit_values_ = pd.DataFrame(
                {
                    column: pd.Series(rank_reference.get(column, []), dtype=float)
                    for column in columns
                }
            )
        return composite
    if constructor_type == "PCAScore":
        pca = PCAScore()
        pca.columns_ = columns
        pca.mean_ = _restore_series(state.get("mean"), columns)
        pca.std_ = _restore_series(state.get("std"), columns)
        pca.loadings_ = _restore_series(state.get("loadings"), columns)
        ratio = state.get("explained_variance_ratio")
        pca.explained_variance_ratio_ = None if ratio is None else float(ratio)
        return pca
    raise ArtifactVersionError(f"unsupported score constructor type: {constructor_type!r}")


def _bin_spec(values: pd.Series, bins: int) -> tuple[list[float], list[float]]:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        raise ValueError("cannot create monitoring bins from an entirely missing column")
    quantiles = np.unique(np.quantile(clean, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        midpoint = float(quantiles.mean())
        half = float((quantiles.max() - quantiles.min()) / 2)
        epsilon = half if half > 0 else max(abs(midpoint), 1.0) * 1e-9
        cuts = [midpoint - epsilon, midpoint + epsilon]
    else:
        cuts = [float(value) for value in quantiles[1:-1]]
    edges = np.array([-np.inf, *cuts, np.inf])
    counts, _ = np.histogram(clean, bins=edges)
    proportions = (counts / len(clean)).astype(float).tolist()
    return cuts, proportions


def _fixed_psi(
    values: pd.Series,
    cuts: list[float],
    expected_proportions: list[float],
) -> float:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return float("nan")
    edges = np.array([-np.inf, *cuts, np.inf])
    counts, _ = np.histogram(clean, bins=edges)
    actual = np.clip(counts / len(clean), 1e-4, None)
    expected = np.clip(np.asarray(expected_proportions, dtype=float), 1e-4, None)
    if len(actual) != len(expected):
        raise ArtifactVersionError("stored bin proportions do not match stored bin edges")
    return float(np.sum((actual - expected) * np.log(actual / expected)))


@dataclass
class MonitoringBaseline:
    """Versioned, JSON-safe reference state for repeatable monitoring."""

    score_id: str
    score_version: str
    created_at: str
    package_version: str
    indicator_columns: list[str]
    indicator_dtypes: dict[str, str]
    baseline_rows: int
    score_bin_cuts: list[float]
    score_bin_proportions: list[float]
    indicator_bin_cuts: dict[str, list[float]]
    indicator_bin_proportions: dict[str, list[float]]
    baseline_missing_rates: dict[str, float]
    thresholds: dict[str, Any]
    monitoring_limits: dict[str, Any]
    score_summary: dict[str, Any]
    baseline_outcome_performance: dict[str, Any] | None = None
    construction_state: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    format_version: str = ARTIFACT_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe artifact mapping."""
        return cast(dict[str, Any], _json_value(asdict(self)))

    def to_json(self, indent: int = 2) -> str:
        """Serialize the baseline with deterministic key ordering."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    def save(self, path: str | Path) -> Path:
        """Write the artifact as UTF-8 JSON and return its resolved path."""
        destination = Path(path)
        destination.write_text(self.to_json() + "\n", encoding="utf-8")
        return destination.resolve()

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> MonitoringBaseline:
        """Validate and load a supported artifact mapping."""
        version = values.get("format_version")
        if version != ARTIFACT_FORMAT_VERSION:
            raise ArtifactVersionError(
                f"unsupported monitoring artifact format {version!r}; "
                f"this package supports {ARTIFACT_FORMAT_VERSION!r}. No automatic "
                "migration is available. Recreate the baseline with a compatible package."
            )
        try:
            artifact = cls(**values)
        except TypeError as exc:
            raise ArtifactVersionError(f"malformed monitoring artifact: {exc}") from exc
        artifact.validate()
        return artifact

    @classmethod
    def from_json(cls, document: str) -> MonitoringBaseline:
        """Load an artifact from a JSON document."""
        try:
            values = json.loads(document)
        except json.JSONDecodeError as exc:
            raise ArtifactVersionError(f"invalid monitoring artifact JSON: {exc}") from exc
        if not isinstance(values, dict):
            raise ArtifactVersionError("monitoring artifact JSON must contain an object")
        return cls.from_dict(values)

    @classmethod
    def load(cls, path: str | Path) -> MonitoringBaseline:
        """Load an artifact from UTF-8 JSON."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def validate(self) -> None:
        """Reject malformed state before a monitoring run starts."""
        if not self.score_id or not self.score_version:
            raise ArtifactVersionError("artifact score_id and score_version must be non-empty")
        if not self.indicator_columns or len(set(self.indicator_columns)) != len(
            self.indicator_columns
        ):
            raise ArtifactVersionError("artifact indicator columns must be non-empty and unique")
        if self.baseline_rows < 1:
            raise ArtifactVersionError("artifact baseline_rows must be positive")
        if set(self.indicator_columns) != set(self.indicator_bin_cuts):
            raise ArtifactVersionError("artifact indicator bin schema is incomplete")
        if set(self.indicator_columns) != set(self.indicator_bin_proportions):
            raise ArtifactVersionError("artifact indicator bin proportions are incomplete")
        Thresholds(**self.thresholds)
        MonitoringLimits(**self.monitoring_limits)
        if self.construction_state is not None:
            _restore_constructor(self.construction_state)

    def constructor(self) -> CompositeScore | PCAScore | None:
        """Restore the fitted score constructor, when one was persisted."""
        return (
            _restore_constructor(self.construction_state)
            if self.construction_state is not None
            else None
        )


@dataclass
class MonitoringResult:
    """One stable, serializable operator monitoring result."""

    score_id: str
    score_version: str
    batch_id: str
    observed_at: str
    alert_state: MonitorStatus
    checks: list[MonitoringCheck]
    metrics: dict[str, Any] = field(default_factory=dict)
    details: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        """Process status for schedulers and future CLI integration."""
        return self.alert_state.exit_code

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe result including detail table records."""
        return cast(
            dict[str, Any],
            _json_value(
                {
                    "score_id": self.score_id,
                    "score_version": self.score_version,
                    "batch_id": self.batch_id,
                    "observed_at": self.observed_at,
                    "alert_state": self.alert_state,
                    "exit_code": self.exit_code,
                    "checks": [asdict(check) for check in self.checks],
                    "metrics": self.metrics,
                    "details": {
                        name: table.to_dict(orient="records")
                        for name, table in self.details.items()
                    },
                }
            ),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize the run deterministically without non-standard NaN values."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    def write_json(self, path: str | Path) -> Path:
        """Write a UTF-8 JSON run record and return its resolved path."""
        destination = Path(path)
        destination.write_text(self.to_json() + "\n", encoding="utf-8")
        return destination.resolve()

    def to_markdown(self, max_rows: int = 100) -> str:
        """Render an operator-focused Markdown report."""
        ensure_count(max_rows, 1, "max_rows")
        lines = [
            "# Proxy score monitoring",
            "",
            f"**Score:** `{self.score_id}` version `{self.score_version}`  ",
            f"**Batch:** `{self.batch_id}`  ",
            f"**Alert:** `{self.alert_state.value}` (exit code {self.exit_code})",
            "",
            "| Check | Status | Summary |",
            "| --- | --- | --- |",
        ]
        for check in self.checks:
            summary = check.summary.replace("|", "\\|")
            lines.append(f"| {check.name} | {check.status.value} | {summary} |")
        for name, table in self.details.items():
            lines += ["", f"## {name.replace('_', ' ').title()}", ""]
            shown = table.head(max_rows)
            try:
                lines.append(shown.to_markdown(index=False, floatfmt=".4g"))
            except ImportError:
                lines.append("```\n" + shown.to_string(index=False) + "\n```")
            if len(shown) < len(table):
                lines += ["", f"_Showing first {len(shown)} of {len(table)} rows._"]
        lines.append("")
        return "\n".join(lines)

    def to_html(self, max_rows: int = 100) -> str:
        """Render one self-contained, escaped operator HTML report."""
        ensure_count(max_rows, 1, "max_rows")
        status_rows = pd.DataFrame(
            [
                {
                    "check": check.name,
                    "status": check.status.value,
                    "summary": check.summary,
                }
                for check in self.checks
            ]
        )
        tables = [
            status_rows.to_html(index=False, escape=True, border=0),
        ]
        for name, table in self.details.items():
            shown = table.head(max_rows)
            rendered = shown.to_html(index=False, escape=True, border=0)
            note = (
                f"<p>Showing first {len(shown)} of {len(table)} rows.</p>"
                if len(shown) < len(table)
                else ""
            )
            tables.append(f"<h2>{html.escape(name.replace('_', ' ').title())}</h2>{rendered}{note}")
        return (
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<title>Proxy score monitoring</title><style>"
            "body{font-family:system-ui,sans-serif;color:#18212b;max-width:1100px;"
            "margin:32px auto;padding:0 16px;line-height:1.5}table{border-collapse:collapse;"
            "width:100%;margin:12px 0 24px}th,td{border:1px solid #cbd3dc;padding:7px;"
            "text-align:left}th{background:#f3f6f8}h1,h2{letter-spacing:0}"
            "</style></head><body>"
            "<h1>Proxy score monitoring</h1>"
            f"<p><strong>Score:</strong> {html.escape(self.score_id)} "
            f"version {html.escape(self.score_version)}</p>"
            f"<p><strong>Batch:</strong> {html.escape(self.batch_id)}</p>"
            f"<p><strong>Alert:</strong> {html.escape(self.alert_state.value)} "
            f"(exit code {self.exit_code})</p>"
            + "".join(tables)
            + "</body></html>"
        )

    def write_html(self, path: str | Path, max_rows: int = 100) -> Path:
        """Write the HTML operator report and return its resolved path."""
        destination = Path(path)
        destination.write_text(self.to_html(max_rows=max_rows), encoding="utf-8")
        return destination.resolve()


def _overall_status(checks: list[MonitoringCheck]) -> MonitorStatus:
    statuses = {check.status for check in checks}
    if MonitorStatus.FAILURE in statuses:
        return MonitorStatus.FAILURE
    if MonitorStatus.WARNING in statuses:
        return MonitorStatus.WARNING
    if MonitorStatus.NOT_ASSESSABLE in statuses:
        return MonitorStatus.NOT_ASSESSABLE
    return MonitorStatus.INFORMATIONAL


def create_monitoring_baseline(
    indicators: pd.DataFrame,
    *,
    score_id: str,
    score_version: str,
    score: Any = None,
    score_constructor: CompositeScore | PCAScore | None = None,
    outcome: Any = None,
    thresholds: Thresholds | None = None,
    monitoring_limits: MonitoringLimits | None = None,
    bins: int = 10,
    created_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> MonitoringBaseline:
    """Create a reusable baseline without changing fitted constructor state."""
    if not isinstance(score_id, str) or not score_id:
        raise ValueError("score_id must be a non-empty string")
    if not isinstance(score_version, str) or not score_version:
        raise ValueError("score_version must be a non-empty string")
    ensure_count(bins, 2, "bins")
    indicator_frame = as_indicator_frame(indicators)
    check_unique_index(indicator_frame.index, "indicators")
    if len(indicator_frame) < 1:
        raise ValueError("baseline indicators must contain at least one row")
    if any(not isinstance(column, str) for column in indicator_frame.columns):
        raise TypeError("monitoring indicator column names must be strings")
    state = _constructor_state(score_constructor)
    if score is None:
        if score_constructor is None:
            raise ValueError("provide score or a fitted score_constructor")
        score_series = score_constructor.transform(indicator_frame)
    else:
        score_series = aligned_series(score, "score", indicator_frame.index)
        validate_score(score_series)
    score_series = score_series.rename("score")
    if score_series.dropna().empty:
        raise ValueError("baseline score has no finite non-missing values")

    t = thresholds or Thresholds()
    limits = monitoring_limits or MonitoringLimits()
    score_cuts, score_proportions = _bin_spec(score_series, bins)
    indicator_cuts: dict[str, list[float]] = {}
    indicator_proportions: dict[str, list[float]] = {}
    for column in indicator_frame.columns:
        cuts, proportions = _bin_spec(indicator_frame[column], bins)
        indicator_cuts[str(column)] = cuts
        indicator_proportions[str(column)] = proportions

    baseline_performance: dict[str, Any] | None = None
    if outcome is not None:
        outcome_series = aligned_series(outcome, "outcome", indicator_frame.index)
        check_outcome_type(outcome_series)
        ensure_finite(outcome_series, "outcome")
        paired = pd.concat([score_series, outcome_series], axis=1).dropna()
        if len(paired) >= 3 and paired["outcome"].nunique() >= 2:
            baseline_performance = downstream_validity(paired["score"], paired["outcome"])

    score_summary = {
        "n": int(score_series.notna().sum()),
        "missing_rate": float(score_series.isna().mean()),
        "mean": float(score_series.mean()),
        "std": float(score_series.std(ddof=1)),
        "min": float(score_series.min()),
        "max": float(score_series.max()),
    }
    artifact = MonitoringBaseline(
        score_id=score_id,
        score_version=score_version,
        created_at=_utc_iso(created_at),
        package_version=_package_version(),
        indicator_columns=[str(column) for column in indicator_frame.columns],
        indicator_dtypes={
            str(column): str(indicator_frame[column].dtype) for column in indicator_frame
        },
        baseline_rows=len(indicator_frame),
        score_bin_cuts=score_cuts,
        score_bin_proportions=score_proportions,
        indicator_bin_cuts=indicator_cuts,
        indicator_bin_proportions=indicator_proportions,
        baseline_missing_rates={
            str(column): float(indicator_frame[column].isna().mean())
            for column in indicator_frame
        },
        thresholds=_ensure_json_object(asdict(t), "thresholds"),
        monitoring_limits=_ensure_json_object(asdict(limits), "monitoring_limits"),
        score_summary=_ensure_json_object(score_summary, "score_summary"),
        baseline_outcome_performance=(
            _ensure_json_object(baseline_performance, "baseline outcome performance")
            if baseline_performance is not None
            else None
        ),
        construction_state=state,
        metadata=_ensure_json_object(metadata or {}, "metadata"),
    )
    artifact.validate()
    return artifact


def _schema_failure(
    baseline: MonitoringBaseline,
    batch_id: str,
    observed_at: str,
    summary: str,
    metrics: dict[str, Any],
) -> MonitoringResult:
    check = MonitoringCheck("schema", MonitorStatus.FAILURE, summary, metrics)
    return MonitoringResult(
        baseline.score_id,
        baseline.score_version,
        batch_id,
        observed_at,
        MonitorStatus.FAILURE,
        [check],
        metrics={"validation_stopped_before_metrics": True},
    )


def monitor_batch(
    baseline: MonitoringBaseline,
    indicators: pd.DataFrame,
    *,
    score: Any = None,
    outcome: Any = None,
    score_version: str | None = None,
    batch_id: str = "batch",
    observed_at: datetime | None = None,
) -> MonitoringResult:
    """Evaluate a batch against fixed baseline state without refitting."""
    if not isinstance(baseline, MonitoringBaseline):
        raise TypeError("baseline must be a MonitoringBaseline")
    baseline.validate()
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("batch_id must be a non-empty string")
    observed = _utc_iso(observed_at)
    if score_version is not None and score_version != baseline.score_version:
        return _schema_failure(
            baseline,
            batch_id,
            observed,
            f"Score version {score_version!r} does not match baseline version "
            f"{baseline.score_version!r}.",
            {
                "expected_score_version": baseline.score_version,
                "actual_score_version": score_version,
            },
        )
    if not isinstance(indicators, pd.DataFrame):
        return _schema_failure(
            baseline,
            batch_id,
            observed,
            "Indicators must be a pandas DataFrame.",
            {"actual_type": type(indicators).__name__},
        )
    missing = [column for column in baseline.indicator_columns if column not in indicators.columns]
    extras = [column for column in indicators.columns if column not in baseline.indicator_columns]
    if missing:
        return _schema_failure(
            baseline,
            batch_id,
            observed,
            f"Batch is missing required indicator columns: {missing}.",
            {"missing_columns": missing, "extra_columns": extras},
        )
    non_numeric = [
        column
        for column in baseline.indicator_columns
        if not pd.api.types.is_numeric_dtype(indicators[column])
    ]
    if non_numeric:
        return _schema_failure(
            baseline,
            batch_id,
            observed,
            f"Batch has non-numeric required indicators: {non_numeric}.",
            {"non_numeric_columns": non_numeric},
        )
    try:
        frame = as_indicator_frame(indicators[baseline.indicator_columns])
        check_unique_index(frame.index, "indicators")
    except (TypeError, ValueError) as exc:
        return _schema_failure(
            baseline,
            batch_id,
            observed,
            f"Batch indicator validation failed: {exc}",
            {"validation_error": str(exc)},
        )
    if len(frame) == 0:
        return _schema_failure(
            baseline,
            batch_id,
            observed,
            "Batch contains no rows.",
            {"batch_rows": 0},
        )

    try:
        if score is None:
            constructor = baseline.constructor()
            if constructor is None:
                return _schema_failure(
                    baseline,
                    batch_id,
                    observed,
                    "No batch score was supplied and the baseline has no fitted constructor state.",
                    {"construction_state_available": False},
                )
            score_series = constructor.transform(frame)
        else:
            score_series = aligned_series(score, "score", frame.index)
            validate_score(score_series)
    except (TypeError, ValueError, RuntimeError, KeyError) as exc:
        return _schema_failure(
            baseline,
            batch_id,
            observed,
            f"Batch scoring validation failed: {exc}",
            {"validation_error": str(exc)},
        )
    score_series = score_series.rename("score")

    thresholds = Thresholds(**baseline.thresholds)
    limits = MonitoringLimits(**baseline.monitoring_limits)
    checks: list[MonitoringCheck] = []
    details: dict[str, pd.DataFrame] = {}
    schema_status = MonitorStatus.WARNING if extras else MonitorStatus.INFORMATIONAL
    schema_summary = (
        f"Required schema is compatible; ignored extra columns: {extras}."
        if extras
        else "Required schema is compatible."
    )
    checks.append(
        MonitoringCheck(
            "schema",
            schema_status,
            schema_summary,
            {"required_columns": len(baseline.indicator_columns), "extra_columns": extras},
        )
    )

    volume_ratio = len(frame) / baseline.baseline_rows
    if volume_ratio < limits.volume_failure_low or volume_ratio > limits.volume_failure_high:
        volume_status = MonitorStatus.FAILURE
    elif volume_ratio < limits.volume_warning_low or volume_ratio > limits.volume_warning_high:
        volume_status = MonitorStatus.WARNING
    else:
        volume_status = MonitorStatus.INFORMATIONAL
    checks.append(
        MonitoringCheck(
            "volume",
            volume_status,
            f"Batch has {len(frame)} rows versus baseline {baseline.baseline_rows} "
            f"(ratio {volume_ratio:.3g}).",
            {
                "batch_rows": len(frame),
                "baseline_rows": baseline.baseline_rows,
                "volume_ratio": volume_ratio,
            },
        )
    )

    score_psi = _fixed_psi(
        score_series,
        baseline.score_bin_cuts,
        baseline.score_bin_proportions,
    )
    if not math.isfinite(score_psi):
        score_status = MonitorStatus.NOT_ASSESSABLE
        score_summary = "Score drift is not assessable because the batch has no usable scores."
    elif score_psi >= thresholds.psi_unstable:
        score_status = MonitorStatus.FAILURE
        score_summary = f"Score PSI {score_psi:.3g} indicates significant shift."
    elif score_psi >= thresholds.psi_stable:
        score_status = MonitorStatus.WARNING
        score_summary = f"Score PSI {score_psi:.3g} indicates moderate shift."
    else:
        score_status = MonitorStatus.INFORMATIONAL
        score_summary = f"Score PSI {score_psi:.3g} is stable."
    checks.append(
        MonitoringCheck(
            "score_drift",
            score_status,
            score_summary,
            {
                "psi": score_psi,
                "stable_below": thresholds.psi_stable,
                "failure_at": thresholds.psi_unstable,
                "score_missing_rate": float(score_series.isna().mean()),
            },
        )
    )

    indicator_rows = []
    for column in baseline.indicator_columns:
        indicator_psi = _fixed_psi(
            frame[column],
            baseline.indicator_bin_cuts[column],
            baseline.indicator_bin_proportions[column],
        )
        if not math.isfinite(indicator_psi):
            status = MonitorStatus.NOT_ASSESSABLE
        elif indicator_psi >= thresholds.psi_unstable:
            status = MonitorStatus.FAILURE
        elif indicator_psi >= thresholds.psi_stable:
            status = MonitorStatus.WARNING
        else:
            status = MonitorStatus.INFORMATIONAL
        indicator_rows.append({"indicator": column, "psi": indicator_psi, "status": status.value})
    indicator_table = pd.DataFrame(indicator_rows)
    details["indicator_drift"] = indicator_table
    indicator_statuses = {MonitorStatus(value) for value in indicator_table["status"]}
    if MonitorStatus.FAILURE in indicator_statuses:
        indicator_status = MonitorStatus.FAILURE
    elif MonitorStatus.WARNING in indicator_statuses:
        indicator_status = MonitorStatus.WARNING
    elif MonitorStatus.NOT_ASSESSABLE in indicator_statuses:
        indicator_status = MonitorStatus.NOT_ASSESSABLE
    else:
        indicator_status = MonitorStatus.INFORMATIONAL
    checks.append(
        MonitoringCheck(
            "indicator_drift",
            indicator_status,
            f"Indicator drift evaluated for {len(indicator_table)} columns; "
            f"worst PSI {indicator_table['psi'].max():.3g}.",
            {
                "n_indicators": len(indicator_table),
                "max_psi": float(indicator_table["psi"].max()),
            },
        )
    )

    missing_rows = []
    for column in baseline.indicator_columns:
        current = float(frame[column].isna().mean())
        reference = baseline.baseline_missing_rates[column]
        delta = current - reference
        if current > thresholds.max_missing_rate:
            status = MonitorStatus.FAILURE
        elif delta >= limits.missing_rate_warning_delta:
            status = MonitorStatus.WARNING
        else:
            status = MonitorStatus.INFORMATIONAL
        missing_rows.append(
            {
                "indicator": column,
                "baseline_missing_rate": reference,
                "batch_missing_rate": current,
                "delta": delta,
                "status": status.value,
            }
        )
    missing_table = pd.DataFrame(missing_rows)
    details["missingness"] = missing_table
    missing_statuses = {MonitorStatus(value) for value in missing_table["status"]}
    missing_status = (
        MonitorStatus.FAILURE
        if MonitorStatus.FAILURE in missing_statuses
        else (
            MonitorStatus.WARNING
            if MonitorStatus.WARNING in missing_statuses
            else MonitorStatus.INFORMATIONAL
        )
    )
    checks.append(
        MonitoringCheck(
            "missingness",
            missing_status,
            f"Missingness evaluated for {len(missing_table)} indicators; maximum batch rate "
            f"{missing_table['batch_missing_rate'].max():.3g}.",
            {
                "max_batch_missing_rate": float(missing_table["batch_missing_rate"].max()),
                "max_increase": float(missing_table["delta"].max()),
            },
        )
    )

    performance_metrics: dict[str, Any] = {}
    if outcome is None:
        performance_status = MonitorStatus.NOT_ASSESSABLE
        performance_summary = (
            "Delayed outcomes are not yet available; performance was not assessed."
        )
    else:
        try:
            outcome_series = aligned_series(outcome, "outcome", frame.index)
            check_outcome_type(outcome_series)
            ensure_finite(outcome_series, "outcome")
            paired = pd.concat([score_series, outcome_series], axis=1).dropna()
            if len(paired) < 3 or paired["outcome"].nunique() < 2:
                raise ValueError("too few matured outcomes or no outcome variation")
            current_metrics = downstream_validity(paired["score"], paired["outcome"])
            if current_metrics["outcome_type"] == "binary" and min(
                current_metrics["n_pos"], current_metrics["n_neg"]
            ) < thresholds.min_class_count:
                raise ValueError(
                    f"need at least {thresholds.min_class_count} matured rows in each class"
                )
            metric_name = (
                "auc_oriented"
                if current_metrics["outcome_type"] == "binary"
                else "abs_spearman"
            )
            current_value = (
                float(current_metrics["auc_oriented"])
                if current_metrics["outcome_type"] == "binary"
                else abs(float(current_metrics["spearman"]))
            )
            weak = (
                thresholds.min_auc_weak
                if current_metrics["outcome_type"] == "binary"
                else thresholds.min_corr_weak
            )
            strong = (
                thresholds.min_auc_strong
                if current_metrics["outcome_type"] == "binary"
                else thresholds.min_corr_strong
            )
            baseline_value: float | None = None
            baseline_performance = baseline.baseline_outcome_performance
            if (
                baseline_performance is not None
                and baseline_performance.get("outcome_type")
                == current_metrics["outcome_type"]
            ):
                baseline_value = (
                    float(baseline_performance["auc_oriented"])
                    if current_metrics["outcome_type"] == "binary"
                    else abs(float(baseline_performance["spearman"]))
                )
            drop = baseline_value - current_value if baseline_value is not None else None
            if current_value < weak or (
                drop is not None and drop >= limits.performance_failure_drop
            ):
                performance_status = MonitorStatus.FAILURE
            elif current_value < strong or (
                drop is not None and drop >= limits.performance_warning_drop
            ):
                performance_status = MonitorStatus.WARNING
            else:
                performance_status = MonitorStatus.INFORMATIONAL
            performance_metrics = {
                "metric": metric_name,
                "current": current_value,
                "baseline": baseline_value,
                "drop": drop,
                "n": int(current_metrics["n"]),
            }
            performance_summary = (
                f"Matured-outcome {metric_name} is {current_value:.3g} "
                f"on n={current_metrics['n']}"
                + (
                    f" versus baseline {baseline_value:.3g} (drop {drop:.3g})."
                    if baseline_value is not None and drop is not None
                    else "."
                )
            )
        except (TypeError, ValueError) as exc:
            performance_status = MonitorStatus.NOT_ASSESSABLE
            performance_summary = f"Delayed-outcome performance is not assessable: {exc}."
            performance_metrics = {"reason": str(exc)}
    checks.append(
        MonitoringCheck(
            "outcome_performance",
            performance_status,
            performance_summary,
            performance_metrics,
        )
    )

    metrics = {
        "batch_rows": len(frame),
        "baseline_rows": baseline.baseline_rows,
        "volume_ratio": volume_ratio,
        "score_psi": score_psi,
        "max_indicator_psi": float(indicator_table["psi"].max()),
        "max_missing_rate": float(missing_table["batch_missing_rate"].max()),
    }
    return MonitoringResult(
        score_id=baseline.score_id,
        score_version=baseline.score_version,
        batch_id=batch_id,
        observed_at=observed,
        alert_state=_overall_status(checks),
        checks=checks,
        metrics=metrics,
        details=details,
    )
