"""Validation and score comparison across named business outcomes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from ._utils import (
    aligned_series,
    as_indicator_frame,
    as_series,
    check_unique_index,
    ensure_count,
    ensure_finite,
    validate_score,
)
from .alignment import AlignmentResult
from .audit import Verdict
from .comparison import ScoreComparison, compare_scores
from .config import Thresholds
from .leakage import check_leakage
from .results import CheckResult, Status
from .validation import check_downstream

OutcomeType = Literal["binary", "continuous"]
OutcomePolarity = Literal["positive", "negative", "auto"]
OutcomeImportance = Literal["required", "supporting"]


@dataclass(frozen=True)
class OutcomeSpec:
    """Values and validation policy for one named delayed outcome.

    ``mature`` is a boolean mask identifying rows whose outcome window has
    closed. Immature rows are excluded from this outcome only. Use
    :meth:`from_alignment` to retain censoring information from BR-001 output.
    """

    values: Any
    outcome_type: OutcomeType
    polarity: OutcomePolarity = "auto"
    window: str = "unspecified"
    importance: OutcomeImportance = "required"
    mature: Any = None

    @classmethod
    def from_alignment(
        cls,
        alignment: AlignmentResult,
        *,
        outcome_type: OutcomeType,
        polarity: OutcomePolarity = "auto",
        window: str,
        importance: OutcomeImportance = "required",
    ) -> OutcomeSpec:
        """Build a specification from point-in-time aligned outcome rows."""
        if not isinstance(alignment, AlignmentResult):
            raise TypeError("alignment must be an AlignmentResult")
        return cls(
            values=alignment.data[alignment.outcome_column],
            outcome_type=outcome_type,
            polarity=polarity,
            window=window,
            importance=importance,
            mature=alignment.data[alignment.status_column] != "censored",
        )


@dataclass(frozen=True)
class OutcomeValidation:
    """Checks and sample diagnostics for one outcome."""

    name: str
    spec: OutcomeSpec
    downstream: CheckResult
    leakage: CheckResult
    input_rows: int
    mature_rows: int
    immature_rows: int
    observed_rows: int
    missing_rows: int
    evaluation_rows: int
    detected_polarity: int | None
    polarity_contradiction: bool

    def summary_row(self) -> dict[str, Any]:
        """Return one report-ready summary row."""
        return {
            "outcome": self.name,
            "importance": self.spec.importance,
            "outcome_type": self.spec.outcome_type,
            "window": self.spec.window,
            "input_rows": self.input_rows,
            "mature_rows": self.mature_rows,
            "immature_rows": self.immature_rows,
            "observed_rows": self.observed_rows,
            "missing_rows": self.missing_rows,
            "missing_rate": (
                self.missing_rows / self.mature_rows if self.mature_rows else float("nan")
            ),
            "evaluation_rows": self.evaluation_rows,
            "expected_polarity": self.spec.polarity,
            "detected_polarity": self.detected_polarity,
            "polarity_contradiction": self.polarity_contradiction,
            "downstream_status": self.downstream.status.value,
            "leakage_status": self.leakage.status.value,
        }


@dataclass
class MultiOutcomeReport:
    """Named outcome evidence plus a non-averaging overall verdict."""

    verdict: Verdict
    verdict_reason: str
    outcomes: dict[str, OutcomeValidation]

    def __getitem__(self, name: str) -> OutcomeValidation:
        return self.outcomes[name]

    def summary(self) -> pd.DataFrame:
        """Return one row per outcome, preserving each outcome's sample and checks."""
        return pd.DataFrame(result.summary_row() for result in self.outcomes.values())

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return summary and per-outcome check details for report systems."""
        tables = {"outcomes": self.summary()}
        for name, result in self.outcomes.items():
            if result.downstream.details is not None:
                tables[f"{name}.downstream"] = result.downstream.details
            if result.leakage.details is not None:
                tables[f"{name}.leakage"] = result.leakage.details
        return tables

    def to_markdown(self, max_rows: int = 50) -> str:
        """Render a portable report without collapsing outcomes into an average."""
        ensure_count(max_rows, 1, "max_rows")
        lines = [
            "# Multi-outcome validation",
            "",
            f"**Verdict: `{self.verdict.value}`** - {self.verdict_reason}",
        ]
        for name, table in self.tables().items():
            lines += ["", f"## {name.replace('.', ' - ').replace('_', ' ').title()}", ""]
            lines.append(_markdown_table(table, max_rows))
        lines.append("")
        return "\n".join(lines)


@dataclass
class MultiOutcomeComparison:
    """Separate score-version comparisons for named outcome samples."""

    baseline_name: str
    candidate_name: str
    comparisons: dict[str, ScoreComparison]
    outcome_summary: pd.DataFrame

    def __getitem__(self, name: str) -> ScoreComparison:
        return self.comparisons[name]

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return a cross-outcome summary and namespaced comparison tables."""
        tables = {"outcomes": self.outcome_summary}
        for name, comparison in self.comparisons.items():
            for section, table in comparison.tables().items():
                tables[f"{name}.{section}"] = table
        return tables

    def to_markdown(self, max_rows: int = 50) -> str:
        """Render all comparisons with their outcome identity and sample intact."""
        ensure_count(max_rows, 1, "max_rows")
        lines = [
            "# Multi-outcome score comparison",
            "",
            f"**Baseline:** `{self.baseline_name}`  ",
            f"**Candidate:** `{self.candidate_name}`",
        ]
        for name, table in self.tables().items():
            lines += ["", f"## {name.replace('.', ' - ').replace('_', ' ').title()}", ""]
            lines.append(_markdown_table(table, max_rows))
        lines.append("")
        return "\n".join(lines)


def _markdown_table(table: pd.DataFrame, max_rows: int) -> str:
    shown = table.head(max_rows)
    try:
        rendered = shown.to_markdown(index=False, floatfmt=".4g")
    except ImportError:
        rendered = "```\n" + shown.to_string(index=False) + "\n```"
    if len(shown) < len(table):
        rendered += f"\n\n_Showing first {len(shown)} of {len(table)} rows._"
    return rendered


def _validate_spec(name: str, spec: OutcomeSpec) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("outcome names must be non-empty strings")
    if not isinstance(spec, OutcomeSpec):
        raise TypeError(f"outcomes[{name!r}] must be an OutcomeSpec")
    if spec.outcome_type not in ("binary", "continuous"):
        raise ValueError(f"outcomes[{name!r}].outcome_type must be 'binary' or 'continuous'")
    if spec.polarity not in ("positive", "negative", "auto"):
        raise ValueError(
            f"outcomes[{name!r}].polarity must be 'positive', 'negative', or 'auto'"
        )
    if spec.importance not in ("required", "supporting"):
        raise ValueError(
            f"outcomes[{name!r}].importance must be 'required' or 'supporting'"
        )
    if not isinstance(spec.window, str) or not spec.window.strip():
        raise ValueError(f"outcomes[{name!r}].window must be a non-empty string")


def _coerce_outcome(
    name: str,
    spec: OutcomeSpec,
    index: pd.Index,
) -> tuple[pd.Series, pd.Series]:
    _validate_spec(name, spec)
    values = aligned_series(spec.values, name, index)
    if spec.mature is None:
        mature = pd.Series(True, index=index, name=f"{name}_mature")
    else:
        mature = aligned_series(spec.mature, f"{name}_mature", index)
        if mature.isna().any():
            raise ValueError(f"outcomes[{name!r}].mature must not contain missing values")
        if not pd.api.types.is_bool_dtype(mature):
            raise TypeError(f"outcomes[{name!r}].mature must be boolean")

    observed = values.loc[mature].dropna()
    ensure_finite(values, f"outcomes[{name!r}].values")
    distinct = observed.nunique()
    if spec.outcome_type == "binary" and distinct > 2:
        raise ValueError(
            f"outcomes[{name!r}] is declared binary but has {distinct} mature values"
        )
    if spec.outcome_type == "continuous":
        if not pd.api.types.is_numeric_dtype(observed):
            raise TypeError(f"outcomes[{name!r}] is continuous and must be numeric")
        if distinct == 2:
            raise ValueError(
                f"outcomes[{name!r}] is declared continuous but has only two mature values; "
                "declare it binary or provide a genuinely continuous outcome"
            )
    return values, mature


def _unassessable_checks(name: str, observed_rows: int) -> tuple[CheckResult, CheckResult]:
    reason = (
        f"Outcome {name!r} has fewer than two distinct observed values across "
        f"{observed_rows} mature row(s), so it cannot be assessed."
    )
    return (
        CheckResult("downstream", Status.SKIP, reason),
        CheckResult("leakage", Status.SKIP, reason),
    )


def _grade(outcomes: dict[str, OutcomeValidation]) -> tuple[Verdict, str]:
    required = [result for result in outcomes.values() if result.spec.importance == "required"]
    required_failures = [
        result.name
        for result in required
        if result.downstream.status is Status.FAIL
        or result.leakage.status is Status.FAIL
        or result.polarity_contradiction
    ]
    if required_failures:
        return (
            Verdict.NOT_VALIDATED,
            "required outcome(s) failed or contradicted configured polarity: "
            + ", ".join(required_failures)
            + ". Outcomes are never averaged into a passing result.",
        )
    unassessable = [
        result.name for result in required if result.downstream.status is Status.SKIP
    ]
    if unassessable:
        return (
            Verdict.NOT_VALIDATED,
            "required outcome(s) could not be assessed: "
            + ", ".join(unassessable)
            + ". A required outcome needs mature, variable observations.",
        )

    supporting_conflicts = [
        result.name
        for result in outcomes.values()
        if result.spec.importance == "supporting"
        and (
            result.downstream.status is Status.FAIL
            or result.leakage.status is Status.FAIL
            or result.polarity_contradiction
        )
    ]
    required_warnings = [
        result.name
        for result in required
        if result.downstream.status is Status.WARN
        or result.leakage.status in (Status.WARN, Status.SKIP)
    ]
    if supporting_conflicts or required_warnings:
        issues = []
        if required_warnings:
            issues.append("required outcome warnings: " + ", ".join(required_warnings))
        if supporting_conflicts:
            issues.append(
                "contradictory supporting evidence: " + ", ".join(supporting_conflicts)
            )
        return (
            Verdict.DIRECTIONAL,
            "; ".join(issues)
            + ". Each outcome remains visible and supporting evidence is not averaged away.",
        )
    return (
        Verdict.DECISION_GRADE,
        "every required outcome had strong downstream evidence and no leakage or polarity "
        "failure; supporting outcomes introduced no contradictory evidence.",
    )


def validate_outcomes(
    score: Any,
    indicators: pd.DataFrame,
    outcomes: Mapping[str, OutcomeSpec],
    *,
    thresholds: Thresholds | None = None,
) -> MultiOutcomeReport:
    """Validate named outcomes independently on each outcome's mature sample.

    No complete-case intersection is taken across outcomes. The overall verdict
    is a gate over named results, never an average of their metrics.
    """
    frame = as_indicator_frame(indicators)
    check_unique_index(frame.index, "indicators")
    aligned_score = aligned_series(score, "score", frame.index)
    validate_score(aligned_score)
    if not isinstance(outcomes, Mapping) or not outcomes:
        raise ValueError("outcomes must be a non-empty mapping of names to OutcomeSpec values")
    for name, spec in outcomes.items():
        _validate_spec(name, spec)
    has_required = any(spec.importance == "required" for spec in outcomes.values())
    if not has_required:
        raise ValueError("at least one outcome must have importance='required'")

    t = thresholds or Thresholds()
    results: dict[str, OutcomeValidation] = {}
    for name, spec in outcomes.items():
        values, mature = _coerce_outcome(name, spec, frame.index)
        mature_values = values.loc[mature]
        mature_score = aligned_score.loc[mature]
        mature_indicators = frame.loc[mature]
        observed_rows = int(mature_values.notna().sum())
        if mature_values.dropna().nunique() < 2:
            downstream, leakage = _unassessable_checks(name, observed_rows)
        else:
            downstream = check_downstream(mature_score, mature_values, t)
            leakage = check_leakage(mature_indicators, mature_values, t)
        detected = downstream.metrics.get("polarity")
        detected_polarity = int(detected) if detected in (-1, 1) else None
        expected = {"positive": 1, "negative": -1}.get(spec.polarity)
        contradiction = expected is not None and detected_polarity not in (None, expected)
        results[name] = OutcomeValidation(
            name=name,
            spec=spec,
            downstream=downstream,
            leakage=leakage,
            input_rows=len(values),
            mature_rows=int(mature.sum()),
            immature_rows=int((~mature).sum()),
            observed_rows=observed_rows,
            missing_rows=int(mature_values.isna().sum()),
            evaluation_rows=int((mature_values.notna() & mature_score.notna()).sum()),
            detected_polarity=detected_polarity,
            polarity_contradiction=contradiction,
        )
    verdict, reason = _grade(results)
    return MultiOutcomeReport(verdict, reason, results)


def compare_outcomes(
    baseline_score: Any,
    candidate_score: Any,
    outcomes: Mapping[str, OutcomeSpec],
    **comparison_options: Any,
) -> MultiOutcomeComparison:
    """Compare score versions separately for every named mature outcome sample."""
    if not isinstance(outcomes, Mapping) or not outcomes:
        raise ValueError("outcomes must be a non-empty mapping of names to OutcomeSpec values")

    indexed = isinstance(baseline_score, pd.Series) or isinstance(candidate_score, pd.Series)
    if indexed:
        if not isinstance(baseline_score, pd.Series) or not isinstance(candidate_score, pd.Series):
            raise TypeError("baseline_score and candidate_score must both be Series")
        reference_index = baseline_score.index
    else:
        reference_index = as_series(baseline_score, "baseline_score").index

    baseline_name = comparison_options.get("baseline_name", "baseline")
    candidate_name = comparison_options.get("candidate_name", "candidate")
    comparisons: dict[str, ScoreComparison] = {}
    rows: list[dict[str, Any]] = []
    for name, spec in outcomes.items():
        if indexed:
            _validate_spec(name, spec)
            if not isinstance(spec.values, pd.Series):
                raise TypeError(
                    f"outcomes[{name!r}].values must be a Series when scores carry indexes"
                )
            check_unique_index(spec.values.index, f"outcomes[{name!r}].values")
            outcome_index = spec.values.index
        else:
            outcome_index = reference_index
        values, mature = _coerce_outcome(name, spec, outcome_index)
        comparison_values = values.where(mature)
        comparison = compare_scores(
            baseline_score,
            candidate_score,
            comparison_values,
            **comparison_options,
        )
        comparisons[name] = comparison
        performance = comparison.performance.iloc[0]
        rows.append(
            {
                "outcome": name,
                "importance": spec.importance,
                "outcome_type": spec.outcome_type,
                "window": spec.window,
                "mature_rows": int(mature.sum()),
                "immature_rows": int((~mature).sum()),
                "evaluation_rows": comparison.coverage.evaluation_rows,
                "metric": performance["metric"],
                "candidate_minus_baseline": performance["candidate_minus_baseline"],
                "assessment": performance["assessment"],
            }
        )
    return MultiOutcomeComparison(
        baseline_name=str(baseline_name),
        candidate_name=str(candidate_name),
        comparisons=comparisons,
        outcome_summary=pd.DataFrame(rows),
    )
