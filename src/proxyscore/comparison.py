"""Paired comparison of baseline and candidate proxy-score versions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from ._utils import (
    aligned_series,
    as_series,
    auc_score,
    check_outcome_type,
    check_unique_index,
    ensure_count,
    ensure_finite,
    is_binary,
    quantile_bins,
    spearman,
    to_binary,
    validate_score,
)
from .actions import analyze_actions
from .bias import segment_summary
from .config import Thresholds
from .stability import psi, psi_over_time
from .validation import downstream_validity, lift_table

Assessment = Literal["improved", "regressed", "inconclusive"]


@dataclass(frozen=True)
class ComparisonCoverage:
    """Entity coverage and exclusions used to form the paired sample."""

    baseline_rows: int
    candidate_rows: int
    overlap_rows: int
    baseline_only_rows: int
    candidate_only_rows: int
    baseline_missing_in_overlap: int
    candidate_missing_in_overlap: int
    outcome_missing_in_overlap: int
    evaluation_rows: int

    def summary(self) -> pd.DataFrame:
        """Return one row per coverage count for reporting."""
        return pd.DataFrame(
            {
                "measure": [
                    "baseline_rows",
                    "candidate_rows",
                    "overlap_rows",
                    "baseline_only_rows",
                    "candidate_only_rows",
                    "baseline_missing_in_overlap",
                    "candidate_missing_in_overlap",
                    "outcome_missing_in_overlap",
                    "evaluation_rows",
                ],
                "value": [
                    self.baseline_rows,
                    self.candidate_rows,
                    self.overlap_rows,
                    self.baseline_only_rows,
                    self.candidate_only_rows,
                    self.baseline_missing_in_overlap,
                    self.candidate_missing_in_overlap,
                    self.outcome_missing_in_overlap,
                    self.evaluation_rows,
                ],
            }
        )


@dataclass
class ScoreComparison:
    """Structured paired evidence for a baseline and candidate score."""

    baseline_name: str
    candidate_name: str
    coverage: ComparisonCoverage
    outcome_type: str
    metrics: dict[str, Any]
    dimensions: pd.DataFrame
    performance: pd.DataFrame
    distributions: pd.DataFrame
    lift: pd.DataFrame
    migration: pd.DataFrame
    rank_movements: pd.DataFrame
    stability: pd.DataFrame | None = None
    segments: pd.DataFrame | None = None
    actions: pd.DataFrame | None = None
    notes: list[str] | None = None

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return all available report-ready tables by stable section name."""
        tables = {
            "coverage": self.coverage.summary(),
            "dimensions": self.dimensions,
            "performance": self.performance,
            "distributions": self.distributions,
            "lift": self.lift,
            "migration": self.migration,
            "rank_movements": self.rank_movements,
        }
        if self.stability is not None:
            tables["stability"] = self.stability
        if self.segments is not None:
            tables["segments"] = self.segments
        if self.actions is not None:
            tables["actions"] = self.actions
        return tables

    def to_markdown(self, max_rows: int = 50) -> str:
        """Render a compact comparison document from the structured tables."""
        ensure_count(max_rows, 1, "max_rows")
        lines = [
            "# Score version comparison",
            "",
            f"**Baseline:** `{self.baseline_name}`  ",
            f"**Candidate:** `{self.candidate_name}`  ",
            f"**Paired evaluation rows:** {self.coverage.evaluation_rows}",
        ]
        for name, table in self.tables().items():
            lines += ["", f"## {name.replace('_', ' ').title()}", ""]
            shown = table.head(max_rows)
            try:
                rendered = shown.to_markdown(index=False, floatfmt=".4g")
            except ImportError:
                rendered = "```\n" + shown.to_string(index=False) + "\n```"
            lines.append(rendered)
            if len(shown) < len(table):
                lines += ["", f"_Showing first {len(shown)} of {len(table)} rows._"]
        if self.notes:
            lines += ["", "## Notes", ""]
            lines.extend(f"> {note}" for note in self.notes)
        lines.append("")
        return "\n".join(lines)


def _comparison_frame(
    baseline_score: Any,
    candidate_score: Any,
    outcome: Any,
    segments: Any,
    period: Any,
) -> tuple[pd.DataFrame, ComparisonCoverage]:
    series_mode = isinstance(baseline_score, pd.Series) or isinstance(candidate_score, pd.Series)
    if series_mode:
        if not isinstance(baseline_score, pd.Series) or not isinstance(candidate_score, pd.Series):
            raise TypeError(
                "baseline_score and candidate_score must both be Series when either carries "
                "an entity index"
            )
        if not isinstance(outcome, pd.Series):
            raise TypeError("outcome must be a Series when scores carry entity indexes")
        for values, name in (
            (baseline_score, "baseline_score"),
            (candidate_score, "candidate_score"),
            (outcome, "outcome"),
        ):
            check_unique_index(values.index, name)
        if segments is not None and not isinstance(segments, pd.Series):
            raise TypeError("segments must be a Series when scores carry entity indexes")
        if period is not None and not isinstance(period, pd.Series):
            raise TypeError("period must be a Series when scores carry entity indexes")
        if isinstance(segments, pd.Series):
            check_unique_index(segments.index, "segments")
        if isinstance(period, pd.Series):
            check_unique_index(period.index, "period")

        baseline = baseline_score.rename("baseline")
        candidate = candidate_score.rename("candidate")
        validate_score(baseline, "baseline_score")
        validate_score(candidate, "candidate_score")
        common = baseline.index.intersection(candidate.index, sort=False)
        frame = pd.DataFrame(index=common)
        frame["baseline"] = baseline.reindex(common)
        frame["candidate"] = candidate.reindex(common)
        frame["outcome"] = outcome.reindex(common)
        if isinstance(segments, pd.Series):
            frame["segment"] = segments.reindex(common)
        if isinstance(period, pd.Series):
            frame["period"] = period.reindex(common)
        baseline_only = len(baseline.index.difference(candidate.index))
        candidate_only = len(candidate.index.difference(baseline.index))
        baseline_rows, candidate_rows = len(baseline), len(candidate)
    else:
        baseline = as_series(baseline_score, "baseline")
        candidate = as_series(candidate_score, "candidate")
        check_unique_index(baseline.index, "baseline_score")
        check_unique_index(candidate.index, "candidate_score")
        validate_score(baseline, "baseline_score")
        validate_score(candidate, "candidate_score")
        if len(candidate) != len(baseline):
            raise ValueError(
                f"candidate_score has length {len(candidate)}, expected {len(baseline)}"
            )
        aligned_outcome = aligned_series(outcome, "outcome", baseline.index)
        frame = pd.concat([baseline, candidate, aligned_outcome], axis=1)
        if segments is not None:
            frame["segment"] = aligned_series(segments, "segment", baseline.index)
        if period is not None:
            frame["period"] = aligned_series(period, "period", baseline.index)
        baseline_only = candidate_only = 0
        baseline_rows = candidate_rows = len(baseline)

    check_outcome_type(frame["outcome"])
    ensure_finite(frame["outcome"], "outcome")
    complete = frame[["baseline", "candidate", "outcome"]].notna().all(axis=1)
    coverage = ComparisonCoverage(
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        overlap_rows=len(frame),
        baseline_only_rows=baseline_only,
        candidate_only_rows=candidate_only,
        baseline_missing_in_overlap=int(frame["baseline"].isna().sum()),
        candidate_missing_in_overlap=int(frame["candidate"].isna().sum()),
        outcome_missing_in_overlap=int(frame["outcome"].isna().sum()),
        evaluation_rows=int(complete.sum()),
    )
    return frame.loc[complete].copy(), coverage


def _performance_value(
    score: pd.Series,
    outcome: pd.Series,
    binary: bool,
    polarity: int,
) -> float:
    oriented = score * polarity
    if binary:
        return auc_score(oriented.to_numpy(), outcome.to_numpy())
    return spearman(oriented, outcome)


def _paired_interval(
    baseline: pd.Series,
    candidate: pd.Series,
    outcome: pd.Series,
    *,
    binary: bool,
    baseline_polarity: int,
    candidate_polarity: int,
    n_bootstrap: int,
    confidence_level: float,
    random_state: int | None,
) -> tuple[float, float, int]:
    if n_bootstrap == 0:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(random_state)
    differences: list[float] = []
    n = len(outcome)
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, n, size=n)
        sampled_outcome = outcome.iloc[sampled].reset_index(drop=True)
        if sampled_outcome.nunique() < 2:
            continue
        baseline_value = _performance_value(
            baseline.iloc[sampled].reset_index(drop=True),
            sampled_outcome,
            binary,
            baseline_polarity,
        )
        candidate_value = _performance_value(
            candidate.iloc[sampled].reset_index(drop=True),
            sampled_outcome,
            binary,
            candidate_polarity,
        )
        difference = candidate_value - baseline_value
        if math.isfinite(difference):
            differences.append(difference)
    if len(differences) < 2:
        return float("nan"), float("nan"), len(differences)
    alpha = (1 - confidence_level) / 2
    lower, upper = np.quantile(differences, [alpha, 1 - alpha])
    return float(lower), float(upper), len(differences)


def _assessment(delta: float, lower: float, upper: float) -> Assessment:
    if math.isfinite(lower) and lower > 0:
        return "improved"
    if math.isfinite(upper) and upper < 0:
        return "regressed"
    return "inconclusive"


def _descriptive_assessment(delta: float, higher_is_better: bool) -> Assessment:
    if abs(delta) <= 1e-12:
        return "inconclusive"
    improved = delta > 0 if higher_is_better else delta < 0
    return "improved" if improved else "regressed"


def _distribution_table(
    frame: pd.DataFrame,
    coverage: ComparisonCoverage,
    baseline_name: str,
    candidate_name: str,
) -> pd.DataFrame:
    rows = []
    missing_rates = {
        baseline_name: (
            coverage.baseline_missing_in_overlap / coverage.overlap_rows
            if coverage.overlap_rows
            else float("nan")
        ),
        candidate_name: (
            coverage.candidate_missing_in_overlap / coverage.overlap_rows
            if coverage.overlap_rows
            else float("nan")
        ),
    }
    for column, version in (("baseline", baseline_name), ("candidate", candidate_name)):
        score = frame[column]
        rows.append(
            {
                "version": version,
                "n": len(score),
                "missing_rate_in_overlap": missing_rates[version],
                "mean": float(score.mean()),
                "std": float(score.std(ddof=1)),
                "min": float(score.min()),
                "p10": float(score.quantile(0.10)),
                "median": float(score.median()),
                "p90": float(score.quantile(0.90)),
                "max": float(score.max()),
            }
        )
    return pd.DataFrame(rows)


def _band_details(
    frame: pd.DataFrame,
    baseline_polarity: int,
    candidate_polarity: int,
    n_bands: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    details = pd.DataFrame(
        {
            "entity": frame.index,
            "baseline_score": frame["baseline"].to_numpy(),
            "candidate_score": frame["candidate"].to_numpy(),
        }
    )
    baseline_oriented = frame["baseline"] * baseline_polarity
    candidate_oriented = frame["candidate"] * candidate_polarity
    details["baseline_rank_percentile"] = baseline_oriented.rank(
        method="average", pct=True
    ).to_numpy()
    details["candidate_rank_percentile"] = candidate_oriented.rank(
        method="average", pct=True
    ).to_numpy()
    details["absolute_rank_change"] = (
        details["candidate_rank_percentile"] - details["baseline_rank_percentile"]
    ).abs()
    details["baseline_band"] = quantile_bins(
        baseline_oriented, n_bands, ascending=False
    ).to_numpy()
    details["candidate_band"] = quantile_bins(
        candidate_oriented, n_bands, ascending=False
    ).to_numpy()
    migration = (
        details.groupby(["baseline_band", "candidate_band"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    migration["rate"] = migration["n"] / len(details)
    details = details.sort_values("absolute_rank_change", ascending=False, kind="stable")
    return details.reset_index(drop=True), migration


def _action_comparison(
    frame: pd.DataFrame,
    baseline_name: str,
    candidate_name: str,
    cutoffs: float | Sequence[float] | None,
    percentiles: float | Sequence[float] | None,
    top_n: int | Sequence[int] | None,
    thresholds: Thresholds,
) -> pd.DataFrame | None:
    if cutoffs is None and percentiles is None and top_n is None:
        return None
    segment = frame["segment"] if "segment" in frame else None
    baseline = analyze_actions(
        frame["baseline"],
        frame["outcome"],
        cutoffs=cutoffs,
        percentiles=percentiles,
        top_n=top_n,
        polarity="auto",
        segments=segment,
        thresholds=thresholds,
    )
    candidate = analyze_actions(
        frame["candidate"],
        frame["outcome"],
        cutoffs=cutoffs,
        percentiles=percentiles,
        top_n=top_n,
        polarity="auto",
        segments=segment,
        thresholds=thresholds,
    )
    rows = []
    for policy_id in baseline.assignments.columns:
        baseline_selected = baseline.assignments[policy_id]
        candidate_selected = candidate.assignments[policy_id]
        both = int((baseline_selected & candidate_selected).sum())
        baseline_only = int((baseline_selected & ~candidate_selected).sum())
        candidate_only = int((~baseline_selected & candidate_selected).sum())
        union = both + baseline_only + candidate_only
        baseline_row = baseline.table.set_index("policy_id").loc[policy_id]
        candidate_row = candidate.table.set_index("policy_id").loc[policy_id]
        row: dict[str, Any] = {
            "policy_id": policy_id,
            "strategy": baseline_row["strategy"],
            "parameter": baseline_row["parameter"],
            f"{baseline_name}_cutoff": baseline_row["score_cutoff"],
            f"{candidate_name}_cutoff": candidate_row["score_cutoff"],
            f"{baseline_name}_selected_n": int(baseline_selected.sum()),
            f"{candidate_name}_selected_n": int(candidate_selected.sum()),
            "selected_by_both": both,
            "baseline_only": baseline_only,
            "candidate_only": candidate_only,
            "changed_n": baseline_only + candidate_only,
            "changed_rate": (baseline_only + candidate_only) / len(frame),
            "jaccard": both / union if union else 1.0,
        }
        for metric in ("precision", "recall", "selected_outcome_mean"):
            if metric in baseline_row.index:
                row[f"{baseline_name}_{metric}"] = baseline_row[metric]
                row[f"{candidate_name}_{metric}"] = candidate_row[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def compare_scores(
    baseline_score: Any,
    candidate_score: Any,
    outcome: Any,
    *,
    segments: Any = None,
    period: Any = None,
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
    n_bands: int = 10,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
    random_state: int | None = 42,
    action_cutoffs: float | Sequence[float] | None = None,
    action_percentiles: float | Sequence[float] | None = None,
    action_top_n: int | Sequence[int] | None = None,
    thresholds: Thresholds | None = None,
) -> ScoreComparison:
    """Compare two score versions on one paired entity/outcome sample.

    Series inputs may have different entity indexes; coverage differences and
    exclusions are reported before complete paired rows are selected. Array-like
    inputs must have equal lengths. Paired bootstrap intervals quantify the
    candidate-minus-baseline downstream-performance delta.
    """
    if not isinstance(baseline_name, str) or not baseline_name:
        raise ValueError("baseline_name must be a non-empty string")
    if not isinstance(candidate_name, str) or not candidate_name:
        raise ValueError("candidate_name must be a non-empty string")
    if baseline_name == candidate_name:
        raise ValueError("baseline_name and candidate_name must differ")
    ensure_count(n_bands, 2, "n_bands")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 0:
        raise ValueError("n_bootstrap must be an integer >= 0")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not math.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError("confidence_level must be a finite number in (0, 1)")

    t = thresholds or Thresholds()
    frame, coverage = _comparison_frame(
        baseline_score,
        candidate_score,
        outcome,
        segments,
        period,
    )
    if len(frame) < 3:
        raise ValueError("need at least 3 complete paired rows to compare scores")
    if frame["baseline"].nunique() < 2 or frame["candidate"].nunique() < 2:
        raise ValueError("both score versions must contain at least two distinct values")
    if frame["outcome"].nunique() < 2:
        raise ValueError("outcome must contain at least two distinct values")

    binary = is_binary(frame["outcome"])
    analyzed_outcome = (
        to_binary(frame["outcome"]) if binary else frame["outcome"].astype(float)
    )
    frame["outcome"] = analyzed_outcome
    baseline_metrics = downstream_validity(frame["baseline"], frame["outcome"])
    candidate_metrics = downstream_validity(frame["candidate"], frame["outcome"])
    baseline_polarity = int(baseline_metrics["polarity"])
    candidate_polarity = int(candidate_metrics["polarity"])
    metric_name = "auc_oriented" if binary else "oriented_spearman"
    baseline_performance = _performance_value(
        frame["baseline"], frame["outcome"], binary, baseline_polarity
    )
    candidate_performance = _performance_value(
        frame["candidate"], frame["outcome"], binary, candidate_polarity
    )
    delta = candidate_performance - baseline_performance
    ci_lower, ci_upper, valid_bootstraps = _paired_interval(
        frame["baseline"],
        frame["candidate"],
        frame["outcome"],
        binary=binary,
        baseline_polarity=baseline_polarity,
        candidate_polarity=candidate_polarity,
        n_bootstrap=n_bootstrap,
        confidence_level=float(confidence_level),
        random_state=random_state,
    )
    downstream_assessment = _assessment(delta, ci_lower, ci_upper)
    method = "paired_bootstrap" if n_bootstrap else "descriptive_only"
    performance = pd.DataFrame(
        [
            {
                "metric": metric_name,
                baseline_name: baseline_performance,
                candidate_name: candidate_performance,
                "candidate_minus_baseline": delta,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "confidence_level": confidence_level,
                "valid_bootstrap_samples": valid_bootstraps,
                "method": method,
                "assessment": downstream_assessment,
            }
        ]
    )

    distributions = _distribution_table(frame, coverage, baseline_name, candidate_name)
    baseline_missing = cast(float, distributions.iloc[0]["missing_rate_in_overlap"])
    candidate_missing = cast(float, distributions.iloc[1]["missing_rate_in_overlap"])
    missing_delta = candidate_missing - baseline_missing
    dimensions_rows: list[dict[str, Any]] = [
        {
            "dimension": "downstream_performance",
            baseline_name: baseline_performance,
            candidate_name: candidate_performance,
            "candidate_minus_baseline": delta,
            "assessment": downstream_assessment,
            "basis": (
                f"paired {confidence_level:.0%} bootstrap interval"
                if n_bootstrap
                else "descriptive only; no bootstrap requested"
            ),
        },
        {
            "dimension": "score_missing_rate",
            baseline_name: baseline_missing,
            candidate_name: candidate_missing,
            "candidate_minus_baseline": missing_delta,
            "assessment": _descriptive_assessment(missing_delta, higher_is_better=False),
            "basis": "descriptive data-quality comparison; lower is better",
        },
    ]

    bands = min(n_bands, len(frame))
    lift_parts = []
    for column, version, polarity in (
        ("baseline", baseline_name, baseline_polarity),
        ("candidate", candidate_name, candidate_polarity),
    ):
        table = lift_table(
            frame[column],
            frame["outcome"],
            n_bands=bands,
            ascending=polarity == -1,
        )
        table.insert(0, "version", version)
        lift_parts.append(table)
    lift = pd.concat(lift_parts, ignore_index=True)

    rank_movements, migration = _band_details(
        frame,
        baseline_polarity,
        candidate_polarity,
        bands,
    )
    raw_spearman = spearman(frame["baseline"], frame["candidate"])
    oriented_spearman = spearman(
        frame["baseline"] * baseline_polarity,
        frame["candidate"] * candidate_polarity,
    )
    raw_pearson = float(cast(float, frame[["baseline", "candidate"]].corr().iloc[0, 1]))

    stability: pd.DataFrame | None = None
    if "period" in frame:
        stability_parts = []
        max_values: dict[str, float] = {}
        for column, version in (("baseline", baseline_name), ("candidate", candidate_name)):
            table = psi_over_time(frame[column], frame["period"])
            table.insert(0, "version", version)
            stability_parts.append(table)
            max_values[version] = float(table["psi"].max()) if len(table) else float("nan")
        stability = pd.concat(stability_parts, ignore_index=True)
        if all(math.isfinite(value) for value in max_values.values()):
            stability_delta = max_values[candidate_name] - max_values[baseline_name]
            dimensions_rows.append(
                {
                    "dimension": "max_period_psi",
                    baseline_name: max_values[baseline_name],
                    candidate_name: max_values[candidate_name],
                    "candidate_minus_baseline": stability_delta,
                    "assessment": _descriptive_assessment(
                        stability_delta, higher_is_better=False
                    ),
                    "basis": "descriptive stability comparison; lower is better",
                }
            )

    segment_table: pd.DataFrame | None = None
    if "segment" in frame:
        segment_parts = []
        min_validity: dict[str, float] = {}
        for column, version in (("baseline", baseline_name), ("candidate", candidate_name)):
            table = segment_summary(frame[column], frame["segment"], frame["outcome"]).reset_index()
            table.insert(0, "version", version)
            segment_parts.append(table)
            valid = table["validity"].dropna() if "validity" in table else pd.Series(dtype=float)
            min_validity[version] = float(valid.min()) if len(valid) else float("nan")
        segment_table = pd.concat(segment_parts, ignore_index=True)
        if all(math.isfinite(value) for value in min_validity.values()):
            segment_delta = min_validity[candidate_name] - min_validity[baseline_name]
            dimensions_rows.append(
                {
                    "dimension": "minimum_segment_validity",
                    baseline_name: min_validity[baseline_name],
                    candidate_name: min_validity[candidate_name],
                    "candidate_minus_baseline": segment_delta,
                    "assessment": _descriptive_assessment(
                        segment_delta, higher_is_better=True
                    ),
                    "basis": "descriptive weakest-segment comparison; higher is better",
                }
            )

    actions = _action_comparison(
        frame,
        baseline_name,
        candidate_name,
        action_cutoffs,
        action_percentiles,
        action_top_n,
        t,
    )
    metrics: dict[str, Any] = {
        "n": len(frame),
        "outcome_type": "binary" if binary else "continuous",
        "baseline_polarity": baseline_polarity,
        "candidate_polarity": candidate_polarity,
        "raw_pearson": raw_pearson,
        "raw_spearman": raw_spearman,
        "oriented_spearman": oriented_spearman,
        "mean_absolute_rank_change": float(rank_movements["absolute_rank_change"].mean()),
        "median_absolute_rank_change": float(rank_movements["absolute_rank_change"].median()),
        "max_absolute_rank_change": float(rank_movements["absolute_rank_change"].max()),
        "cross_version_psi": psi(frame["baseline"], frame["candidate"]),
    }
    notes = [
        "All performance, lift, rank, migration, segment, and action comparisons use the same "
        "complete paired entity/outcome sample.",
        "Cross-version PSI is scale-dependent; treat it as descriptive when score ranges or "
        "units changed.",
        "Only downstream performance uses paired inferential uncertainty. Other improved or "
        "regressed labels are descriptive and state their basis in the dimensions table.",
    ]
    if n_bootstrap and valid_bootstraps < n_bootstrap:
        notes.append(
            f"{n_bootstrap - valid_bootstraps} bootstrap samples were excluded because the "
            "resampled outcome lacked variation or produced a non-finite metric."
        )
    return ScoreComparison(
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        coverage=coverage,
        outcome_type="binary" if binary else "continuous",
        metrics=metrics,
        dimensions=pd.DataFrame(dimensions_rows),
        performance=performance,
        distributions=distributions,
        lift=lift,
        migration=migration,
        rank_movements=rank_movements,
        stability=stability,
        segments=segment_table,
        actions=actions,
        notes=notes,
    )
