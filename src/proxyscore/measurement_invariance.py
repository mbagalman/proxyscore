"""Staged multigroup CFA measurement-invariance assessment."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from ._utils import aligned_series, as_indicator_frame, check_unique_index, ensure_count
from .construct_validity import _normalize_constructs

Level = Literal["configural", "metric", "scalar", "strict"]
_LEVELS: tuple[Level, Level, Level, Level] = (
    "configural",
    "metric",
    "scalar",
    "strict",
)
_SEGMENT_COLUMN = "__proxyscore_segment__"


@dataclass
class MeasurementInvarianceAssessment:
    """Results from a prerequisite-gated multigroup CFA invariance ladder."""

    constructs: dict[str, tuple[str, ...]]
    groups: tuple[Any, ...]
    input_rows: int
    complete_rows: int
    dropped_rows: int
    group_sizes: pd.DataFrame
    levels: pd.DataFrame
    parameters: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def highest_supported_level(self) -> str | None:
        """Return the highest consecutively supported level, if any."""
        supported = self.levels.loc[self.levels["supported"], "level"]
        return str(supported.iloc[-1]) if len(supported) else None

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return report-ready sample, model-level, and parameter tables."""
        return {
            "group_sizes": self.group_sizes,
            "levels": self.levels,
            "parameters": self.parameters,
        }

    def to_markdown(self) -> str:
        """Render every invariance level and its prerequisite status."""
        highest = self.highest_supported_level or "none"
        lines = [
            "# Measurement invariance",
            "",
            f"**Sample:** {self.complete_rows} complete rows from {self.input_rows} input rows",
            f"**Highest consecutively supported level:** {highest}",
        ]
        for name, table in self.tables().items():
            lines += ["", f"## {name.replace('_', ' ').title()}", "", _markdown_table(table)]
        if self.warnings:
            lines += ["", "## Warnings", ""] + [f"> {item}" for item in self.warnings]
        if self.notes:
            lines += ["", "## Notes", ""] + [f"> {item}" for item in self.notes]
        return "\n".join(lines) + "\n"


@dataclass
class _GroupStats:
    label: Any
    n: int
    mean: np.ndarray
    covariance: np.ndarray
    logdet_covariance: float


@dataclass
class _Fit:
    level: Level
    converged: bool
    message: str
    chi_square: float
    df: int
    cfi: float
    rmsea: float
    srmr: float
    parameter_count: int
    parameters: list[dict[str, Any]]


def _markdown_table(table: pd.DataFrame) -> str:
    try:
        return table.to_markdown(index=False, floatfmt=".4g")
    except ImportError:
        return "```\n" + table.to_string(index=False) + "\n```"


def _threshold(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return float(value)


def _factor_map(
    constructs: Mapping[str, tuple[str, ...]], selected: Sequence[str]
) -> tuple[np.ndarray, list[int]]:
    factor_for = {
        indicator: factor
        for factor, values in enumerate(constructs.values())
        for indicator in values
    }
    factor_indexes = np.asarray([factor_for[indicator] for indicator in selected], dtype=int)
    markers = [selected.index(values[0]) for values in constructs.values()]
    return factor_indexes, markers


def _group_stats(frame: pd.DataFrame, group: pd.Series, groups: Sequence[Any]) -> list[_GroupStats]:
    stats: list[_GroupStats] = []
    for label in groups:
        values = frame.loc[group == label].to_numpy(dtype=float)
        covariance = np.cov(values, rowvar=False, ddof=0)
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError(
                f"indicator covariance is singular in segment {label!r}; remove redundant "
                "indicators or use a larger sample"
            )
        stats.append(
            _GroupStats(label, len(values), values.mean(axis=0), covariance, float(logdet))
        )
    return stats


def _initial_values(
    level: Level,
    stats: Sequence[_GroupStats],
    factor_indexes: np.ndarray,
    markers: Sequence[int],
) -> np.ndarray:
    g = len(stats)
    k = len(markers)
    values: list[float] = []

    loading_sets = 1 if level != "configural" else g
    for loading_set in range(loading_sets):
        sources = stats if loading_sets == 1 else [stats[loading_set]]
        for indicator, factor in enumerate(factor_indexes):
            marker = markers[factor]
            if indicator != marker:
                estimates = [
                    source.covariance[indicator, marker]
                    / source.covariance[marker, marker]
                    for source in sources
                ]
                values.append(float(np.mean(estimates)))

    intercept_sets = 1 if level in ("scalar", "strict") else g
    for intercept_set in range(intercept_sets):
        sources = stats if intercept_sets == 1 else [stats[intercept_set]]
        values.extend(np.mean([source.mean for source in sources], axis=0).tolist())

    residual_sets = 1 if level == "strict" else g
    for residual_set in range(residual_sets):
        sources = stats if residual_sets == 1 else [stats[residual_set]]
        variances = np.mean([np.diag(source.covariance) for source in sources], axis=0)
        values.extend(np.log(np.maximum(variances * 0.4, 1e-6)).tolist())

    for source in stats:
        marker_variances = np.diag(source.covariance)[list(markers)]
        for row in range(k):
            for column in range(row + 1):
                values.append(
                    math.log(math.sqrt(max(marker_variances[row] * 0.6, 1e-6)))
                    if row == column
                    else 0.0
                )

    if level in ("scalar", "strict"):
        reference = stats[0].mean
        for source in stats[1:]:
            values.extend((source.mean[list(markers)] - reference[list(markers)]).tolist())
    return np.asarray(values, dtype=float)


def _decode(
    values: np.ndarray,
    level: Level,
    group_count: int,
    factor_indexes: np.ndarray,
    markers: Sequence[int],
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    p = len(factor_indexes)
    k = len(markers)
    cursor = 0

    loading_sets: list[np.ndarray] = []
    for _ in range(1 if level != "configural" else group_count):
        loading = np.zeros((p, k))
        for indicator, factor in enumerate(factor_indexes):
            if indicator == markers[factor]:
                loading[indicator, factor] = 1.0
            else:
                loading[indicator, factor] = values[cursor]
                cursor += 1
        loading_sets.append(loading)
    loadings = loading_sets * group_count if len(loading_sets) == 1 else loading_sets

    intercept_sets: list[np.ndarray] = []
    for _ in range(1 if level in ("scalar", "strict") else group_count):
        intercept_sets.append(values[cursor : cursor + p])
        cursor += p
    intercepts = intercept_sets * group_count if len(intercept_sets) == 1 else intercept_sets

    residual_sets: list[np.ndarray] = []
    for _ in range(1 if level == "strict" else group_count):
        residual_sets.append(np.exp(values[cursor : cursor + p]))
        cursor += p
    residuals = residual_sets * group_count if len(residual_sets) == 1 else residual_sets

    factor_covariances: list[np.ndarray] = []
    for _ in range(group_count):
        cholesky = np.zeros((k, k))
        for row in range(k):
            for column in range(row + 1):
                cholesky[row, column] = (
                    math.exp(values[cursor]) if row == column else values[cursor]
                )
                cursor += 1
        factor_covariances.append(cholesky @ cholesky.T)

    factor_means = [np.zeros(k)]
    if level in ("scalar", "strict"):
        for _ in range(1, group_count):
            factor_means.append(values[cursor : cursor + k])
            cursor += k
    else:
        factor_means *= group_count
    if cursor != len(values):
        raise RuntimeError("internal measurement-invariance parameter mismatch")
    return loadings, intercepts, residuals, factor_covariances, factor_means


def _fit_level(
    level: Level,
    stats: Sequence[_GroupStats],
    constructs: Mapping[str, tuple[str, ...]],
    selected: Sequence[str],
    factor_indexes: np.ndarray,
    markers: Sequence[int],
) -> _Fit:
    start = _initial_values(level, stats, factor_indexes, markers)

    def objective(values: np.ndarray) -> float:
        try:
            loadings, intercepts, residuals, factor_covariances, factor_means = _decode(
                values, level, len(stats), factor_indexes, markers
            )
            total = 0.0
            for index, source in enumerate(stats):
                implied_covariance = (
                    loadings[index] @ factor_covariances[index] @ loadings[index].T
                    + np.diag(residuals[index])
                )
                sign, logdet = np.linalg.slogdet(implied_covariance)
                if sign <= 0:
                    return 1e100
                difference = source.mean - (
                    intercepts[index] + loadings[index] @ factor_means[index]
                )
                solved_covariance = np.linalg.solve(implied_covariance, source.covariance)
                solved_difference = np.linalg.solve(implied_covariance, difference)
                total += 0.5 * source.n * (
                    logdet + np.trace(solved_covariance) + difference @ solved_difference
                )
            return float(total)
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError):
            return 1e100

    result = minimize(objective, start, method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-11})
    decoded = _decode(result.x, level, len(stats), factor_indexes, markers)
    loadings, intercepts, residuals, factor_covariances, factor_means = decoded
    chi_square = 0.0
    srmr_sum = 0.0
    total_n = sum(source.n for source in stats)
    for index, source in enumerate(stats):
        implied_covariance = (
            loadings[index] @ factor_covariances[index] @ loadings[index].T
            + np.diag(residuals[index])
        )
        difference = source.mean - (intercepts[index] + loadings[index] @ factor_means[index])
        discrepancy = (
            np.linalg.slogdet(implied_covariance)[1]
            + np.trace(np.linalg.solve(implied_covariance, source.covariance))
            + difference @ np.linalg.solve(implied_covariance, difference)
            - source.logdet_covariance
            - len(selected)
        )
        chi_square += source.n * float(discrepancy)
        scales = np.sqrt(np.outer(np.diag(source.covariance), np.diag(source.covariance)))
        standardized = (source.covariance - implied_covariance) / scales
        lower = standardized[np.tril_indices(len(selected))]
        srmr_sum += source.n * float(np.mean(lower**2))

    observed_moments = len(stats) * len(selected) * (len(selected) + 3) // 2
    parameter_count = len(start)
    df = observed_moments - parameter_count
    baseline_chi_square = 0.0
    for source in stats:
        diagonal = np.diag(np.diag(source.covariance))
        baseline_chi_square += source.n * (
            np.linalg.slogdet(diagonal)[1]
            + np.trace(np.linalg.solve(diagonal, source.covariance))
            - source.logdet_covariance
            - len(selected)
        )
    baseline_df = len(stats) * len(selected) * (len(selected) - 1) // 2
    numerator = max(chi_square - df, 0.0)
    denominator = max(baseline_chi_square - baseline_df, numerator, 1e-12)
    cfi = 1.0 - numerator / denominator
    average_group_n = total_n / len(stats)
    rmsea = math.sqrt(max((chi_square - df) / (df * average_group_n), 0.0))

    parameter_rows: list[dict[str, Any]] = []
    construct_names = list(constructs)
    for group_index, source in enumerate(stats):
        for indicator_index, indicator in enumerate(selected):
            factor = factor_indexes[indicator_index]
            parameter_rows.append(
                {
                    "level": level,
                    "segment": source.label,
                    "construct": construct_names[factor],
                    "indicator": indicator,
                    "loading": loadings[group_index][indicator_index, factor],
                    "intercept": intercepts[group_index][indicator_index],
                    "residual_variance": residuals[group_index][indicator_index],
                    "latent_mean": factor_means[group_index][factor],
                }
            )
    return _Fit(
        level=level,
        converged=bool(result.success and math.isfinite(result.fun)),
        message=str(result.message),
        chi_square=max(float(chi_square), 0.0),
        df=df,
        cfi=float(cfi),
        rmsea=float(rmsea),
        srmr=math.sqrt(srmr_sum / total_n),
        parameter_count=parameter_count,
        parameters=parameter_rows,
    )


def _unassessed_levels(reason: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "level": level,
                "constraints": constraint,
                "converged": False,
                "chi_square": math.nan,
                "df": math.nan,
                "cfi": math.nan,
                "rmsea": math.nan,
                "srmr": math.nan,
                "delta_chi_square": math.nan,
                "delta_df": math.nan,
                "chi_square_p": math.nan,
                "delta_cfi": math.nan,
                "delta_rmsea": math.nan,
                "delta_srmr": math.nan,
                "prerequisite_met": level == "configural",
                "supported": False,
                "interpretation": reason,
            }
            for level, constraint in zip(
                _LEVELS,
                (
                    "same factor pattern",
                    "equal loadings",
                    "equal loadings + intercepts",
                    "equal loadings + intercepts + residuals",
                ),
                strict=True,
            )
        ]
    )


def assess_measurement_invariance(
    indicators: pd.DataFrame,
    segments: Any,
    constructs: Mapping[str, Sequence[str]],
    *,
    min_group_size: int = 100,
    min_cfi: float = 0.90,
    max_rmsea: float = 0.08,
    max_srmr: float = 0.08,
    max_delta_cfi: float = 0.01,
    max_delta_rmsea: float = 0.015,
    max_delta_srmr_metric: float = 0.030,
    max_delta_srmr_scalar: float = 0.010,
    max_delta_srmr_strict: float = 0.010,
) -> MeasurementInvarianceAssessment:
    """Fit a staged continuous-indicator multigroup CFA invariance ladder.

    The model uses maximum likelihood, marker-variable identification, freely
    correlated factors, simple structure, and listwise-complete rows. It is for
    continuous, approximately normal reflective indicators. Ordinal data,
    cross-loadings, correlated errors, partial invariance, robust estimators, and
    longitudinal dependence require specialized SEM software.
    """
    ensure_count(min_group_size, 3, "min_group_size")
    thresholds = {
        name: _threshold(value, name)
        for name, value in {
            "min_cfi": min_cfi,
            "max_rmsea": max_rmsea,
            "max_srmr": max_srmr,
            "max_delta_cfi": max_delta_cfi,
            "max_delta_rmsea": max_delta_rmsea,
            "max_delta_srmr_metric": max_delta_srmr_metric,
            "max_delta_srmr_scalar": max_delta_srmr_scalar,
            "max_delta_srmr_strict": max_delta_srmr_strict,
        }.items()
    }
    frame = as_indicator_frame(indicators)
    check_unique_index(frame.index, "indicators")
    normalized = _normalize_constructs(constructs, frame.columns)
    selected = [indicator for values in normalized.values() for indicator in values]
    segment = aligned_series(segments, _SEGMENT_COLUMN, frame.index)
    combined = pd.concat([frame.loc[:, selected], segment], axis=1).dropna()
    groups = tuple(combined[_SEGMENT_COLUMN].drop_duplicates().tolist())
    if len(groups) < 2:
        raise ValueError("measurement invariance requires at least two observed segments")
    complete = combined.loc[:, selected]
    complete_segment = combined[_SEGMENT_COLUMN]
    sizes = complete_segment.value_counts(sort=False)
    group_sizes = pd.DataFrame(
        {
            "segment": groups,
            "n": [int(sizes[label]) for label in groups],
            "minimum": min_group_size,
            "assessed": [int(sizes[label]) >= min_group_size for label in groups],
        }
    )
    warnings: list[str] = []
    dropped_rows = len(frame) - len(complete)
    if dropped_rows:
        warnings.append(
            f"{dropped_rows} row(s) with missing selected indicators or segment labels were "
            "excluded; every model uses the same complete-case sample."
        )
    sparse = group_sizes.loc[~group_sizes["assessed"], "segment"].tolist()
    notes = [
        "Levels are prerequisite-gated: failure at one level blocks comparability claims at all "
        "later levels.",
        "Metric support permits latent covariance comparisons; scalar support is required for "
        "latent mean comparisons; strict support additionally equates residual variances.",
        "Change-in-fit thresholds follow Chen (2007) as screening guidance, not universal "
        "pass/fail laws.",
        "Use established SEM software for ordinal indicators, robust or missing-data estimators, "
        "partial invariance, cross-loadings, correlated errors, or longitudinal models.",
    ]
    if sparse:
        reason = f"Unassessed because segment(s) {sparse} have n < {min_group_size}."
        warnings.append(reason)
        return MeasurementInvarianceAssessment(
            normalized,
            groups,
            len(frame),
            len(complete),
            dropped_rows,
            group_sizes,
            _unassessed_levels(reason),
            pd.DataFrame(),
            warnings,
            notes,
        )

    constant: list[str] = []
    for label in groups:
        group_frame = complete.loc[complete_segment == label]
        constant.extend(
            f"{label!r}:{column}"
            for column in group_frame.columns[group_frame.nunique(dropna=False) < 2]
        )
    if constant:
        raise ValueError(f"construct indicators must vary within every segment: {constant}")

    factor_indexes, markers = _factor_map(normalized, selected)
    stats = _group_stats(complete, complete_segment, groups)
    fits = [
        _fit_level(level, stats, normalized, selected, factor_indexes, markers)
        for level in _LEVELS
    ]
    rows: list[dict[str, Any]] = []
    parameters: list[dict[str, Any]] = []
    prerequisite = True
    constraints = (
        "same factor pattern",
        "equal loadings",
        "equal loadings + intercepts",
        "equal loadings + intercepts + residuals",
    )
    srmr_limits = (
        math.nan,
        thresholds["max_delta_srmr_metric"],
        thresholds["max_delta_srmr_scalar"],
        thresholds["max_delta_srmr_strict"],
    )
    for index, fit in enumerate(fits):
        previous = fits[index - 1] if index else None
        delta_chi_square = fit.chi_square - previous.chi_square if previous else math.nan
        delta_df = fit.df - previous.df if previous else math.nan
        delta_cfi = previous.cfi - fit.cfi if previous else math.nan
        delta_rmsea = fit.rmsea - previous.rmsea if previous else math.nan
        delta_srmr = fit.srmr - previous.srmr if previous else math.nan
        if index == 0:
            acceptable = (
                fit.cfi >= thresholds["min_cfi"]
                and fit.rmsea <= thresholds["max_rmsea"]
                and fit.srmr <= thresholds["max_srmr"]
            )
        else:
            acceptable = (
                delta_cfi <= thresholds["max_delta_cfi"]
                and delta_rmsea <= thresholds["max_delta_rmsea"]
                and delta_srmr <= srmr_limits[index]
            )
        supported = bool(fit.converged and prerequisite and acceptable)
        if not fit.converged:
            interpretation = f"Not supported: optimizer did not converge ({fit.message})."
        elif not prerequisite:
            interpretation = (
                f"No comparability claim: prerequisite {_LEVELS[index - 1]} level failed."
            )
        elif acceptable:
            interpretation = "Supported within the declared model and thresholds."
        else:
            interpretation = "Not supported by the declared fit thresholds."
        rows.append(
            {
                "level": fit.level,
                "constraints": constraints[index],
                "converged": fit.converged,
                "chi_square": fit.chi_square,
                "df": fit.df,
                "cfi": fit.cfi,
                "rmsea": fit.rmsea,
                "srmr": fit.srmr,
                "delta_chi_square": delta_chi_square,
                "delta_df": delta_df,
                "chi_square_p": (
                    float(chi2.sf(delta_chi_square, delta_df))
                    if previous and delta_df > 0
                    else math.nan
                ),
                "delta_cfi": delta_cfi,
                "delta_rmsea": delta_rmsea,
                "delta_srmr": delta_srmr,
                "prerequisite_met": prerequisite,
                "supported": supported,
                "interpretation": interpretation,
            }
        )
        parameters.extend(fit.parameters)
        prerequisite = supported
    if any(not fit.converged for fit in fits):
        warnings.append(
            "At least one model did not converge; that level and all later levels are "
            "unsupported."
        )
    return MeasurementInvarianceAssessment(
        normalized,
        groups,
        len(frame),
        len(complete),
        dropped_rows,
        group_sizes,
        pd.DataFrame(rows),
        pd.DataFrame(parameters),
        warnings,
        notes,
    )
