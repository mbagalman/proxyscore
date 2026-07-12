"""Exploratory convergent and discriminant validity for named constructs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from ._utils import as_indicator_frame, check_unique_index, ensure_count


@dataclass
class ConstructValidityAssessment:
    """AVE and HTMT estimates calculated on one shared complete-case sample."""

    constructs: dict[str, tuple[str, ...]]
    input_rows: int
    complete_rows: int
    dropped_rows: int
    loadings: pd.DataFrame
    ave: pd.DataFrame
    polarity: pd.DataFrame
    htmt: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return report-ready loading, AVE, polarity, and HTMT+ tables."""
        return {
            "loadings": self.loadings,
            "ave": self.ave,
            "polarity": self.polarity,
            "htmt": self.htmt,
        }

    def to_markdown(self) -> str:
        """Render the assessment without collapsing constructs into one verdict."""
        lines = [
            "# Multi-construct validity",
            "",
            f"**Sample:** {self.complete_rows} complete rows from {self.input_rows} input rows",
        ]
        for name, table in self.tables().items():
            lines += ["", f"## {name.upper()}", "", _markdown_table(table)]
        if self.warnings:
            lines += ["", "## Warnings", ""] + [f"> {warning}" for warning in self.warnings]
        if self.notes:
            lines += ["", "## Notes", ""] + [f"> {note}" for note in self.notes]
        return "\n".join(lines) + "\n"


def _markdown_table(table: pd.DataFrame) -> str:
    try:
        return table.to_markdown(index=False, floatfmt=".4g")
    except ImportError:
        return "```\n" + table.to_string(index=False) + "\n```"


def _validate_threshold(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= 1
    ):
        raise ValueError(f"{name} must be a finite number in (0, 1]")
    return float(value)


def _validate_correlation_threshold(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not -1 <= value <= 1
    ):
        raise ValueError(f"{name} must be a finite number in [-1, 1]")
    return float(value)


def _normalize_constructs(
    constructs: Mapping[str, Sequence[str]], columns: pd.Index
) -> dict[str, tuple[str, ...]]:
    if not isinstance(constructs, Mapping) or len(constructs) < 2:
        raise ValueError("constructs must map at least two names to indicator columns")

    normalized: dict[str, tuple[str, ...]] = {}
    assigned: dict[str, str] = {}
    for name, indicators in constructs.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("construct names must be non-empty strings")
        if isinstance(indicators, (str, bytes)) or not isinstance(indicators, Sequence):
            raise TypeError(f"construct {name!r} indicators must be a sequence of column names")
        values = tuple(indicators)
        if len(values) < 2:
            raise ValueError(f"construct {name!r} must contain at least two indicators")
        if any(not isinstance(value, str) or not value for value in values):
            raise TypeError(f"construct {name!r} indicator names must be non-empty strings")
        if len(set(values)) != len(values):
            raise ValueError(f"construct {name!r} contains duplicate indicator names")
        missing = [value for value in values if value not in columns]
        if missing:
            raise KeyError(f"construct {name!r} references missing columns: {missing}")
        for value in values:
            if value in assigned:
                raise ValueError(
                    f"indicator {value!r} is assigned to both {assigned[value]!r} and {name!r}"
                )
            assigned[value] = name
        normalized[name] = values
    return normalized


def _one_factor_loadings(values: np.ndarray) -> np.ndarray:
    correlation = np.corrcoef(values, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    leading = int(np.argmax(eigenvalues))
    loadings = eigenvectors[:, leading] * math.sqrt(max(float(eigenvalues[leading]), 0.0))
    anchor = int(np.argmax(np.abs(loadings)))
    if loadings[anchor] < 0:
        loadings = -loadings
    return loadings


def _estimate(
    values: np.ndarray,
    constructs: Mapping[str, tuple[str, ...]],
    positions: Mapping[str, int],
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[tuple[str, str], float]]:
    correlation = np.corrcoef(values, rowvar=False)
    loading_values: dict[str, np.ndarray] = {}
    ave_values: dict[str, float] = {}
    monotrait: dict[str, float] = {}

    for name, indicators in constructs.items():
        indexes = [positions[indicator] for indicator in indicators]
        block = values[:, indexes]
        loadings = _one_factor_loadings(block)
        loading_values[name] = loadings
        ave_values[name] = float(np.mean(loadings**2))
        within = np.abs(correlation[np.ix_(indexes, indexes)])
        monotrait[name] = float(within[np.triu_indices(len(indexes), k=1)].mean())

    htmt_values: dict[tuple[str, str], float] = {}
    for left, right in combinations(constructs, 2):
        left_indexes = [positions[indicator] for indicator in constructs[left]]
        right_indexes = [positions[indicator] for indicator in constructs[right]]
        denominator = math.sqrt(monotrait[left] * monotrait[right])
        cross_mean = float(
            np.abs(correlation[np.ix_(left_indexes, right_indexes)]).mean()
        )
        htmt_values[(left, right)] = cross_mean / denominator if denominator > 0 else math.nan
    return loading_values, ave_values, htmt_values


def _polarity_diagnostics(
    values: np.ndarray,
    constructs: Mapping[str, tuple[str, ...]],
    positions: Mapping[str, int],
    minimum: float,
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    correlation = np.corrcoef(values, rowvar=False)
    rows: list[dict[str, Any]] = []
    aligned: dict[str, bool] = {}
    for name, indicators in constructs.items():
        construct_aligned = True
        for indicator_a, indicator_b in combinations(indicators, 2):
            value = float(correlation[positions[indicator_a], positions[indicator_b]])
            pair_aligned = value >= minimum
            construct_aligned &= pair_aligned
            rows.append(
                {
                    "construct": name,
                    "indicator_a": indicator_a,
                    "indicator_b": indicator_b,
                    "correlation": value,
                    "minimum": minimum,
                    "aligned": pair_aligned,
                }
            )
        aligned[name] = construct_aligned
    return rows, aligned


def _interval(values: Sequence[float], confidence_level: float) -> tuple[float, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(finite):
        return math.nan, math.nan
    alpha = (1 - confidence_level) / 2
    lower, upper = np.quantile(finite, [alpha, 1 - alpha])
    return float(lower), float(upper)


def assess_construct_validity(
    indicators: pd.DataFrame,
    constructs: Mapping[str, Sequence[str]],
    *,
    ave_threshold: float = 0.50,
    htmt_threshold: float = 0.85,
    min_within_correlation: float = 0.0,
    min_sample_size: int = 100,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
    random_state: int | None = 0,
) -> ConstructValidityAssessment:
    """Estimate exploratory AVE and HTMT for named reflective constructs.

    AVE uses standardized loadings from the leading component of each construct's
    correlation matrix. HTMT is the mean absolute cross-construct correlation
    divided by the geometric mean of each construct's mean absolute within-
    construct correlation. This absolute-correlation variant is HTMT+. Raw
    within-construct correlations are checked separately for polarity alignment,
    and favorable AVE/HTMT flags are withheld when any pair falls below
    ``min_within_correlation``. Percentile intervals resample rows from one
    shared complete-case analysis sample.

    These diagnostics are screening statistics, not a fitted CFA measurement
    model. Use a structural-equation-modeling package when inference depends on
    factor identification, correlated errors, cross-loadings, ordinal indicators,
    model fit, or latent structural paths.
    """
    ensure_count(min_sample_size, 3, "min_sample_size")
    ensure_count(n_bootstrap, 0, "n_bootstrap")
    ave_threshold = _validate_threshold(ave_threshold, "ave_threshold")
    htmt_threshold = _validate_threshold(htmt_threshold, "htmt_threshold")
    min_within_correlation = _validate_correlation_threshold(
        min_within_correlation, "min_within_correlation"
    )
    confidence_level = _validate_threshold(confidence_level, "confidence_level")
    if confidence_level == 1:
        raise ValueError("confidence_level must be a finite number in (0, 1)")

    frame = as_indicator_frame(indicators)
    check_unique_index(frame.index, "indicators")
    normalized = _normalize_constructs(constructs, frame.columns)
    selected = [indicator for values in normalized.values() for indicator in values]
    complete = frame.loc[:, selected].dropna()
    if len(complete) < min_sample_size:
        raise ValueError(
            f"construct validity requires at least {min_sample_size} complete rows; "
            f"found {len(complete)} from {len(frame)} input rows"
        )
    constant = complete.columns[complete.nunique(dropna=False) < 2].tolist()
    if constant:
        raise ValueError(f"construct indicators must vary in the complete sample: {constant}")

    values = complete.to_numpy(dtype=float)
    positions = {column: index for index, column in enumerate(selected)}
    loading_values, ave_values, htmt_values = _estimate(values, normalized, positions)
    polarity_rows, polarity_aligned = _polarity_diagnostics(
        values, normalized, positions, min_within_correlation
    )

    bootstrap_ave: dict[str, list[float]] = {name: [] for name in normalized}
    bootstrap_htmt: dict[tuple[str, str], list[float]] = {
        pair: [] for pair in htmt_values
    }
    if n_bootstrap:
        rng = np.random.default_rng(random_state)
        for _ in range(n_bootstrap):
            sampled = values[rng.integers(0, len(values), size=len(values))]
            if np.any(np.ptp(sampled, axis=0) == 0):
                continue
            try:
                _, sampled_ave, sampled_htmt = _estimate(sampled, normalized, positions)
            except np.linalg.LinAlgError:
                continue
            for name, value in sampled_ave.items():
                if math.isfinite(value):
                    bootstrap_ave[name].append(value)
            for pair, value in sampled_htmt.items():
                if math.isfinite(value):
                    bootstrap_htmt[pair].append(value)

    loading_rows: list[dict[str, Any]] = []
    ave_rows: list[dict[str, Any]] = []
    for name, construct_indicators in normalized.items():
        for indicator, loading in zip(
            construct_indicators, loading_values[name], strict=True
        ):
            loading_rows.append(
                {"construct": name, "indicator": indicator, "loading": float(loading)}
            )
        lower, upper = _interval(bootstrap_ave[name], confidence_level)
        ave_rows.append(
            {
                "construct": name,
                "indicators": len(construct_indicators),
                "ave": ave_values[name],
                "ci_lower": lower,
                "ci_upper": upper,
                "threshold": ave_threshold,
                "estimate_meets_threshold": ave_values[name] >= ave_threshold,
                "polarity_aligned": polarity_aligned[name],
                "meets_threshold": (
                    ave_values[name] >= ave_threshold and polarity_aligned[name]
                ),
                "valid_bootstrap_samples": len(bootstrap_ave[name]),
            }
        )

    htmt_rows: list[dict[str, Any]] = []
    for pair, value in htmt_values.items():
        lower, upper = _interval(bootstrap_htmt[pair], confidence_level)
        pair_polarity_aligned = polarity_aligned[pair[0]] and polarity_aligned[pair[1]]
        htmt_rows.append(
            {
                "construct_a": pair[0],
                "construct_b": pair[1],
                "htmt": value,
                "ci_lower": lower,
                "ci_upper": upper,
                "threshold": htmt_threshold,
                "estimate_below_threshold": math.isfinite(value) and value < htmt_threshold,
                "polarity_aligned": pair_polarity_aligned,
                "below_threshold": (
                    math.isfinite(value)
                    and value < htmt_threshold
                    and pair_polarity_aligned
                ),
                "valid_bootstrap_samples": len(bootstrap_htmt[pair]),
            }
        )

    warnings: list[str] = []
    dropped_rows = len(frame) - len(complete)
    if dropped_rows:
        warnings.append(
            f"{dropped_rows} row(s) with missing selected indicators were excluded; all "
            "statistics use the same complete-case sample."
        )
    two_indicator = [name for name, columns in normalized.items() if len(columns) == 2]
    if two_indicator:
        warnings.append(
            "Two-indicator constructs have limited identification and less stable AVE/HTMT "
            f"estimates: {two_indicator}."
        )
    polarity_problems = [row for row in polarity_rows if not row["aligned"]]
    if polarity_problems:
        descriptions = [
            f"{row['construct']}:{row['indicator_a']}/{row['indicator_b']} "
            f"({row['correlation']:.3f})"
            for row in polarity_problems
        ]
        warnings.append(
            "Within-construct indicator polarity is unresolved for pair(s) "
            f"{descriptions}; correlations are below min_within_correlation="
            f"{min_within_correlation:.3f}. Recode reversed items or revise the construct. "
            "Favorable AVE and HTMT+ flags involving those constructs are withheld."
        )
    if not n_bootstrap:
        warnings.append("No bootstrap requested; confidence intervals are unavailable.")
    elif any(len(samples) < n_bootstrap for samples in bootstrap_htmt.values()):
        warnings.append(
            "Some HTMT bootstrap samples were non-computable because a resample had no usable "
            "within-construct association."
        )
    unassessed = [pair for pair, value in htmt_values.items() if not math.isfinite(value)]
    if unassessed:
        warnings.append(
            f"HTMT could not be assessed for {unassessed}; within-construct association was zero."
        )

    notes = [
        "AVE is an exploratory one-factor correlation/PCA estimate, not a CFA estimate.",
        "HTMT+ uses absolute correlations, while the polarity table preserves raw signs and "
        "gates favorable threshold flags.",
        "Threshold flags are conventional screening aids, not pass/fail proof of construct "
        "validity.",
        "Use SEM/CFA for model-fit tests, latent-variable inference, ordinal indicators, "
        "cross-loadings, correlated errors, or structural paths.",
    ]
    return ConstructValidityAssessment(
        constructs=normalized,
        input_rows=len(frame),
        complete_rows=len(complete),
        dropped_rows=dropped_rows,
        loadings=pd.DataFrame(loading_rows),
        ave=pd.DataFrame(ave_rows),
        polarity=pd.DataFrame(polarity_rows),
        htmt=pd.DataFrame(htmt_rows),
        warnings=warnings,
        notes=notes,
    )
