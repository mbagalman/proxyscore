"""PCA loading-drift assessment against a fixed fitted baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._utils import as_indicator_frame, check_unique_index, ensure_count
from .construct import PCAScore


@dataclass
class PCALoadingDriftAssessment:
    """Sign-aligned PCA loading comparison with current-sample uncertainty."""

    input_rows: int
    complete_rows: int
    dropped_rows: int
    cosine_similarity: float
    cosine_ci_lower: float
    cosine_ci_upper: float
    max_abs_loading_delta: float
    max_abs_loading_delta_ci_lower: float
    max_abs_loading_delta_ci_upper: float
    baseline_explained_variance_ratio: float
    current_explained_variance_ratio: float
    explained_variance_delta: float
    explained_variance_delta_ci_lower: float
    explained_variance_delta_ci_upper: float
    valid_bootstrap_samples: int
    loadings: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def metrics(self) -> dict[str, float | int]:
        """Return report-ready scalar drift metrics."""
        return {
            "input_rows": self.input_rows,
            "complete_rows": self.complete_rows,
            "dropped_rows": self.dropped_rows,
            "cosine_similarity": self.cosine_similarity,
            "cosine_ci_lower": self.cosine_ci_lower,
            "cosine_ci_upper": self.cosine_ci_upper,
            "max_abs_loading_delta": self.max_abs_loading_delta,
            "max_abs_loading_delta_ci_lower": self.max_abs_loading_delta_ci_lower,
            "max_abs_loading_delta_ci_upper": self.max_abs_loading_delta_ci_upper,
            "baseline_explained_variance_ratio": self.baseline_explained_variance_ratio,
            "current_explained_variance_ratio": self.current_explained_variance_ratio,
            "explained_variance_delta": self.explained_variance_delta,
            "explained_variance_delta_ci_lower": self.explained_variance_delta_ci_lower,
            "explained_variance_delta_ci_upper": self.explained_variance_delta_ci_upper,
            "valid_bootstrap_samples": self.valid_bootstrap_samples,
        }

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return the report-ready per-indicator loading table."""
        return {"loadings": self.loadings}

    def to_markdown(self) -> str:
        """Render scalar metrics, loading deltas, warnings, and interpretation notes."""
        lines = ["# PCA loading drift", ""]
        for name, value in self.metrics().items():
            rendered = f"{value:.4g}" if isinstance(value, float) else str(value)
            lines.append(f"- **{name.replace('_', ' ').title()}:** {rendered}")
        lines += ["", "## Loadings", ""]
        try:
            lines.append(self.loadings.to_markdown(index=False, floatfmt=".4g"))
        except ImportError:
            lines.append("```\n" + self.loadings.to_string(index=False) + "\n```")
        if self.warnings:
            lines += ["", "## Warnings", ""] + [f"> {item}" for item in self.warnings]
        if self.notes:
            lines += ["", "## Notes", ""] + [f"> {item}" for item in self.notes]
        return "\n".join(lines) + "\n"


def _fit_component(values: np.ndarray) -> tuple[np.ndarray, float]:
    means = values.mean(axis=0)
    standard_deviations = values.std(axis=0, ddof=0)
    safe_standard_deviations = np.where(standard_deviations == 0, 1.0, standard_deviations)
    standardized = (values - means) / safe_standard_deviations
    _, singular_values, right_vectors = np.linalg.svd(
        standardized - standardized.mean(axis=0), full_matrices=False
    )
    variance = singular_values**2
    total_variance = float(variance.sum())
    if total_variance == 0:
        raise ValueError("current indicators contain no varying principal direction")
    return right_vectors[0], float(variance[0] / total_variance)


def _align(
    baseline: np.ndarray, current: np.ndarray
) -> tuple[np.ndarray, float]:
    baseline_norm = float(np.linalg.norm(baseline))
    current_norm = float(np.linalg.norm(current))
    if baseline_norm == 0 or current_norm == 0:
        raise ValueError("PCA loading vectors must have non-zero length")
    similarity = float(np.dot(baseline, current) / (baseline_norm * current_norm))
    if similarity < 0:
        current = -current
        similarity = -similarity
    return current, similarity


def _interval(values: list[float], confidence_level: float) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    alpha = (1 - confidence_level) / 2
    lower, upper = np.quantile(np.asarray(values), [alpha, 1 - alpha])
    return float(lower), float(upper)


def assess_pca_loading_drift(
    baseline: PCAScore,
    indicators: pd.DataFrame,
    *,
    min_sample_size: int = 100,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
    random_state: int | None = 0,
) -> PCALoadingDriftAssessment:
    """Compare a later sample with one fitted, unchanged ``PCAScore``.

    The later sample is standardized and fitted independently for diagnostic
    comparison only. Its first-component sign is aligned to the stored baseline
    loading vector before similarity and loading deltas are calculated. The
    supplied baseline object is never fitted or mutated.
    """
    if not isinstance(baseline, PCAScore):
        raise TypeError("baseline must be a fitted PCAScore")
    if (
        baseline.columns_ is None
        or baseline.loadings_ is None
        or baseline.explained_variance_ratio_ is None
    ):
        raise ValueError("baseline must be a fitted PCAScore with loading state")
    ensure_count(min_sample_size, 3, "min_sample_size")
    ensure_count(n_bootstrap, 0, "n_bootstrap")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not math.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError("confidence_level must be a finite number in (0, 1)")

    frame = as_indicator_frame(indicators)
    check_unique_index(frame.index, "indicators")
    missing_columns = [column for column in baseline.columns_ if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"current indicators are missing baseline columns: {missing_columns}")
    selected = frame.loc[:, baseline.columns_]
    complete = selected.dropna()
    if len(complete) < min_sample_size:
        raise ValueError(
            f"PCA loading drift requires at least {min_sample_size} complete rows; "
            f"found {len(complete)} from {len(frame)} input rows"
        )

    baseline_values = baseline.loadings_.loc[baseline.columns_].to_numpy(dtype=float)
    current_values, current_explained = _fit_component(complete.to_numpy(dtype=float))
    aligned_current, cosine = _align(baseline_values, current_values)
    loading_delta = aligned_current - baseline_values
    max_abs_delta = float(np.max(np.abs(loading_delta)))
    baseline_explained = float(baseline.explained_variance_ratio_)
    explained_delta = current_explained - baseline_explained

    bootstrap_loadings: list[np.ndarray] = []
    bootstrap_cosines: list[float] = []
    bootstrap_max_deltas: list[float] = []
    bootstrap_explained_deltas: list[float] = []
    values = complete.to_numpy(dtype=float)
    if n_bootstrap:
        rng = np.random.default_rng(random_state)
        for _ in range(n_bootstrap):
            sampled = values[rng.integers(0, len(values), size=len(values))]
            try:
                sampled_loadings, sampled_explained = _fit_component(sampled)
                aligned_sampled, sampled_cosine = _align(
                    baseline_values, sampled_loadings
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            bootstrap_loadings.append(aligned_sampled)
            bootstrap_cosines.append(sampled_cosine)
            bootstrap_max_deltas.append(
                float(np.max(np.abs(aligned_sampled - baseline_values)))
            )
            bootstrap_explained_deltas.append(sampled_explained - baseline_explained)

    cosine_lower, cosine_upper = _interval(bootstrap_cosines, confidence_level)
    max_delta_lower, max_delta_upper = _interval(
        bootstrap_max_deltas, confidence_level
    )
    explained_lower, explained_upper = _interval(
        bootstrap_explained_deltas, confidence_level
    )
    if bootstrap_loadings:
        loading_samples = np.vstack(bootstrap_loadings)
        alpha = (1 - confidence_level) / 2
        loading_lower, loading_upper = np.quantile(
            loading_samples, [alpha, 1 - alpha], axis=0
        )
        delta_samples = loading_samples - baseline_values
        delta_lower, delta_upper = np.quantile(
            delta_samples, [alpha, 1 - alpha], axis=0
        )
    else:
        loading_lower = loading_upper = np.full(len(baseline_values), np.nan)
        delta_lower = delta_upper = np.full(len(baseline_values), np.nan)

    loading_table = pd.DataFrame(
        {
            "indicator": baseline.columns_,
            "baseline_loading": baseline_values,
            "current_loading": aligned_current,
            "loading_delta": loading_delta,
            "abs_loading_delta": np.abs(loading_delta),
            "current_loading_ci_lower": loading_lower,
            "current_loading_ci_upper": loading_upper,
            "loading_delta_ci_lower": delta_lower,
            "loading_delta_ci_upper": delta_upper,
        }
    )
    dropped_rows = len(frame) - len(complete)
    warnings: list[str] = []
    if dropped_rows:
        warnings.append(
            f"{dropped_rows} row(s) with missing baseline indicators were excluded; "
            "the diagnostic PCA uses complete rows only."
        )
    if not n_bootstrap:
        warnings.append("No bootstrap requested; uncertainty intervals are unavailable.")
    elif len(bootstrap_cosines) < n_bootstrap:
        warnings.append(
            f"{n_bootstrap - len(bootstrap_cosines)} bootstrap sample(s) had no usable "
            "principal direction and were excluded."
        )
    notes = [
        "The stored baseline loadings and explained variance are fixed approved state and are "
        "never refitted by this assessment.",
        "The current PCA is diagnostic only; production scores continue to use the baseline "
        "PCAScore transform.",
        "Bootstrap intervals represent current-batch sampling uncertainty only; baseline-fit "
        "uncertainty is unavailable without the original baseline rows.",
        "PCA signs are arbitrary, so every current and bootstrap loading vector is sign-aligned "
        "to the stored baseline before comparison.",
    ]
    return PCALoadingDriftAssessment(
        input_rows=len(frame),
        complete_rows=len(complete),
        dropped_rows=dropped_rows,
        cosine_similarity=cosine,
        cosine_ci_lower=cosine_lower,
        cosine_ci_upper=cosine_upper,
        max_abs_loading_delta=max_abs_delta,
        max_abs_loading_delta_ci_lower=max_delta_lower,
        max_abs_loading_delta_ci_upper=max_delta_upper,
        baseline_explained_variance_ratio=baseline_explained,
        current_explained_variance_ratio=current_explained,
        explained_variance_delta=explained_delta,
        explained_variance_delta_ci_lower=explained_lower,
        explained_variance_delta_ci_upper=explained_upper,
        valid_bootstrap_samples=len(bootstrap_cosines),
        loadings=loading_table,
        warnings=warnings,
        notes=notes,
    )
