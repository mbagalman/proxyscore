"""Probability calibration mappings and held-out calibration assessment."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

from ._utils import (
    aligned_series,
    as_series,
    check_outcome_type,
    check_unique_index,
    ensure_count,
    ensure_finite,
    is_binary,
    to_binary,
    validate_score,
)

CalibrationMethod = Literal["logistic", "isotonic"]
CALIBRATION_MODEL_VERSION = 1


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-12, 1 - 1e-12)
    return cast(np.ndarray, np.log(clipped / (1 - clipped)))


def _validate_probability(values: pd.Series, what: str) -> None:
    validate_score(values, what)
    clean = values.dropna().astype(float)
    if ((clean < 0) | (clean > 1)).any():
        raise ValueError(f"{what} must contain values in [0, 1]")


def _paired_frame(score: Any, outcome: Any) -> pd.DataFrame:
    score_series = as_series(score, "score")
    check_unique_index(score_series.index, "score")
    validate_score(score_series)
    outcome_series = aligned_series(outcome, "outcome", score_series.index)
    check_outcome_type(outcome_series)
    ensure_finite(outcome_series, "outcome")
    if not is_binary(outcome_series):
        raise ValueError("calibration requires a binary outcome with exactly two observed values")
    frame = pd.concat([score_series, to_binary(outcome_series)], axis=1).dropna()
    if frame.empty:
        raise ValueError("score and outcome have no complete observations")
    if frame["outcome"].nunique() != 2:
        raise ValueError("the complete calibration sample must contain both outcome classes")
    return frame.astype(float)


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    if np.unique(x).size == 1:
        prevalence = float(np.clip(y.mean(), 1e-12, 1 - 1e-12))
        return [float(_logit(np.array([prevalence]))[0]), 0.0], []

    design = np.column_stack([np.ones(len(x)), x])

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        linear = design @ parameters
        probabilities = expit(linear)
        loss = float(np.logaddexp(0, linear).sum() - y @ linear)
        gradient = design.T @ (probabilities - y)
        return loss, gradient

    initial = np.array([float(_logit(np.array([y.mean()]))[0]), 0.0])
    fitted = minimize(objective, initial, jac=True, method="BFGS")
    if not fitted.success and not np.all(np.isfinite(fitted.x)):
        raise RuntimeError(f"logistic calibration failed to converge: {fitted.message}")
    return [float(value) for value in fitted.x], []


def _fit_isotonic(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    order = np.argsort(x, kind="stable")
    sorted_x, sorted_y = x[order], y[order]
    unique_x, starts, counts = np.unique(sorted_x, return_index=True, return_counts=True)
    sums = np.add.reduceat(sorted_y, starts).astype(float)
    blocks: list[list[float]] = [
        [float(start), float(start), float(total), float(count)]
        for start, (total, count) in enumerate(zip(sums, counts, strict=True))
    ]
    index = 0
    while index < len(blocks) - 1:
        left, right = blocks[index], blocks[index + 1]
        if left[2] / left[3] <= right[2] / right[3]:
            index += 1
            continue
        merged = [left[0], right[1], left[2] + right[2], left[3] + right[3]]
        blocks[index : index + 2] = [merged]
        index = max(index - 1, 0)

    fitted = np.empty(len(unique_x), dtype=float)
    for start, end, total, count in blocks:
        fitted[int(start) : int(end) + 1] = total / count
    return unique_x.astype(float).tolist(), fitted.tolist()


@dataclass(frozen=True)
class CalibrationModel:
    """Reusable mapping from an arbitrary score to an estimated probability."""

    method: CalibrationMethod
    parameters: list[float]
    fitted_values: list[float]
    fit_sample_size: int
    fit_positive_count: int
    artifact_version: int = CALIBRATION_MODEL_VERSION

    def predict(self, score: Any) -> pd.Series:
        """Map score values to probabilities without refitting."""
        values = as_series(score, "score")
        validate_score(values)
        output = pd.Series(np.nan, index=values.index, name="probability", dtype=float)
        clean = values.dropna().astype(float)
        if self.method == "logistic":
            intercept, slope = self.parameters
            output.loc[clean.index] = expit(intercept + slope * clean.to_numpy())
        else:
            if not self.parameters or len(self.parameters) != len(self.fitted_values):
                raise ValueError("isotonic model has invalid fitted state")
            positions = np.searchsorted(np.asarray(self.parameters), clean.to_numpy(), side="right")
            positions = np.clip(positions - 1, 0, len(self.fitted_values) - 1)
            output.loc[clean.index] = np.asarray(self.fitted_values)[positions]
        return output

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible, versioned model state."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the mapping without row-level training data."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> CalibrationModel:
        """Restore and validate serialized model state."""
        if values.get("artifact_version") != CALIBRATION_MODEL_VERSION:
            raise ValueError("unsupported calibration model artifact_version")
        model = cls(**values)
        if model.method not in ("logistic", "isotonic"):
            raise ValueError(f"unsupported calibration method: {model.method!r}")
        if model.method == "logistic" and len(model.parameters) != 2:
            raise ValueError("logistic model must contain intercept and slope")
        if model.method == "isotonic" and len(model.parameters) != len(model.fitted_values):
            raise ValueError("isotonic model state lengths do not match")
        return model

    @classmethod
    def from_json(cls, document: str) -> CalibrationModel:
        """Restore a model from JSON."""
        values = json.loads(document)
        if not isinstance(values, dict):
            raise ValueError("calibration model JSON must contain an object")
        return cls.from_dict(values)


@dataclass
class CalibrationAssessment:
    """Held-out calibration metrics and reliability-curve data."""

    probabilities: pd.Series
    curve: pd.DataFrame
    metrics: dict[str, float | int | str]
    model: CalibrationModel | None = None
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render a compact, report-ready assessment."""
        lines = ["# Calibration assessment", ""]
        for name, value in self.metrics.items():
            rendered = f"{value:.4g}" if isinstance(value, float) else str(value)
            lines.append(f"- **{name.replace('_', ' ').title()}:** {rendered}")
        lines += ["", "## Calibration curve", ""]
        try:
            lines.append(self.curve.to_markdown(index=False, floatfmt=".4g"))
        except ImportError:
            lines.append("```\n" + self.curve.to_string(index=False) + "\n```")
        if self.warnings:
            lines += ["", "## Warnings", ""] + [f"> {item}" for item in self.warnings]
        if self.notes:
            lines += ["", "## Notes", ""] + [f"> {item}" for item in self.notes]
        return "\n".join(lines) + "\n"


def fit_calibrator(
    score: Any,
    outcome: Any,
    *,
    method: CalibrationMethod = "logistic",
) -> CalibrationModel:
    """Fit a probability mapping on a development sample.

    The returned model contains aggregate fitted state only. Evaluate it on a
    distinct sample with :func:`assess_calibration`.
    """
    if method not in ("logistic", "isotonic"):
        raise ValueError("method must be 'logistic' or 'isotonic'")
    frame = _paired_frame(score, outcome)
    x = frame["score"].to_numpy()
    y = frame["outcome"].to_numpy()
    if method == "logistic":
        parameters, fitted_values = _fit_logistic(x, y)
    else:
        parameters, fitted_values = _fit_isotonic(x, y)
    return CalibrationModel(
        method=method,
        parameters=parameters,
        fitted_values=fitted_values,
        fit_sample_size=len(frame),
        fit_positive_count=int(y.sum()),
    )


def _calibration_regression(probabilities: np.ndarray, outcome: np.ndarray) -> tuple[float, float]:
    predictor = _logit(probabilities)
    if np.unique(predictor).size == 1:
        prevalence = float(np.clip(outcome.mean(), 1e-12, 1 - 1e-12))
        return float(_logit(np.array([prevalence]))[0] - predictor[0]), float("nan")
    parameters, _ = _fit_logistic(predictor, outcome)
    return parameters[0], parameters[1]


def _wilson_interval(positives: int, n: int, confidence_level: float) -> tuple[float, float]:
    z_value = float(norm.ppf(0.5 + confidence_level / 2))
    rate = positives / n
    denominator = 1 + z_value**2 / n
    center = (rate + z_value**2 / (2 * n)) / denominator
    radius = z_value * math.sqrt(rate * (1 - rate) / n + z_value**2 / (4 * n**2))
    return max(0.0, center - radius / denominator), min(1.0, center + radius / denominator)


def assess_calibration(
    score: Any,
    outcome: Any,
    *,
    model: CalibrationModel | None = None,
    assume_probabilities: bool = False,
    bins: int = 10,
    min_bin_size: int = 30,
    confidence_level: float = 0.95,
    n_bootstrap: int = 500,
    random_state: int | None = 0,
) -> CalibrationAssessment:
    """Evaluate probability calibration on a sample not used for fitting.

    Arbitrary scores are never interpreted as probabilities. Supply a fitted
    ``model`` or explicitly set ``assume_probabilities=True`` for an already
    probabilistic score. Curve bins are equal-frequency bins based on stable
    probability ranks; ECE is the sample-weighted absolute bin gap.
    """
    ensure_count(bins, 2, "bins")
    ensure_count(min_bin_size, 1, "min_bin_size")
    ensure_count(n_bootstrap, 0, "n_bootstrap")
    if (
        not isinstance(confidence_level, (int, float))
        or isinstance(confidence_level, bool)
        or not 0 < confidence_level < 1
    ):
        raise ValueError("confidence_level must be a finite number in (0, 1)")
    if model is not None and assume_probabilities:
        raise ValueError("supply model or assume_probabilities=True, not both")
    if model is None and not assume_probabilities:
        raise ValueError(
            "arbitrary scores are not probabilities; supply a fitted model or set "
            "assume_probabilities=True for a genuine probability score"
        )

    frame = _paired_frame(score, outcome)
    probabilities = model.predict(frame["score"]) if model is not None else frame["score"]
    _validate_probability(probabilities, "predicted probabilities")
    frame["probability"] = probabilities
    frame = frame.dropna(subset=["probability"])
    probability_values = frame["probability"].to_numpy()
    outcome_values = frame["outcome"].to_numpy()

    effective_bins = min(bins, len(frame))
    ranks = frame["probability"].rank(method="first")
    frame["bin"] = pd.qcut(ranks, q=effective_bins, labels=False) + 1
    rows: list[dict[str, Any]] = []
    for bin_number, group in frame.groupby("bin", sort=True):
        n = len(group)
        positives = int(group["outcome"].sum())
        lower, upper = _wilson_interval(positives, n, float(confidence_level))
        mean_probability = float(group["probability"].mean())
        observed_rate = positives / n
        rows.append(
            {
                "bin": cast(int, bin_number),
                "n": n,
                "positives": positives,
                "mean_predicted": mean_probability,
                "observed_rate": observed_rate,
                "observed_rate_ci_lower": lower,
                "observed_rate_ci_upper": upper,
                "absolute_gap": abs(mean_probability - observed_rate),
                "sparse": n < min_bin_size,
            }
        )
    curve = pd.DataFrame(rows)
    brier = float(np.mean((probability_values - outcome_values) ** 2))
    ece = float((curve["absolute_gap"] * curve["n"]).sum() / len(frame))
    intercept, slope = _calibration_regression(probability_values, outcome_values)

    bootstrap_values: list[float] = []
    if n_bootstrap:
        rng = np.random.default_rng(random_state)
        for _ in range(n_bootstrap):
            sampled = rng.integers(0, len(frame), size=len(frame))
            bootstrap_values.append(
                float(np.mean((probability_values[sampled] - outcome_values[sampled]) ** 2))
            )
        alpha = (1 - float(confidence_level)) / 2
        brier_lower, brier_upper = np.quantile(bootstrap_values, [alpha, 1 - alpha])
    else:
        brier_lower = brier_upper = float("nan")

    warnings: list[str] = []
    sparse_count = int(curve["sparse"].sum())
    if sparse_count:
        warnings.append(
            f"{sparse_count} of {len(curve)} equal-frequency bins contain fewer than "
            f"min_bin_size={min_bin_size} observations; interpret their observed rates cautiously."
        )
    if np.unique(probability_values).size == 1:
        warnings.append(
            "Predicted probabilities are constant; calibration slope is not identifiable."
        )
    positives = int(outcome_values.sum())
    if min(positives, len(frame) - positives) < min_bin_size:
        warnings.append(
            "The evaluation outcome is severely imbalanced relative to min_bin_size; "
            "calibration estimates may be unstable."
        )
    metrics: dict[str, float | int | str] = {
        "evaluation_sample_size": len(frame),
        "positive_count": positives,
        "positive_rate": float(outcome_values.mean()),
        "brier_score": brier,
        "brier_ci_lower": float(brier_lower),
        "brier_ci_upper": float(brier_upper),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "expected_calibration_error": ece,
        "requested_bins": bins,
        "effective_bins": len(curve),
        "binning": "equal_frequency_stable_rank",
        "confidence_level": float(confidence_level),
        "bootstrap_samples": n_bootstrap,
        "evaluation_source": "fitted_mapping" if model is not None else "explicit_probabilities",
    }
    notes = [
        "Ideal calibration has intercept 0 and slope 1; Brier score and ECE are better near 0.",
        "Bin-rate uncertainty uses Wilson intervals; the Brier interval uses "
        "nonparametric bootstrap.",
    ]
    return CalibrationAssessment(
        probabilities=probabilities,
        curve=curve,
        metrics=metrics,
        model=model,
        warnings=warnings,
        notes=notes,
    )


def fit_and_assess_calibration(
    score: Any,
    outcome: Any,
    *,
    method: CalibrationMethod = "logistic",
    evaluation_fraction: float = 0.25,
    random_state: int | None = 0,
    **assessment_options: Any,
) -> CalibrationAssessment:
    """Fit and assess a mapping with a stratified holdout by default."""
    if (
        isinstance(evaluation_fraction, bool)
        or not isinstance(evaluation_fraction, (int, float))
        or not math.isfinite(evaluation_fraction)
        or not 0 < evaluation_fraction < 1
    ):
        raise ValueError("evaluation_fraction must be a finite number in (0, 1)")
    frame = _paired_frame(score, outcome)
    rng = np.random.default_rng(random_state)
    fit_indices: list[Any] = []
    evaluation_indices: list[Any] = []
    for _, group in frame.groupby("outcome", sort=True):
        order = rng.permutation(len(group))
        evaluation_count = max(1, int(round(len(group) * evaluation_fraction)))
        if evaluation_count >= len(group):
            raise ValueError(
                "each outcome class needs at least two rows for a fit/evaluation split"
            )
        evaluation_indices.extend(group.index[order[:evaluation_count]].tolist())
        fit_indices.extend(group.index[order[evaluation_count:]].tolist())
    fit_frame = frame.loc[fit_indices]
    evaluation_frame = frame.loc[evaluation_indices]
    model = fit_calibrator(fit_frame["score"], fit_frame["outcome"], method=method)
    result = assess_calibration(
        evaluation_frame["score"],
        evaluation_frame["outcome"],
        model=model,
        random_state=random_state,
        **assessment_options,
    )
    result.metrics["fit_sample_size"] = len(fit_frame)
    result.metrics["split_method"] = "stratified_random_holdout"
    result.metrics["evaluation_fraction"] = float(evaluation_fraction)
    result.notes.append(
        "The calibration mapping was fitted only on the development partition and all metrics "
        "were calculated on the disjoint stratified holdout."
    )
    return result
