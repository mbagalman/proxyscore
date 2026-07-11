"""Operating-threshold and business-action analysis."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from ._utils import (
    aligned_series,
    as_series,
    check_outcome_type,
    check_unique_index,
    ensure_count,
    ensure_finite,
    is_binary,
    spearman,
    to_binary,
    validate_score,
)
from .config import Thresholds

OutcomeType = Literal["binary", "continuous"]
Objective = Literal["expected_value", "f1", "precision", "recall", "youden_j"]


@dataclass(frozen=True)
class ActionRecommendation:
    """An explicitly requested candidate action policy."""

    policy_id: str
    strategy: str
    score_cutoff: float
    selected_n: int
    objective: str
    objective_value: float
    constraints: dict[str, float | int]
    assumptions: dict[str, float]
    sample_size: int
    statement: str


@dataclass
class ActionAnalysis:
    """Structured action-policy results for an evaluated score and outcome."""

    outcome_type: OutcomeType
    polarity: int
    table: pd.DataFrame
    segment_table: pd.DataFrame | None = None
    assumptions: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def recommend(
        self,
        objective: Objective,
        *,
        max_actions: int | None = None,
        min_recall: float | None = None,
    ) -> ActionRecommendation:
        """Select a candidate policy for an explicit objective and constraints.

        This is a sample-specific candidate, not a production authorization. The
        returned statement records the objective, constraints, sample, and economic
        assumptions used to select it.
        """
        allowed: dict[str, set[str]] = {
            "binary": {"expected_value", "f1", "precision", "recall", "youden_j"},
            "continuous": set(),
        }
        if objective not in allowed[self.outcome_type]:
            raise ValueError(
                f"objective {objective!r} is not available for {self.outcome_type} outcomes"
            )
        if objective == "expected_value" and not self.assumptions:
            raise ValueError(
                "expected_value recommendations require explicit business-value assumptions"
            )
        if max_actions is not None:
            ensure_count(max_actions, 0, "max_actions")
        if min_recall is not None:
            if (
                isinstance(min_recall, bool)
                or not isinstance(min_recall, (int, float))
                or not math.isfinite(min_recall)
                or not 0 <= min_recall <= 1
            ):
                raise ValueError("min_recall must be a finite number in [0, 1]")

        candidates = self.table.copy()
        constraints: dict[str, float | int] = {}
        if max_actions is not None:
            candidates = candidates.loc[candidates["selected_n"] <= max_actions]
            constraints["max_actions"] = max_actions
        if min_recall is not None:
            candidates = candidates.loc[candidates["recall"] >= min_recall]
            constraints["min_recall"] = float(min_recall)
        candidates = candidates.loc[candidates[objective].notna()]
        if candidates.empty:
            raise ValueError("no evaluated policy satisfies the requested constraints")

        row = candidates.sort_values(
            [objective, "selected_n", "policy_id"],
            ascending=[False, True, True],
            kind="stable",
        ).iloc[0]
        assumption_text = (
            ", ".join(f"{key}={value:g}" for key, value in self.assumptions.items())
            if self.assumptions
            else "none"
        )
        constraint_text = (
            ", ".join(f"{key}={value:g}" for key, value in constraints.items())
            if constraints
            else "none"
        )
        statement = (
            f"Candidate {row['policy_id']} maximizes {objective}={row[objective]:.4g} "
            f"among the evaluated policies on n={int(row['n'])}; constraints: "
            f"{constraint_text}; assumptions: {assumption_text}. Validate prospectively "
            "before production use."
        )
        return ActionRecommendation(
            policy_id=str(row["policy_id"]),
            strategy=str(row["strategy"]),
            score_cutoff=float(row["score_cutoff"]),
            selected_n=int(row["selected_n"]),
            objective=objective,
            objective_value=float(row[objective]),
            constraints=constraints,
            assumptions=dict(self.assumptions),
            sample_size=int(row["n"]),
            statement=statement,
        )


@dataclass(frozen=True)
class _Policy:
    policy_id: str
    strategy: str
    parameter: float
    score_cutoff: float
    selected: pd.Series


def _numbers(
    values: float | int | Sequence[float] | Sequence[int] | None,
    name: str,
) -> list[float]:
    if values is None:
        return []
    if isinstance(values, bool):
        raise ValueError(f"{name} must contain finite numbers")
    if isinstance(values, (int, float, np.integer, np.floating)):
        raw = [values]
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        raw = list(values)
    else:
        raise TypeError(f"{name} must be a number or sequence of numbers")
    result: list[float] = []
    for value in raw:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{name} must contain finite numbers, got {value!r}")
        result.append(float(value))
    return result


def _economics(
    true_positive_benefit: float | None,
    false_positive_cost: float | None,
    false_negative_cost: float | None,
    action_cost: float | None,
) -> dict[str, float]:
    supplied = {
        "true_positive_benefit": true_positive_benefit,
        "false_positive_cost": false_positive_cost,
        "false_negative_cost": false_negative_cost,
        "action_cost": action_cost,
    }
    if all(value is None for value in supplied.values()):
        return {}
    assumptions: dict[str, float] = {}
    for name, value in supplied.items():
        resolved = 0.0 if value is None else value
        if (
            isinstance(resolved, bool)
            or not isinstance(resolved, (int, float))
            or not math.isfinite(resolved)
            or resolved < 0
        ):
            raise ValueError(f"{name} must be a finite non-negative number")
        assumptions[name] = float(resolved)
    return assumptions


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _binary_metrics(
    selected: pd.Series,
    outcome: pd.Series,
    assumptions: dict[str, float],
) -> dict[str, float | int]:
    chosen = selected.astype(bool)
    positive = outcome.astype(bool)
    tp = int((chosen & positive).sum())
    fp = int((chosen & ~positive).sum())
    tn = int((~chosen & ~positive).sum())
    fn = int((~chosen & positive).sum())
    selected_n = tp + fp
    n = tp + fp + tn + fn
    precision = _safe_ratio(tp, selected_n)
    recall = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    metrics: dict[str, float | int] = {
        "n": n,
        "selected_n": selected_n,
        "selected_rate": _safe_ratio(selected_n, n),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": _safe_ratio(fp, fp + tn),
        "false_negative_rate": _safe_ratio(fn, fn + tp),
        "accuracy": _safe_ratio(tp + tn, n),
        "f1": _safe_ratio(2 * precision * recall, precision + recall),
        "youden_j": recall + specificity - 1,
    }
    if assumptions:
        benefit = assumptions["true_positive_benefit"]
        fp_cost = assumptions["false_positive_cost"]
        fn_cost = assumptions["false_negative_cost"]
        per_action = assumptions["action_cost"]
        expected_value = tp * benefit - fp * fp_cost - fn * fn_cost - selected_n * per_action
        no_action_value = -(tp + fn) * fn_cost
        metrics.update(
            {
                "expected_value": expected_value,
                "expected_value_per_record": _safe_ratio(expected_value, n),
                "net_value_vs_no_action": expected_value - no_action_value,
                "break_even_tp_benefit": (
                    _safe_ratio(fp * fp_cost + fn * fn_cost + selected_n * per_action, tp)
                ),
            }
        )
    return metrics


def _continuous_metrics(selected: pd.Series, outcome: pd.Series) -> dict[str, float | int]:
    chosen = selected.astype(bool)
    selected_outcome = outcome.loc[chosen]
    unselected_outcome = outcome.loc[~chosen]
    selected_mean = float(selected_outcome.mean()) if len(selected_outcome) else float("nan")
    unselected_mean = (
        float(unselected_outcome.mean()) if len(unselected_outcome) else float("nan")
    )
    return {
        "n": int(len(outcome)),
        "selected_n": int(chosen.sum()),
        "selected_rate": float(chosen.mean()),
        "outcome_mean": float(outcome.mean()),
        "selected_outcome_mean": selected_mean,
        "unselected_outcome_mean": unselected_mean,
        "outcome_mean_difference": selected_mean - unselected_mean,
    }


def _make_policies(
    score: pd.Series,
    oriented_score: pd.Series,
    polarity: int,
    cutoffs: list[float],
    percentiles: list[float],
    top_counts: list[float],
    grid_size: int | None,
) -> list[_Policy]:
    policies: list[_Policy] = []

    def add(strategy: str, parameter: float, raw_cutoff: float, selected: pd.Series) -> None:
        policy_id = f"{strategy}_{len(policies) + 1}"
        policies.append(_Policy(policy_id, strategy, parameter, raw_cutoff, selected))

    for cutoff in cutoffs:
        selected = score >= cutoff if polarity == 1 else score <= cutoff
        add("cutoff", cutoff, cutoff, selected)

    for percentile in percentiles:
        if not 0 < percentile <= 100:
            raise ValueError("percentiles must be in (0, 100]")
        oriented_cutoff = float(oriented_score.quantile(1 - percentile / 100))
        add(
            "percentile",
            percentile,
            oriented_cutoff * polarity,
            oriented_score >= oriented_cutoff,
        )

    for raw_count in top_counts:
        if not raw_count.is_integer() or raw_count < 0:
            raise ValueError("top_n must contain integers >= 0")
        count = int(raw_count)
        if count > len(score):
            raise ValueError(f"top_n cannot exceed the usable sample size ({len(score)})")
        selected = pd.Series(False, index=score.index)
        order = oriented_score.sort_values(ascending=False, kind="stable").index
        if count:
            selected.loc[order[:count]] = True
            raw_cutoff = float(score.loc[order[:count]].iloc[-1])
        else:
            raw_cutoff = float("inf") * polarity
        add("top_n", float(count), raw_cutoff, selected)

    if grid_size is not None:
        ensure_count(grid_size, 2, "grid_size")
        quantiles = np.linspace(0, 1, grid_size)
        oriented_cutoffs = oriented_score.quantile(quantiles).drop_duplicates().sort_values()
        for quantile, oriented_cutoff in oriented_cutoffs.items():
            quantile_value = cast(float, quantile)
            add(
                "grid",
                (1 - quantile_value) * 100,
                float(oriented_cutoff) * polarity,
                oriented_score >= float(oriented_cutoff),
            )
    return policies


def analyze_actions(
    score: Any,
    outcome: Any,
    *,
    cutoffs: float | Sequence[float] | None = None,
    percentiles: float | Sequence[float] | None = None,
    top_n: int | Sequence[int] | None = None,
    grid_size: int | None = None,
    polarity: Literal["auto"] | int = "auto",
    segments: Any = None,
    thresholds: Thresholds | None = None,
    true_positive_benefit: float | None = None,
    false_positive_cost: float | None = None,
    false_negative_cost: float | None = None,
    action_cost: float | None = None,
) -> ActionAnalysis:
    """Evaluate candidate score-based action policies.

    ``percentiles`` means the percentage of the usable population selected
    after orienting the score toward more of the outcome. For example, ``10``
    evaluates the highest-priority 10 percent; ties at its empirical cutoff are
    all selected and can make the realized selection rate slightly larger.
    ``top_n`` is capacity-exact and resolves boundary ties by original row order.

    When no policy arguments are supplied, a 20-point empirical cutoff grid is
    generated. Explicit cutoffs are applied as ``score >= cutoff`` for positive
    polarity and ``score <= cutoff`` for negative polarity.
    """
    t = thresholds or Thresholds()
    s = as_series(score, "score")
    check_unique_index(s.index, "score")
    validate_score(s)
    y = aligned_series(outcome, "outcome", s.index)
    check_outcome_type(y)
    ensure_finite(y, "outcome")
    parts = [s, y]
    if segments is not None:
        segment = aligned_series(segments, "segment", s.index)
        parts.append(segment)
    frame = pd.concat(parts, axis=1).dropna(subset=["score", "outcome"])
    if frame.empty:
        raise ValueError("score and outcome have no complete overlapping rows")
    if frame["score"].nunique() < 2:
        raise ValueError("score must contain at least two distinct values")
    if frame["outcome"].nunique() < 2:
        raise ValueError("outcome must contain at least two distinct values")

    binary = is_binary(frame["outcome"])
    outcome_type: OutcomeType = "binary" if binary else "continuous"
    analyzed_outcome = to_binary(frame["outcome"]) if binary else frame["outcome"].astype(float)
    if polarity == "auto":
        rho = spearman(frame["score"], analyzed_outcome)
        if np.isnan(rho):
            raise ValueError("score polarity could not be determined from the usable sample")
        resolved_polarity = -1 if rho < 0 else 1
    elif polarity in (-1, 1) and not isinstance(polarity, bool):
        resolved_polarity = polarity
    else:
        raise ValueError("polarity must be 'auto', 1, or -1")

    assumptions = _economics(
        true_positive_benefit,
        false_positive_cost,
        false_negative_cost,
        action_cost,
    )
    if assumptions and not binary:
        raise ValueError("business-value assumptions are currently supported for binary outcomes")

    cutoff_values = _numbers(cutoffs, "cutoffs")
    percentile_values = _numbers(percentiles, "percentiles")
    top_counts = _numbers(top_n, "top_n")
    if not cutoff_values and not percentile_values and not top_counts and grid_size is None:
        grid_size = 20
    policies = _make_policies(
        frame["score"],
        frame["score"] * resolved_polarity,
        resolved_polarity,
        cutoff_values,
        percentile_values,
        top_counts,
        grid_size,
    )

    rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    for policy in policies:
        metrics = (
            _binary_metrics(policy.selected, analyzed_outcome, assumptions)
            if binary
            else _continuous_metrics(policy.selected, analyzed_outcome)
        )
        rows.append(
            {
                "policy_id": policy.policy_id,
                "strategy": policy.strategy,
                "parameter": policy.parameter,
                "score_cutoff": policy.score_cutoff,
                **metrics,
            }
        )
        if segments is not None:
            for segment_value, group in frame.groupby("segment", observed=True, dropna=False):
                group_selected = policy.selected.loc[group.index]
                group_outcome = analyzed_outcome.loc[group.index]
                n_pos = int(group_outcome.sum()) if binary else None
                n_neg = len(group_outcome) - n_pos if binary and n_pos is not None else None
                assessable = len(group) >= t.min_segment_size
                reason = ""
                if not assessable:
                    reason = f"n < {t.min_segment_size}"
                elif binary and (
                    n_pos is None
                    or n_neg is None
                    or min(n_pos, n_neg) < t.min_class_count
                ):
                    assessable = False
                    reason = f"fewer than {t.min_class_count} rows in an outcome class"
                group_metrics = (
                    _binary_metrics(group_selected, group_outcome, assumptions)
                    if binary and assessable
                    else (
                        _continuous_metrics(group_selected, group_outcome)
                        if not binary and assessable
                        else {
                            "n": int(len(group)),
                            "selected_n": int(group_selected.sum()),
                            "selected_rate": float(group_selected.mean()),
                        }
                    )
                )
                segment_rows.append(
                    {
                        "policy_id": policy.policy_id,
                        "segment": segment_value,
                        "assessed": assessable,
                        "unassessed_reason": reason,
                        **group_metrics,
                    }
                )

    notes = [
        f"Polarity {resolved_polarity:+d}: action selects "
        f"{'higher' if resolved_polarity == 1 else 'lower'} raw scores.",
        "Policies are evaluated on the supplied sample and are candidates only; validate the "
        "chosen operating rule prospectively before production use.",
    ]
    if top_counts:
        notes.append("Top-N policies break score ties by original row order to honor capacity.")
    if assumptions:
        notes.append(
            "Expected value equals TP benefit minus FP cost, FN cost, and per-selected-record "
            "action cost; unspecified economic inputs were recorded as zero."
        )
    return ActionAnalysis(
        outcome_type=outcome_type,
        polarity=resolved_polarity,
        table=pd.DataFrame(rows),
        segment_table=pd.DataFrame(segment_rows) if segments is not None else None,
        assumptions=assumptions,
        notes=notes,
    )
