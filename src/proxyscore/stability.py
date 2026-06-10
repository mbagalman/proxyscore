"""Temporal stability of the score distribution (PSI)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._utils import (
    aligned_series,
    as_series,
    check_unique_index,
    ensure_count,
    fmt,
    validate_score,
)
from .config import Thresholds
from .results import CheckResult, Status


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index between two score samples.

    Bin edges are taken from the quantiles of ``expected`` (the baseline),
    extended to cover the full real line. Rules of thumb: < 0.10 stable,
    0.10-0.25 moderate shift, >= 0.25 significant shift.
    """
    ensure_count(bins, 2, "bins")
    expected_s = pd.Series(expected).dropna()
    actual_s = pd.Series(actual).dropna()
    for name, sample in (("expected", expected_s), ("actual", actual_s)):
        if pd.api.types.is_complex_dtype(sample):
            raise TypeError(f"psi {name} sample must be real-valued, got complex dtype")
    expected = np.asarray(expected_s, dtype=float)
    actual = np.asarray(actual_s, dtype=float)
    if np.isinf(expected).any() or np.isinf(actual).any():
        raise ValueError(
            "psi inputs contain infinite values; replace them with NaN or finite "
            "values first."
        )
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # nearly-constant baseline: bracket the midpoint so a shift in EITHER
        # direction leaves the middle bin and registers as a change
        mid = edges.mean()
        half = (edges.max() - edges.min()) / 2
        eps = half if half > 0 else max(abs(mid), 1.0) * 1e-9
        edges = np.array([-np.inf, mid - eps, mid + eps, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_prop = np.clip(e_counts / len(expected), 1e-4, None)
    a_prop = np.clip(a_counts / len(actual), 1e-4, None)
    return float(np.sum((a_prop - e_prop) * np.log(a_prop / e_prop)))


def _score_period_frame(score, period) -> pd.DataFrame:
    s = as_series(score, "score")
    check_unique_index(s.index, "score")
    validate_score(s)
    p = aligned_series(period, "period", s.index)
    return pd.concat([s, p], axis=1).dropna()


def psi_over_time(score, period, baseline_period=None, bins: int = 10) -> pd.DataFrame:
    """PSI of each period's score distribution against a baseline period.

    ``baseline_period`` defaults to the earliest period (sorted order).
    Returns a DataFrame with one row per non-baseline period (``period``,
    ``n``, ``psi``).
    """
    ensure_count(bins, 2, "bins")
    df = _score_period_frame(score, period)
    periods = sorted(df["period"].unique())
    if len(periods) < 2:
        return pd.DataFrame(columns=["period", "n", "psi"])
    base = baseline_period if baseline_period is not None else periods[0]
    if base not in periods:
        raise ValueError(f"baseline_period {base!r} not found in period values")
    base_scores = df.loc[df["period"] == base, "score"]
    rows = []
    for per in periods:
        if per == base:
            continue
        cur = df.loc[df["period"] == per, "score"]
        rows.append({"period": per, "n": int(len(cur)), "psi": psi(base_scores, cur, bins)})
    return pd.DataFrame(rows)


def check_stability(
    score,
    period,
    baseline_period=None,
    bins: int = 10,
    thresholds: Thresholds | None = None,
) -> CheckResult:
    """Judge score stability over time using PSI against the baseline period.

    Periods (including the baseline) with fewer than
    ``thresholds.min_period_rows`` rows are too noisy for fixed PSI
    thresholds: an undersized baseline skips the check, and undersized
    comparison periods are excluded from grading and listed in the notes.
    """
    t = thresholds or Thresholds()
    ensure_count(bins, 2, "bins")
    df = _score_period_frame(score, period)
    periods = sorted(df["period"].unique())
    if len(periods) < 2:
        return CheckResult(
            "stability",
            Status.SKIP,
            "Fewer than two periods available - stability not assessed.",
        )
    base = baseline_period if baseline_period is not None else periods[0]
    if base not in periods:
        raise ValueError(f"baseline_period {base!r} not found in period values")
    base_n = int((df["period"] == base).sum())
    if base_n < t.min_period_rows:
        return CheckResult(
            "stability",
            Status.SKIP,
            f"Baseline period {base!r} has only {base_n} rows "
            f"(min_period_rows={t.min_period_rows}) - PSI would be too noisy to grade.",
        )
    table = psi_over_time(df["score"], df["period"], baseline_period, bins)
    table["underpowered"] = table["n"] < t.min_period_rows
    table["band"] = pd.cut(
        table["psi"],
        [-np.inf, t.psi_stable, t.psi_unstable, np.inf],
        labels=["stable", "moderate_shift", "significant_shift"],
    )
    powered = table[~table["underpowered"]]
    notes = [
        "PSI compares each period's score distribution to the baseline period. "
        "Seasonal businesses may show benign PSI spikes; compare like-for-like periods if so."
    ]
    if table["underpowered"].any():
        skipped = list(table.loc[table["underpowered"], "period"])
        notes.append(
            f"Excluded {len(skipped)} period(s) with fewer than {t.min_period_rows} rows "
            f"from grading: {skipped}"
        )
    if len(powered) == 0:
        return CheckResult(
            "stability",
            Status.SKIP,
            f"No comparison period has at least {t.min_period_rows} rows - "
            f"stability not graded.",
            {"n_periods": int(len(table) + 1)},
            table,
            notes,
        )
    worst_row = powered.loc[powered["psi"].idxmax()]
    max_psi = float(worst_row["psi"])
    if max_psi >= t.psi_unstable:
        status = Status.FAIL
        text = (
            f"Significant distribution shift: PSI reached {fmt(max_psi)} in period "
            f"{worst_row['period']!r} (threshold {t.psi_unstable}). Scores from different "
            f"periods are not comparable; recalibrate before using thresholds or trends."
        )
    elif max_psi >= t.psi_stable:
        status = Status.WARN
        text = (
            f"Moderate distribution shift: max PSI {fmt(max_psi)} in period "
            f"{worst_row['period']!r} (stable < {t.psi_stable}). Review whether the shift "
            f"reflects the business or the measurement."
        )
    else:
        status = Status.PASS
        text = (
            f"Score distribution stable across {len(powered) + 1} graded periods "
            f"(max PSI {fmt(max_psi)})."
        )
    metrics = {
        "max_psi": max_psi,
        "n_periods": int(len(table) + 1),
        "n_graded_periods": int(len(powered) + 1),
        "baseline_n": base_n,
    }
    return CheckResult("stability", status, text, metrics, table, notes)
