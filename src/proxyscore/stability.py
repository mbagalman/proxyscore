"""Temporal stability of the score distribution (PSI)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._utils import as_series, fmt
from .config import Thresholds
from .results import CheckResult, Status


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index between two score samples.

    Bin edges are taken from the quantiles of ``expected`` (the baseline),
    extended to cover the full real line. Rules of thumb: < 0.10 stable,
    0.10-0.25 moderate shift, >= 0.25 significant shift.
    """
    expected = np.asarray(pd.Series(expected).dropna(), dtype=float)
    actual = np.asarray(pd.Series(actual).dropna(), dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # nearly-constant baseline: fall back to 2 bins around midpoint
        mid = edges.mean()
        edges = np.array([-np.inf, mid, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_prop = np.clip(e_counts / len(expected), 1e-4, None)
    a_prop = np.clip(a_counts / len(actual), 1e-4, None)
    return float(np.sum((a_prop - e_prop) * np.log(a_prop / e_prop)))


def psi_over_time(
    score, period, baseline_period=None, bins: int = 10
) -> pd.DataFrame:
    """PSI of each period's score distribution against a baseline period.

    ``baseline_period`` defaults to the earliest period (sorted order).
    Returns a DataFrame with one row per non-baseline period.
    """
    s = as_series(score, "score")
    p = as_series(period, "period", index=s.index)
    df = pd.concat([s, p], axis=1).dropna()
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
    """Judge score stability over time using PSI against the baseline period."""
    t = thresholds or Thresholds()
    table = psi_over_time(score, period, baseline_period, bins)
    if len(table) == 0:
        return CheckResult(
            "stability",
            Status.SKIP,
            "Fewer than two periods available - stability not assessed.",
        )
    table["band"] = pd.cut(
        table["psi"],
        [-np.inf, t.psi_stable, t.psi_unstable, np.inf],
        labels=["stable", "moderate_shift", "significant_shift"],
    )
    worst_row = table.loc[table["psi"].idxmax()]
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
            f"Score distribution stable across {len(table) + 1} periods "
            f"(max PSI {fmt(max_psi)})."
        )
    metrics = {"max_psi": max_psi, "n_periods": int(len(table) + 1)}
    notes = [
        "PSI compares each period's score distribution to the baseline period. "
        "Seasonal businesses may show benign PSI spikes; compare like-for-like periods if so."
    ]
    return CheckResult("stability", status, text, metrics, table, notes)
