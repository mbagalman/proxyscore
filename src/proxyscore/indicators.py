"""Indicator quality, reliability, and redundancy diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._utils import as_indicator_frame, fmt, zscore
from .config import Thresholds
from .results import CheckResult, Status, worst


def indicator_summary(indicators: pd.DataFrame) -> pd.DataFrame:
    """Per-indicator quality table: missingness, spread, item-rest correlation.

    The item-rest correlation is each indicator's Pearson correlation with
    the mean of the *other* standardized indicators. High values support a
    reflective reading of the construct; low or negative values are normal
    for formative composites or reverse-oriented indicators.
    """
    X = as_indicator_frame(indicators)
    Z = zscore(X)
    rows = []
    for c in X.columns:
        rest = Z.drop(columns=c).mean(axis=1)
        pair = pd.concat([Z[c], rest], axis=1).dropna()
        if X.shape[1] >= 2 and len(pair) >= 3 and pair.iloc[:, 0].std() > 0:
            item_rest = float(pair.corr().iloc[0, 1])
        else:
            item_rest = float("nan")
        rows.append(
            {
                "indicator": c,
                "missing_rate": float(X[c].isna().mean()),
                "n_unique": int(X[c].nunique()),
                "std": float(X[c].std(ddof=0)),
                "item_rest_corr": item_rest,
            }
        )
    return pd.DataFrame(rows).set_index("indicator")


def cronbach_alpha(indicators: pd.DataFrame) -> float:
    """Cronbach's alpha on standardized indicators (listwise-complete rows).

    Only meaningful for reflective constructs, where indicators are
    expected to covary. Reverse-oriented indicators should be flipped
    first, otherwise alpha is understated.
    """
    X = zscore(as_indicator_frame(indicators)).dropna()
    k = X.shape[1]
    if k < 2 or len(X) < 3:
        return float("nan")
    item_var = X.var(axis=0, ddof=1).sum()
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return float(k / (k - 1) * (1 - item_var / total_var))


def vif(indicators: pd.DataFrame) -> pd.Series:
    """Variance inflation factor per indicator (listwise-complete rows)."""
    X = zscore(as_indicator_frame(indicators)).dropna()
    cols = list(X.columns)
    out = {}
    for c in cols:
        if len(cols) < 2 or len(X) <= len(cols):
            out[c] = float("nan")
            continue
        y = X[c].to_numpy()
        A = X.drop(columns=c).to_numpy()
        A = np.column_stack([np.ones(len(A)), A])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        ss_res = float((resid**2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        if ss_tot == 0:
            out[c] = float("nan")
        elif ss_res / ss_tot < 1e-12:
            out[c] = float("inf")
        else:
            out[c] = float(1.0 / (ss_res / ss_tot))
    return pd.Series(out, name="vif")


def redundant_pairs(indicators: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """Pairs of indicators with |Pearson r| at or above ``threshold``."""
    X = as_indicator_frame(indicators)
    corr = X.corr()
    rows = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                rows.append({"indicator_a": a, "indicator_b": b, "corr": float(r)})
    return pd.DataFrame(rows, columns=["indicator_a", "indicator_b", "corr"])


def check_indicators(
    indicators: pd.DataFrame,
    score: pd.Series | None = None,
    thresholds: Thresholds | None = None,
) -> CheckResult:
    """Run all indicator quality and redundancy diagnostics.

    Flags: excessive missingness, zero variance, weak item-rest correlation
    (note: reflective constructs only), low Cronbach's alpha, near-duplicate
    indicator pairs, high VIF, and a single indicator dominating the score.
    """
    t = thresholds or Thresholds()
    X = as_indicator_frame(indicators)
    summary = indicator_summary(X)
    summary["vif"] = vif(X)
    pairs = redundant_pairs(X, t.max_pairwise_corr)
    alpha = cronbach_alpha(X)

    statuses: list[Status] = []
    problems: list[str] = []
    notes = [
        "Item-rest correlation and Cronbach's alpha assume a reflective construct "
        "(indicators caused by the latent variable, expected to covary). For formative "
        "composites (indicators define the construct), low values are not a defect.",
        "Negative item-rest correlations often mean an indicator is reverse-oriented "
        "(e.g. support tickets in a health score); flip its sign or weight if so.",
    ]

    dead = summary[summary["std"] == 0]
    if len(dead) > 0:
        statuses.append(Status.FAIL)
        problems.append(f"{len(dead)} indicator(s) with zero variance: {list(dead.index)}")

    high_missing = summary[summary["missing_rate"] > t.max_missing_rate]
    if len(high_missing) > 0:
        statuses.append(Status.WARN)
        problems.append(
            f"{len(high_missing)} indicator(s) missing in >{t.max_missing_rate:.0%} of rows: "
            f"{list(high_missing.index)}"
        )

    weak = summary[summary["item_rest_corr"].abs() < t.min_item_rest_corr].dropna(
        subset=["item_rest_corr"]
    )
    if len(weak) > 0:
        statuses.append(Status.WARN)
        problems.append(
            f"{len(weak)} indicator(s) with |item-rest corr| < {t.min_item_rest_corr}: "
            f"{list(weak.index)} (fine if formative)"
        )

    if not np.isnan(alpha) and alpha < t.min_cronbach_alpha:
        statuses.append(Status.WARN)
        problems.append(
            f"Cronbach's alpha {fmt(alpha, 2)} < {t.min_cronbach_alpha} (fine if formative)"
        )

    if len(pairs) > 0:
        statuses.append(Status.WARN)
        pair_desc = ", ".join(
            f"{r.indicator_a}~{r.indicator_b} (r={r.corr:.2f})" for r in pairs.itertuples()
        )
        problems.append(f"near-duplicate indicator pairs: {pair_desc}")

    high_vif = summary[summary["vif"] > t.max_vif].dropna(subset=["vif"])
    if len(high_vif) > 0:
        statuses.append(Status.WARN)
        problems.append(f"VIF > {t.max_vif:g} for: {list(high_vif.index)}")

    if score is not None and X.shape[1] >= 2:
        dom = {}
        for c in X.columns:
            pair = pd.concat([X[c], score], axis=1).dropna()
            if len(pair) >= 3 and pair.iloc[:, 0].std() > 0 and pair.iloc[:, 1].std() > 0:
                dom[c] = abs(float(pair.corr().iloc[0, 1]))
        summary["score_corr_abs"] = pd.Series(dom)
        dominant = [c for c, v in dom.items() if v >= t.max_score_indicator_corr]
        if dominant:
            statuses.append(Status.WARN)
            problems.append(
                f"score is nearly a single indicator (|r| >= {t.max_score_indicator_corr}): "
                f"{dominant} - the composite adds little beyond it"
            )

    status = worst(statuses)
    if status is Status.PASS:
        text = (
            f"{X.shape[1]} indicators look healthy: no zero-variance or high-missing "
            f"indicators, no near-duplicates, no dominance."
        )
    else:
        text = "; ".join(problems)
    metrics = {
        "n_indicators": int(X.shape[1]),
        "cronbach_alpha": alpha,
        "n_redundant_pairs": int(len(pairs)),
        "max_missing_rate": float(summary["missing_rate"].max()),
    }
    return CheckResult("indicators", status, text, metrics, summary.reset_index(), notes)
