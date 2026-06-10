"""Downstream (criterion) validation against delayed hard outcomes.

A proxy for a latent construct cannot be validated against the construct
itself - only against observable consequences the construct is supposed to
drive (renewal, conversion, expansion revenue, churn). These functions
quantify how much real signal the score carries about such an outcome.

Important: the caller is responsible for temporal alignment - the outcome
must be measured *after* the window the score was computed from, otherwise
the validation is circular (see :mod:`proxyscore.leakage`).
"""

from __future__ import annotations

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
    fmt,
    is_binary,
    spearman,
    to_binary,
    validate_score,
)
from .config import Thresholds
from .results import CheckResult, Status


def _paired(score, outcome) -> pd.DataFrame:
    """Align and validate score and outcome, dropping incomplete rows."""
    s = as_series(score, "score")
    check_unique_index(s.index, "score")
    validate_score(s)
    y = aligned_series(outcome, "outcome", s.index)
    check_outcome_type(y)
    ensure_finite(y, "outcome")
    return pd.concat([s, y], axis=1).dropna()


def lift_table(score, outcome, n_bands: int = 10, ascending: bool = False) -> pd.DataFrame:
    """Outcome rate and lift per score band (band 1 = highest scores).

    Works for binary outcomes (rate = positive share; two-valued
    non-numeric outcomes are mapped to 0/1 with the larger/later sorted
    value as 1) and continuous outcomes (rate = mean outcome).
    ``ascending=True`` puts the lowest scores in band 1 instead (useful
    for risk scores read low-to-high).
    """
    ensure_count(n_bands, 2, "n_bands")
    df = _paired(score, outcome)
    if not pd.api.types.is_numeric_dtype(df["outcome"]):
        if is_binary(df["outcome"]):
            df["outcome"] = to_binary(df["outcome"])
        else:
            raise TypeError(
                "outcome must be numeric or two-valued; got a non-numeric outcome "
                f"with {df['outcome'].nunique()} distinct values"
            )
    if len(df) < n_bands:
        raise ValueError(f"need at least {n_bands} rows for {n_bands} bands")
    ranks = df["score"].rank(method="first", ascending=ascending)
    df["band"] = pd.qcut(ranks, q=n_bands, labels=range(1, n_bands + 1)).astype(int)
    overall = df["outcome"].mean()
    g = df.groupby("band").agg(
        n=("outcome", "size"),
        score_min=("score", "min"),
        score_max=("score", "max"),
        outcome_rate=("outcome", "mean"),
    )
    g["lift"] = g["outcome_rate"] / overall if overall != 0 else np.nan
    g["cum_capture"] = (
        (g["outcome_rate"] * g["n"]).cumsum() / (overall * len(df)) if overall != 0 else np.nan
    )
    return g.reset_index()


def downstream_validity(score, outcome) -> dict:
    """Headline association metrics between score and outcome.

    Returns a dict with ``outcome_type`` ("binary" or "continuous"),
    ``spearman``, ``polarity`` (+1 if higher score means more outcome,
    -1 otherwise), and for binary outcomes ``auc`` (raw), ``auc_oriented``
    (after applying polarity, so values are >= 0.5 when there is any
    signal), and the per-class counts ``n_pos`` / ``n_neg``.

    Binary outcomes may use any two values (0/1, bool, "no"/"yes"); the
    larger / later-sorting value is treated as the positive class.
    """
    df = _paired(score, outcome)
    binary = is_binary(df["outcome"])
    y = to_binary(df["outcome"]) if binary else df["outcome"]
    rho = spearman(df["score"], y)
    polarity = -1 if (not np.isnan(rho) and rho < 0) else 1
    out: dict = {
        "n": int(len(df)),
        "spearman": rho,
        "polarity": polarity,
    }
    if binary:
        auc = auc_score(df["score"].to_numpy(), y.to_numpy())
        out["outcome_type"] = "binary"
        out["auc"] = auc
        out["auc_oriented"] = auc if polarity == 1 else 1 - auc
        out["base_rate"] = float(y.mean())
        out["n_pos"] = int(y.sum())
        out["n_neg"] = int(len(y) - y.sum())
    else:
        out["outcome_type"] = "continuous"
        pair = pd.concat([df["score"], y], axis=1)
        out["pearson"] = (
            float(pair.corr().iloc[0, 1]) if pair["score"].std() > 0 else float("nan")
        )
    return out


def check_downstream(
    score,
    outcome,
    thresholds: Thresholds | None = None,
    n_bands: int = 10,
) -> CheckResult:
    """Judge whether the score carries decision-grade signal about the outcome."""
    ensure_count(n_bands, 2, "n_bands")
    t = thresholds or Thresholds()
    df = _paired(score, outcome)
    if df["outcome"].nunique() < 2:
        return CheckResult(
            "downstream",
            Status.SKIP,
            "Outcome has no variation - downstream validity cannot be assessed "
            "against a constant outcome.",
        )
    if len(df) < 30:
        return CheckResult(
            "downstream",
            Status.SKIP,
            f"Only {len(df)} rows with both score and outcome - too few to validate.",
        )
    m = downstream_validity(df["score"], df["outcome"])
    if m["outcome_type"] == "binary" and min(m["n_pos"], m["n_neg"]) < t.min_class_count:
        return CheckResult(
            "downstream",
            Status.SKIP,
            f"Only {m['n_pos']} positive and {m['n_neg']} negative outcomes - need at "
            f"least {t.min_class_count} of each. Any AUC on this sample would be too "
            f"unstable to support a verdict.",
            m,
        )
    # len(df) >= 30 here, so bands is always within [2, len(df)]
    bands = min(n_bands, max(2, len(df) // 30))
    details = lift_table(df["score"], df["outcome"], n_bands=bands)

    notes = [
        "Validation is only meaningful if the outcome was observed AFTER the score "
        "window (delayed hard outcome). Re-check the leakage scan if results look "
        "too good to be true.",
    ]
    if m["polarity"] == -1:
        notes.append(
            "Detected negative polarity: higher scores associate with LESS of the outcome "
            "(e.g. health score vs churn). Oriented metrics account for this."
        )

    if m["outcome_type"] == "binary":
        eff = m["auc_oriented"]
        strong, weak = t.min_auc_strong, t.min_auc_weak
        label = f"AUC {fmt(eff)} ({m['n_pos']} positives / {m['n_neg']} negatives)"
    else:
        eff = abs(m["spearman"]) if not np.isnan(m["spearman"]) else 0.0
        strong, weak, label = t.min_corr_strong, t.min_corr_weak, f"|spearman| {fmt(eff)}"

    if np.isnan(eff) or eff < weak:
        status = Status.FAIL
        text = (
            f"No usable downstream signal: {label} (need >= {weak} for directional use). "
            f"The score does not measurably relate to the outcome."
        )
    elif eff < strong:
        status = Status.WARN
        text = (
            f"Moderate downstream signal: {label} (>= {strong} considered strong). "
            f"Usable for directional dashboards, not for automated per-record decisions."
        )
    else:
        status = Status.PASS
        text = f"Strong downstream signal: {label} on n={m['n']}."
    return CheckResult("downstream", status, text, m, details, notes)
