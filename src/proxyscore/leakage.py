"""Leakage risk: is the proxy secretly built from the outcome?

The most common failure mode of business proxy scores is circularity - an
"engagement score" that includes an indicator only populated once the
customer has already decided to churn, or a "lead quality score" containing
a field set by sales after qualification. Such scores validate spectacularly
and predict nothing going forward.

Two heuristics are applied per indicator:

1. **Statistical**: a standalone association with the outcome that is too
   strong to be plausible for a genuinely upstream signal (default: AUC
   >= 0.90 or <= 0.10, or |spearman| >= 0.80).
2. **Nominal**: column names containing outcome-like fragments
   ("churn", "renewal", "closed_won", ...).

These are heuristics: they cannot prove temporal soundness. The only real
guarantee is a pipeline where indicators are snapshotted strictly before
the outcome window opens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._utils import (
    aligned_series,
    as_indicator_frame,
    auc_score,
    check_unique_index,
    is_binary,
    spearman,
    to_binary,
)
from .config import Thresholds
from .results import CheckResult, Status, worst


def leakage_scan(
    indicators: pd.DataFrame,
    outcome,
    thresholds: Thresholds | None = None,
) -> pd.DataFrame:
    """Per-indicator leakage diagnostics against the outcome.

    Returns a DataFrame with each indicator's association with the outcome
    (oriented AUC for binary outcomes, |spearman| otherwise), the number of
    overlapping rows it was computed on, whether the statistical check was
    assessable at all (``assessed``), whether its name matches a
    leak-suggestive pattern, and the statistical flag.
    """
    t = thresholds or Thresholds()
    X = as_indicator_frame(indicators)
    check_unique_index(X.index, "indicators")
    y = aligned_series(outcome, "outcome", X.index)
    binary = is_binary(y)
    y01 = to_binary(y) if binary else y

    rows = []
    for c in X.columns:
        df = pd.concat([X[c], y01], axis=1).dropna()
        n_overlap = int(len(df))
        assessed = n_overlap >= t.min_leak_rows
        assoc = np.nan
        stat_flag = False
        if assessed:
            if binary:
                auc = auc_score(df[c].to_numpy(), df.iloc[:, 1].to_numpy())
                assoc = max(auc, 1 - auc) if not np.isnan(auc) else np.nan
                stat_flag = not np.isnan(assoc) and assoc >= t.leak_auc
            else:
                rho = spearman(df[c], df.iloc[:, 1])
                assoc = abs(rho) if not np.isnan(rho) else np.nan
                stat_flag = not np.isnan(assoc) and assoc >= t.leak_corr
        name_l = str(c).lower()
        name_flag = any(p in name_l for p in t.leak_name_patterns)
        rows.append(
            {
                "indicator": c,
                "association": float(assoc) if not np.isnan(assoc) else np.nan,
                "association_metric": "oriented_auc" if binary else "abs_spearman",
                "n_overlap": n_overlap,
                "assessed": bool(assessed),
                "statistical_flag": bool(stat_flag),
                "name_flag": bool(name_flag),
            }
        )
    return pd.DataFrame(rows).set_index("indicator")


def check_leakage(
    indicators: pd.DataFrame,
    outcome,
    thresholds: Thresholds | None = None,
) -> CheckResult:
    """Flag indicators that look like they encode the outcome."""
    t = thresholds or Thresholds()
    table = leakage_scan(indicators, outcome, t)

    unassessed = table[~table["assessed"]]
    if len(unassessed) == len(table):
        return CheckResult(
            "leakage",
            Status.SKIP,
            f"No indicator had at least {t.min_leak_rows} rows overlapping the outcome - "
            f"statistical leakage could not be assessed at all.",
            {"n_statistical_flags": 0, "n_unassessed": int(len(table))},
            table.reset_index(),
        )

    statuses: list[Status] = []
    problems: list[str] = []
    stat = table[table["statistical_flag"]]
    if len(stat) > 0:
        statuses.append(Status.FAIL)
        desc = ", ".join(f"{i} ({r:.2f})" for i, r in stat["association"].items())
        problems.append(
            f"indicator(s) with implausibly strong standalone association with the "
            f"outcome: {desc} - likely leakage (indicator measured after, or defined "
            f"by, the outcome)"
        )
    named = table[table["name_flag"] & ~table["statistical_flag"]]
    if len(named) > 0:
        statuses.append(Status.WARN)
        problems.append(
            f"indicator name(s) suggest outcome content: {list(named.index)} - verify "
            f"these are snapshotted strictly before the outcome window"
        )
    if len(unassessed) > 0:
        statuses.append(Status.WARN)
        problems.append(
            f"statistical leakage could not be assessed for {list(unassessed.index)} "
            f"(fewer than {t.min_leak_rows} rows overlapping the outcome)"
        )

    status = worst(statuses)
    if status is Status.PASS:
        text = (
            f"No leakage signals: all {len(table)} indicators assessed, none "
            f"suspiciously close to the outcome."
        )
    else:
        text = "; ".join(problems)
    metrics = {
        "n_statistical_flags": int(table["statistical_flag"].sum()),
        "n_name_flags": int(table["name_flag"].sum()),
        "n_unassessed": int(len(unassessed)),
        "max_association": float(table["association"].max())
        if table["association"].notna().any()
        else float("nan"),
    }
    notes = [
        "These are heuristics. The only hard guarantee against leakage is a pipeline "
        "where indicators are snapshotted strictly before the outcome window opens."
    ]
    return CheckResult("leakage", status, text, metrics, table.reset_index(), notes)
