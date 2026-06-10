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

from ._utils import as_indicator_frame, as_series, auc_score, is_binary, spearman, to_binary
from .config import Thresholds
from .results import CheckResult, Status, worst


def leakage_scan(
    indicators: pd.DataFrame,
    outcome,
    thresholds: Thresholds | None = None,
) -> pd.DataFrame:
    """Per-indicator leakage diagnostics against the outcome.

    Returns a DataFrame with each indicator's association with the outcome
    (oriented AUC for binary outcomes, |spearman| otherwise), whether its
    name matches a leak-suggestive pattern, and a combined flag.
    """
    t = thresholds or Thresholds()
    X = as_indicator_frame(indicators)
    y = as_series(outcome, "outcome", index=X.index)
    binary = is_binary(y.dropna())
    y01 = to_binary(y) if binary else None

    rows = []
    for c in X.columns:
        df = pd.concat([X[c], y01 if binary else y], axis=1).dropna()
        assoc = np.nan
        if len(df) >= 10:
            if binary:
                auc = auc_score(df[c].to_numpy(), df.iloc[:, 1].to_numpy())
                assoc = max(auc, 1 - auc) if not np.isnan(auc) else np.nan
                stat_flag = not np.isnan(assoc) and assoc >= t.leak_auc
            else:
                rho = spearman(df[c], df.iloc[:, 1])
                assoc = abs(rho) if not np.isnan(rho) else np.nan
                stat_flag = not np.isnan(assoc) and assoc >= t.leak_corr
        else:
            stat_flag = False
        name_l = str(c).lower()
        name_flag = any(p in name_l for p in t.leak_name_patterns)
        rows.append(
            {
                "indicator": c,
                "association": float(assoc) if not np.isnan(assoc) else np.nan,
                "association_metric": "oriented_auc" if binary else "abs_spearman",
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

    status = worst(statuses)
    if status is Status.PASS:
        text = "No leakage signals: no indicator is suspiciously close to the outcome."
    else:
        text = "; ".join(problems)
    metrics = {
        "n_statistical_flags": int(table["statistical_flag"].sum()),
        "n_name_flags": int(table["name_flag"].sum()),
        "max_association": float(table["association"].max())
        if table["association"].notna().any()
        else float("nan"),
    }
    notes = [
        "These are heuristics. The only hard guarantee against leakage is a pipeline "
        "where indicators are snapshotted strictly before the outcome window opens."
    ]
    return CheckResult("leakage", status, text, metrics, table.reset_index(), notes)
