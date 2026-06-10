"""Segment bias: does the score mean the same thing for every segment?

Two distinct questions are asked:

1. **Level differences** - do segments get systematically higher or lower
   scores? Measured by standardized mean difference (SMD) of each segment
   versus the rest. A large SMD is not automatically a defect (enterprise
   accounts may genuinely be healthier), but it should be a conscious
   choice, not an artifact.
2. **Validity differences** - does the score predict the outcome equally
   well in every segment? A score that works for SMB but is random noise
   for Enterprise will quietly misallocate attention. Measured by
   per-segment oriented AUC (binary) or |spearman| (continuous).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._utils import as_series, auc_score, fmt, is_binary, spearman, to_binary
from .config import Thresholds
from .results import CheckResult, Status, worst


def segment_summary(score, segments, outcome=None) -> pd.DataFrame:
    """Per-segment score stats, SMD vs rest, and (optionally) validity."""
    s = as_series(score, "score")
    g = as_series(segments, "segment", index=s.index)
    parts = [s, g]
    if outcome is not None:
        parts.append(as_series(outcome, "outcome", index=s.index))
    df = pd.concat(parts, axis=1).dropna(subset=["score", "segment"])

    binary = "outcome" in df and is_binary(df["outcome"].dropna())
    global_rho = spearman(df["score"], df["outcome"]) if "outcome" in df else float("nan")
    polarity = -1 if (not np.isnan(global_rho) and global_rho < 0) else 1

    rows = []
    for seg, sub in df.groupby("segment", observed=True):
        rest = df.loc[df["segment"] != seg, "score"]
        if len(rest) > 1 and len(sub) > 1:
            pooled = np.sqrt((sub["score"].var(ddof=1) + rest.var(ddof=1)) / 2)
        else:
            pooled = np.nan
        smd = (sub["score"].mean() - rest.mean()) / pooled if pooled and pooled > 0 else np.nan
        row = {
            "segment": seg,
            "n": int(len(sub)),
            "score_mean": float(sub["score"].mean()),
            "score_std": float(sub["score"].std(ddof=1)) if len(sub) > 1 else np.nan,
            "smd_vs_rest": float(smd) if pd.notna(smd) else np.nan,
        }
        if "outcome" in df:
            sub_o = sub.dropna(subset=["outcome"])
            if binary:
                classes = sub_o["outcome"].nunique()
                if classes == 2:
                    auc = auc_score(
                        sub_o["score"].to_numpy(), to_binary(sub_o["outcome"]).to_numpy()
                    )
                    row["validity"] = auc if polarity == 1 else 1 - auc
                else:
                    row["validity"] = np.nan
                row["outcome_rate"] = (
                    float(to_binary(sub_o["outcome"]).mean()) if len(sub_o) else np.nan
                )
            else:
                rho = spearman(sub_o["score"], sub_o["outcome"])
                # orient by global polarity so inverted segments show as negative
                row["validity"] = rho * polarity if not np.isnan(rho) else np.nan
                row["outcome_mean"] = float(sub_o["outcome"].mean()) if len(sub_o) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("segment")


def check_segments(
    score,
    segments,
    outcome=None,
    thresholds: Thresholds | None = None,
) -> CheckResult:
    """Audit score level and score validity across segments."""
    t = thresholds or Thresholds()
    table = segment_summary(score, segments, outcome)
    if len(table) < 2:
        return CheckResult(
            "segments", Status.SKIP, "Fewer than two segments - bias not assessed."
        )
    small = table[table["n"] < t.min_segment_size]
    eval_table = table[table["n"] >= t.min_segment_size]
    if len(eval_table) < 2:
        return CheckResult(
            "segments",
            Status.SKIP,
            f"Fewer than two segments with n >= {t.min_segment_size} - bias not assessed.",
            details=table.reset_index(),
        )

    statuses: list[Status] = []
    problems: list[str] = []
    notes = [
        "A large score-level gap between segments (|SMD|) is not automatically bias - "
        "segments can genuinely differ. Verify the gap matches outcome reality "
        "(see per-segment outcome rates in the details table).",
    ]
    if len(small) > 0:
        notes.append(
            f"Skipped {len(small)} segment(s) with n < {t.min_segment_size}: "
            f"{list(small.index)}"
        )

    big_smd = eval_table[eval_table["smd_vs_rest"].abs() > t.max_segment_smd]
    if len(big_smd) > 0:
        statuses.append(Status.WARN)
        desc = ", ".join(f"{i} (SMD {r:.2f})" for i, r in big_smd["smd_vs_rest"].items())
        problems.append(f"large score-level gaps: {desc}")

    binary = None
    if "validity" in eval_table.columns:
        v = eval_table["validity"].dropna()
        if len(v) >= 2:
            binary = "outcome_rate" in eval_table.columns
            gap = float(v.max() - v.min())
            gap_limit = t.max_segment_auc_gap if binary else t.max_segment_corr_gap
            floor = 0.5 if binary else 0.0
            useless = v[v <= floor + 1e-9]
            if len(useless) > 0:
                statuses.append(Status.FAIL)
                problems.append(
                    f"score has no (or inverted) signal in segment(s): {list(useless.index)} "
                    f"- decisions there would be arbitrary"
                )
            elif gap > gap_limit:
                statuses.append(Status.WARN)
                problems.append(
                    f"validity gap across segments is {fmt(gap)} "
                    f"(weakest: {v.idxmin()} at {fmt(v.min())}, strongest: {v.idxmax()} "
                    f"at {fmt(v.max())}; limit {gap_limit})"
                )

    status = worst(statuses)
    if status is Status.PASS:
        text = f"Score levels and validity are consistent across {len(eval_table)} segments."
    else:
        text = "; ".join(problems)
    metrics = {
        "n_segments": int(len(eval_table)),
        "max_abs_smd": float(eval_table["smd_vs_rest"].abs().max()),
    }
    if "validity" in eval_table.columns and eval_table["validity"].notna().sum() >= 2:
        v = eval_table["validity"].dropna()
        metrics["validity_gap"] = float(v.max() - v.min())
        metrics["min_validity"] = float(v.min())
    return CheckResult("segments", status, text, metrics, table.reset_index(), notes)
