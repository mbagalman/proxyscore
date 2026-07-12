"""Segment bias: does the score mean the same thing for every segment?

Two distinct questions are asked:

1. **Level differences** - do segments get systematically higher or lower
   scores? Measured by standardized mean difference (SMD) of each segment
   versus the rest, using one pooled within-segment standard deviation for
   every contrast. A large SMD is not automatically a defect (enterprise
   accounts may genuinely be healthier), but it should be a conscious choice,
   not an artifact.
2. **Validity differences** - does the score predict the outcome equally
   well in every segment? A score that works for SMB but is random noise
   for Enterprise will quietly misallocate attention. Measured by
   per-segment oriented AUC (binary) or polarity-oriented spearman
   (continuous). Validity is only graded in segments with enough outcome
   evidence; segments without it are reported as unassessed, never as
   consistent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._utils import (
    aligned_series,
    as_series,
    auc_score,
    check_outcome_type,
    check_unique_index,
    ensure_finite,
    fmt,
    is_binary,
    spearman,
    to_binary,
    validate_score,
)
from .config import Thresholds
from .results import CheckResult, Status, worst


def segment_summary(score: Any, segments: Any, outcome: Any = None) -> pd.DataFrame:
    """Per-segment score stats, common-scale SMD vs rest, and optional validity.

    With an outcome, each row also carries ``n_outcome`` (rows with an
    observed outcome) and, for binary outcomes, ``n_pos`` / ``n_neg`` -
    use these to judge how much evidence backs each segment's validity.
    """
    s = as_series(score, "score")
    check_unique_index(s.index, "score")
    validate_score(s)
    g = aligned_series(segments, "segment", s.index)
    parts = [s, g]
    if outcome is not None:
        y = aligned_series(outcome, "outcome", s.index)
        check_outcome_type(y)
        ensure_finite(y, "outcome")
        parts.append(y)
    df = pd.concat(parts, axis=1).dropna(subset=["score", "segment"])

    binary = "outcome" in df and is_binary(df["outcome"])
    if binary:
        df["outcome"] = to_binary(df["outcome"])
    global_rho = spearman(df["score"], df["outcome"]) if "outcome" in df else float("nan")
    polarity = -1 if (not np.isnan(global_rho) and global_rho < 0) else 1

    segment_means = df.groupby("segment", observed=True)["score"].transform("mean")
    within_degrees_of_freedom = len(df) - df["segment"].nunique()
    pooled_within_std = (
        float(np.sqrt(((df["score"] - segment_means) ** 2).sum() / within_degrees_of_freedom))
        if within_degrees_of_freedom > 0
        else float("nan")
    )

    rows = []
    for seg, sub in df.groupby("segment", observed=True):
        rest = df.loc[df["segment"] != seg, "score"]
        smd = (
            (sub["score"].mean() - rest.mean()) / pooled_within_std
            if len(rest) and pooled_within_std > 0
            else np.nan
        )
        row = {
            "segment": seg,
            "n": int(len(sub)),
            "score_mean": float(sub["score"].mean()),
            "score_std": float(sub["score"].std(ddof=1)) if len(sub) > 1 else np.nan,
            "pooled_within_std": pooled_within_std,
            "smd_vs_rest": float(smd) if pd.notna(smd) else np.nan,
        }
        if "outcome" in df:
            sub_o = sub.dropna(subset=["outcome"])
            row["n_outcome"] = int(len(sub_o))
            if binary:
                n_pos = int(sub_o["outcome"].sum())
                n_neg = int(len(sub_o) - n_pos)
                row["n_pos"], row["n_neg"] = n_pos, n_neg
                row["outcome_rate"] = float(sub_o["outcome"].mean()) if len(sub_o) else np.nan
                if n_pos > 0 and n_neg > 0:
                    auc = auc_score(sub_o["score"].to_numpy(), sub_o["outcome"].to_numpy())
                    row["validity"] = auc if polarity == 1 else 1 - auc
                else:
                    row["validity"] = np.nan
            else:
                rho = spearman(sub_o["score"], sub_o["outcome"])
                # orient by global polarity so inverted segments show as negative
                row["validity"] = rho * polarity if not np.isnan(rho) else np.nan
                row["outcome_mean"] = float(sub_o["outcome"].mean()) if len(sub_o) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("segment")


def check_segments(
    score: Any,
    segments: Any,
    outcome: Any = None,
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
        "Every SMD uses the same ANOVA-style pooled within-segment standard deviation; "
        "between-segment differences never inflate its denominator.",
    ]
    if len(small) > 0:
        # an excluded supplied segment is unresolved evidence, not a footnote -
        # the warning keeps the overall verdict from claiming full coverage
        statuses.append(Status.WARN)
        problems.append(
            f"segment(s) {list(small.index)} excluded from assessment "
            f"(n < {t.min_segment_size}) - score level and validity are unverified there"
        )

    big_smd = eval_table[eval_table["smd_vs_rest"].abs() > t.max_segment_smd]
    if len(big_smd) > 0:
        statuses.append(Status.WARN)
        desc = ", ".join(f"{i} (SMD {r:.2f})" for i, r in big_smd["smd_vs_rest"].items())
        problems.append(f"large score-level gaps: {desc}")

    validity_assessed = False
    if outcome is not None and "validity" in eval_table.columns:
        binary = "n_pos" in eval_table.columns
        if binary:
            assessable = (eval_table["n_pos"] >= t.min_class_count) & (
                eval_table["n_neg"] >= t.min_class_count
            )
        else:
            assessable = eval_table["n_outcome"] >= t.min_segment_size
        # a NaN validity (e.g. constant score or outcome within the segment)
        # is unassessed evidence, not a segment to drop silently
        assessable &= eval_table["validity"].notna()
        unassessed = list(eval_table.index[~assessable])
        if unassessed:
            statuses.append(Status.WARN)
            problems.append(
                f"validity could not be assessed in segment(s) {unassessed} "
                f"(insufficient outcome data or no computable validity) - "
                f"consistency across segments is unproven"
            )
        v = eval_table.loc[assessable, "validity"].dropna()
        if len(v) >= 2:
            validity_assessed = True
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
        elif not unassessed:
            statuses.append(Status.WARN)
            problems.append(
                "per-segment validity could not be compared (fewer than two segments "
                "with computable validity)"
            )

    status = worst(statuses)
    if status is Status.PASS:
        if validity_assessed:
            text = f"Score levels and validity are consistent across {len(eval_table)} segments."
        else:
            text = f"Score levels are consistent across {len(eval_table)} segments."
    else:
        text = "; ".join(problems)
    metrics = {
        "n_segments": int(len(eval_table)),
        "max_abs_smd": float(eval_table["smd_vs_rest"].abs().max()),
        "pooled_within_std": float(eval_table["pooled_within_std"].iloc[0]),
    }
    if validity_assessed:
        metrics["validity_gap"] = float(v.max() - v.min())
        metrics["min_validity"] = float(v.min())
    return CheckResult("segments", status, text, metrics, table.reset_index(), notes)
