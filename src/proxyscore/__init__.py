"""proxyscore: construct, validate, and monitor proxy scores for latent
business constructs - for when the business needs a number for something
it cannot directly observe.

Quick start::

    from proxyscore import ProxyAudit

    report = ProxyAudit(
        indicators=df[indicator_cols],
        score=df["health_score"],          # or omit to build one
        outcome=df["churned_next_quarter"],  # delayed hard outcome
        segments=df["plan_tier"],
        period=df["month"],
    ).run()

    print(report.verdict)        # decision_grade / directional / not_validated
    print(report.to_markdown())  # full audit report
"""

from .actions import ActionAnalysis, ActionRecommendation, analyze_actions
from .alignment import AlignmentDiagnostics, AlignmentResult, align_delayed_outcomes
from .audit import AuditReport, ProxyAudit, Verdict
from .bias import check_segments, segment_summary
from .config import Thresholds
from .construct import CompositeScore, PCAScore
from .indicators import (
    check_indicators,
    cronbach_alpha,
    indicator_summary,
    redundant_pairs,
    vif,
)
from .leakage import check_leakage, leakage_scan
from .results import CheckResult, Status
from .stability import check_stability, psi, psi_over_time
from .validation import check_downstream, downstream_validity, lift_table

__version__ = "0.1.0"

__all__ = [
    "AuditReport",
    "AlignmentDiagnostics",
    "AlignmentResult",
    "ActionAnalysis",
    "ActionRecommendation",
    "CheckResult",
    "CompositeScore",
    "PCAScore",
    "ProxyAudit",
    "Status",
    "Thresholds",
    "Verdict",
    "align_delayed_outcomes",
    "analyze_actions",
    "check_downstream",
    "check_indicators",
    "check_leakage",
    "check_segments",
    "check_stability",
    "cronbach_alpha",
    "downstream_validity",
    "indicator_summary",
    "leakage_scan",
    "lift_table",
    "psi",
    "psi_over_time",
    "redundant_pairs",
    "segment_summary",
    "vif",
]
