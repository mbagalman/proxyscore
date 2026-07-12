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
from .calibration import (
    CalibrationAssessment,
    CalibrationModel,
    assess_calibration,
    fit_and_assess_calibration,
    fit_calibrator,
)
from .comparison import ComparisonCoverage, ScoreComparison, compare_scores
from .config import Thresholds
from .construct import CompositeScore, PCAScore
from .governance import (
    GOVERNANCE_SCHEMA_VERSION,
    GovernanceContext,
    GovernanceManifest,
    GovernanceVersionError,
    configuration_fingerprint,
    create_governance_manifest,
    redact_secrets,
)
from .indicators import (
    check_indicators,
    cronbach_alpha,
    indicator_summary,
    redundant_pairs,
    vif,
)
from .leakage import check_leakage, leakage_scan
from .monitoring import (
    ArtifactVersionError,
    MonitoringBaseline,
    MonitoringCheck,
    MonitoringLimits,
    MonitoringResult,
    MonitorStatus,
    create_monitoring_baseline,
    monitor_batch,
)
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
    "ArtifactVersionError",
    "CalibrationAssessment",
    "CalibrationModel",
    "CheckResult",
    "CompositeScore",
    "ComparisonCoverage",
    "GOVERNANCE_SCHEMA_VERSION",
    "GovernanceContext",
    "GovernanceManifest",
    "GovernanceVersionError",
    "PCAScore",
    "MonitorStatus",
    "MonitoringBaseline",
    "MonitoringCheck",
    "MonitoringLimits",
    "MonitoringResult",
    "ProxyAudit",
    "ScoreComparison",
    "Status",
    "Thresholds",
    "Verdict",
    "align_delayed_outcomes",
    "analyze_actions",
    "assess_calibration",
    "check_downstream",
    "check_indicators",
    "check_leakage",
    "check_segments",
    "check_stability",
    "compare_scores",
    "configuration_fingerprint",
    "create_monitoring_baseline",
    "create_governance_manifest",
    "cronbach_alpha",
    "downstream_validity",
    "fit_and_assess_calibration",
    "fit_calibrator",
    "indicator_summary",
    "leakage_scan",
    "lift_table",
    "monitor_batch",
    "psi",
    "psi_over_time",
    "redact_secrets",
    "redundant_pairs",
    "segment_summary",
    "vif",
]
