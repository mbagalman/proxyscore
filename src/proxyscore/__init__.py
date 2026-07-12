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
from .adapters import (
    AdapterProvenance,
    LocalCSVAdapter,
    LocalParquetAdapter,
    TableProvenance,
    TabularAdapter,
    TabularData,
)
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
from .construct_validity import ConstructValidityAssessment, assess_construct_validity
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
from .measurement_invariance import (
    MeasurementInvarianceAssessment,
    assess_measurement_invariance,
)
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
from .multi_outcome import (
    MultiOutcomeComparison,
    MultiOutcomeReport,
    OutcomeImportance,
    OutcomePolarity,
    OutcomeSpec,
    OutcomeType,
    OutcomeValidation,
    compare_outcomes,
    validate_outcomes,
)
from .pca_drift import PCALoadingDriftAssessment, assess_pca_loading_drift
from .recipes import (
    BusinessRecipe,
    RecipeResult,
    account_risk_recipe,
    customer_health_recipe,
    get_business_recipe,
    lead_quality_recipe,
    list_business_recipes,
)
from .results import CheckResult, Status
from .stability import check_stability, psi, psi_over_time
from .survival_validation import SurvivalValidationAssessment, assess_survival_validation
from .validation import check_downstream, downstream_validity, lift_table

__version__ = "0.1.0"

__all__ = [
    "AuditReport",
    "AlignmentDiagnostics",
    "AlignmentResult",
    "ActionAnalysis",
    "ActionRecommendation",
    "ArtifactVersionError",
    "AdapterProvenance",
    "BusinessRecipe",
    "CalibrationAssessment",
    "CalibrationModel",
    "CheckResult",
    "CompositeScore",
    "ConstructValidityAssessment",
    "ComparisonCoverage",
    "GOVERNANCE_SCHEMA_VERSION",
    "GovernanceContext",
    "GovernanceManifest",
    "GovernanceVersionError",
    "LocalCSVAdapter",
    "LocalParquetAdapter",
    "MeasurementInvarianceAssessment",
    "PCAScore",
    "PCALoadingDriftAssessment",
    "MonitorStatus",
    "MonitoringBaseline",
    "MonitoringCheck",
    "MonitoringLimits",
    "MonitoringResult",
    "MultiOutcomeComparison",
    "MultiOutcomeReport",
    "OutcomeImportance",
    "OutcomePolarity",
    "OutcomeSpec",
    "OutcomeType",
    "OutcomeValidation",
    "ProxyAudit",
    "RecipeResult",
    "ScoreComparison",
    "Status",
    "SurvivalValidationAssessment",
    "TableProvenance",
    "TabularAdapter",
    "TabularData",
    "Thresholds",
    "Verdict",
    "account_risk_recipe",
    "align_delayed_outcomes",
    "analyze_actions",
    "assess_calibration",
    "assess_construct_validity",
    "assess_measurement_invariance",
    "assess_pca_loading_drift",
    "assess_survival_validation",
    "check_downstream",
    "check_indicators",
    "check_leakage",
    "check_segments",
    "check_stability",
    "compare_scores",
    "compare_outcomes",
    "configuration_fingerprint",
    "create_monitoring_baseline",
    "create_governance_manifest",
    "cronbach_alpha",
    "customer_health_recipe",
    "downstream_validity",
    "fit_and_assess_calibration",
    "fit_calibrator",
    "get_business_recipe",
    "indicator_summary",
    "lead_quality_recipe",
    "leakage_scan",
    "list_business_recipes",
    "lift_table",
    "monitor_batch",
    "psi",
    "psi_over_time",
    "redact_secrets",
    "redundant_pairs",
    "segment_summary",
    "vif",
    "validate_outcomes",
]
