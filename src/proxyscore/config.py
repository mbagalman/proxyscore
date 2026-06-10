"""Tunable thresholds for all checks.

Defaults follow common rules of thumb from psychometrics, credit-risk
monitoring, and applied ML. Every threshold can be overridden by passing a
custom :class:`Thresholds` to :class:`proxyscore.ProxyAudit` or to the
individual ``check_*`` functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Thresholds:
    # --- indicator quality -------------------------------------------------
    #: warn when an indicator is missing in more than this share of rows
    max_missing_rate: float = 0.20
    #: warn when an indicator's item-rest correlation is below this
    #: (only meaningful for reflective constructs)
    min_item_rest_corr: float = 0.10
    #: warn when Cronbach's alpha is below this (reflective constructs)
    min_cronbach_alpha: float = 0.70
    #: warn when a pair of indicators correlates above this
    max_pairwise_corr: float = 0.90
    #: warn when an indicator's variance inflation factor exceeds this
    max_vif: float = 10.0
    #: warn when a single indicator explains the composite almost entirely
    max_score_indicator_corr: float = 0.95

    # --- stability (PSI) ---------------------------------------------------
    #: PSI below this = stable
    psi_stable: float = 0.10
    #: PSI at or above this = significant shift
    psi_unstable: float = 0.25
    #: periods with fewer rows than this are too noisy to judge with PSI
    min_period_rows: int = 50

    # --- downstream validation --------------------------------------------
    #: binary outcomes: AUC at or above this = strong signal
    min_auc_strong: float = 0.65
    #: binary outcomes: AUC below this = no usable signal
    min_auc_weak: float = 0.55
    #: continuous outcomes: |spearman| at or above this = strong signal
    min_corr_strong: float = 0.30
    #: continuous outcomes: |spearman| below this = no usable signal
    min_corr_weak: float = 0.10
    #: binary outcomes: minimum count of EACH class required to validate
    min_class_count: int = 10

    # --- segment bias --------------------------------------------------
    #: warn when a segment's standardized mean difference exceeds this
    max_segment_smd: float = 0.50
    #: warn when per-segment AUC spread (max - min) exceeds this
    max_segment_auc_gap: float = 0.10
    #: warn when per-segment |spearman| spread exceeds this
    max_segment_corr_gap: float = 0.20
    #: segments smaller than this are skipped (too noisy to judge)
    min_segment_size: int = 30

    # --- leakage -------------------------------------------------------
    #: fail when an indicator's standalone AUC against the outcome is at
    #: or above this (or at or below 1 minus this)
    leak_auc: float = 0.90
    #: fail when an indicator's |spearman| with the outcome is at or above this
    leak_corr: float = 0.80
    #: indicators with fewer overlapping outcome rows than this are
    #: reported as unassessed rather than clean
    min_leak_rows: int = 10
    #: column-name fragments that suggest the indicator encodes the outcome
    leak_name_patterns: list[str] = field(
        default_factory=lambda: [
            "churn",
            "renew",
            "cancel",
            "terminat",
            "outcome",
            "target",
            "label",
            "won",
            "lost",
            "converted",
            "closed",
        ]
    )

    def __post_init__(self) -> None:
        unit_interval = [
            "max_missing_rate",
            "min_item_rest_corr",
            "min_cronbach_alpha",
            "max_pairwise_corr",
            "max_score_indicator_corr",
            "min_corr_strong",
            "min_corr_weak",
            "leak_corr",
        ]
        for name in unit_interval:
            v = getattr(self, name)
            if not 0 <= v <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        if not 0 <= self.psi_stable < self.psi_unstable:
            raise ValueError(
                f"require 0 <= psi_stable < psi_unstable, got "
                f"{self.psi_stable} / {self.psi_unstable}"
            )
        if not 0.5 <= self.min_auc_weak <= self.min_auc_strong <= 1:
            raise ValueError(
                f"require 0.5 <= min_auc_weak <= min_auc_strong <= 1, got "
                f"{self.min_auc_weak} / {self.min_auc_strong}"
            )
        if not self.min_corr_weak <= self.min_corr_strong:
            raise ValueError("require min_corr_weak <= min_corr_strong")
        if not 0.5 <= self.leak_auc <= 1:
            raise ValueError(f"leak_auc must be in [0.5, 1], got {self.leak_auc}")
        if self.max_vif <= 1:
            raise ValueError(f"max_vif must be > 1, got {self.max_vif}")
        gaps = (self.max_segment_smd, self.max_segment_auc_gap, self.max_segment_corr_gap)
        if any(g <= 0 for g in gaps):
            raise ValueError("segment gap thresholds must be positive")
        for name in ("min_segment_size", "min_class_count", "min_leak_rows", "min_period_rows"):
            v = getattr(self, name)
            if not (isinstance(v, int) and v >= 1):
                raise ValueError(f"{name} must be a positive integer, got {v!r}")
