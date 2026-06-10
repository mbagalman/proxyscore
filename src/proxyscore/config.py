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

    # --- downstream validation --------------------------------------------
    #: binary outcomes: AUC at or above this = strong signal
    min_auc_strong: float = 0.65
    #: binary outcomes: AUC below this = no usable signal
    min_auc_weak: float = 0.55
    #: continuous outcomes: |spearman| at or above this = strong signal
    min_corr_strong: float = 0.30
    #: continuous outcomes: |spearman| below this = no usable signal
    min_corr_weak: float = 0.10

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
