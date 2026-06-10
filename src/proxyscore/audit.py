"""The decision-grade audit: run every applicable check, grade the proxy."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from ._utils import aligned_series, as_indicator_frame, check_unique_index, validate_score
from .bias import check_segments
from .config import Thresholds
from .construct import CompositeScore
from .indicators import check_indicators
from .leakage import check_leakage
from .results import CheckResult, Status
from .stability import check_stability
from .validation import check_downstream


class Verdict(str, Enum):
    """Overall grade of the proxy score.

    - ``decision_grade``: validated against a delayed hard outcome with
      strong signal, no failed checks - usable for per-record decisions
      (prioritization, alerts, automation) within the validated scope.
    - ``directional``: no failed checks, but signal is moderate or some
      evidence is missing - usable for dashboards and trend reading,
      not for automated per-record action.
    - ``not_validated``: at least one failed check, or no outcome
      provided - the score is an untested hypothesis, not a measurement.
    """

    DECISION_GRADE = "decision_grade"
    DIRECTIONAL = "directional"
    NOT_VALIDATED = "not_validated"


@dataclass
class AuditReport:
    """Container for all check results plus the overall verdict."""

    verdict: Verdict
    verdict_reason: str
    results: list[CheckResult] = field(default_factory=list)

    def __getitem__(self, name: str) -> CheckResult:
        for r in self.results:
            if r.name == name:
                return r
        raise KeyError(name)

    def summary(self) -> pd.DataFrame:
        """One row per check: name, status, summary text."""
        return pd.DataFrame(
            [
                {"check": r.name, "status": r.status.value, "summary": r.summary}
                for r in self.results
            ]
        )

    def to_markdown(self) -> str:
        """Full report as a markdown document."""
        lines = [
            "# Proxy score audit",
            "",
            f"**Verdict: `{self.verdict.value}`** - {self.verdict_reason}",
            "",
            "| Check | Status | Summary |",
            "| --- | --- | --- |",
        ]
        for r in self.results:
            summary = r.summary.replace("|", "\\|")
            lines.append(f"| {r.name} | {r.status.symbol} | {summary} |")
        for r in self.results:
            lines += ["", f"## {r.name}", ""]
            lines.append(f"**{r.status.symbol}** {r.summary}")
            if r.metrics:
                lines += ["", "Metrics:", ""]
                for k, v in r.metrics.items():
                    if isinstance(v, float):
                        lines.append(f"- {k}: {v:.4g}")
                    else:
                        lines.append(f"- {k}: {v}")
            if r.details is not None and len(r.details) > 0:
                try:
                    table = r.details.to_markdown(index=False, floatfmt=".3f")
                except ImportError:  # tabulate not installed
                    table = "```\n" + r.details.to_string(index=False) + "\n```"
                lines += ["", table]
            for n in r.notes:
                lines += ["", f"> {n}"]
        lines.append("")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        checks = ", ".join(f"{r.name}={r.status.value}" for r in self.results)
        return f"AuditReport(verdict={self.verdict.value!r}, {checks})"


class ProxyAudit:
    """Audit a proxy score for a latent business construct.

    Parameters
    ----------
    indicators:
        DataFrame of numeric indicator columns (one row per entity, or per
        entity-period).
    score:
        The proxy score to audit. If omitted, an equal-weight z-score
        composite of the indicators is built and audited (and the report
        notes this).
    outcome:
        Optional delayed hard outcome (binary or continuous) used for
        downstream validation, segment-validity, and leakage checks.
        Must be observed *after* the score window.
    segments:
        Optional categorical segment labels (plan tier, region, ...).
    period:
        Optional period labels (months, quarters, ...) for stability
        monitoring. Any sortable values work.
    thresholds:
        Optional :class:`proxyscore.Thresholds` overriding the defaults.

    Examples
    --------
    >>> from proxyscore import ProxyAudit
    >>> from proxyscore.datasets import make_customer_health
    >>> df = make_customer_health()
    >>> indicator_cols = ["logins", "feature_depth", "support_tickets", "nps", "payment_delay_days"]
    >>> report = ProxyAudit(
    ...     indicators=df[indicator_cols],
    ...     score=df["health_score"],
    ...     outcome=df["churned"],
    ...     segments=df["segment"],
    ...     period=df["month"],
    ... ).run()
    >>> report.verdict  # doctest: +SKIP
    <Verdict.DECISION_GRADE: 'decision_grade'>
    """

    def __init__(
        self,
        indicators: pd.DataFrame,
        score=None,
        outcome=None,
        segments=None,
        period=None,
        thresholds: Thresholds | None = None,
    ):
        self.indicators = as_indicator_frame(indicators)
        check_unique_index(self.indicators.index, "indicators")
        idx = self.indicators.index
        self.score_provided = score is not None
        if score is None:
            self.score = CompositeScore().fit_transform(self.indicators)
        else:
            self.score = aligned_series(score, "proxy_score", idx)
            validate_score(self.score)
        self.outcome = aligned_series(outcome, "outcome", idx) if outcome is not None else None
        self.segments = aligned_series(segments, "segment", idx) if segments is not None else None
        self.period = aligned_series(period, "period", idx) if period is not None else None
        self.thresholds = thresholds or Thresholds()

    def run(self) -> AuditReport:
        """Run all applicable checks and grade the score."""
        t = self.thresholds
        results: list[CheckResult] = []

        ind = check_indicators(self.indicators, self.score, t)
        if not self.score_provided:
            ind.notes.insert(
                0,
                "No score was provided; an equal-weight z-score composite of the "
                "indicators was built and audited instead.",
            )
        results.append(ind)

        if self.period is not None:
            results.append(check_stability(self.score, self.period, thresholds=t))
        else:
            results.append(
                CheckResult(
                    "stability", Status.SKIP, "No period provided - stability not assessed."
                )
            )

        if self.outcome is not None:
            results.append(check_downstream(self.score, self.outcome, t))
            results.append(check_leakage(self.indicators, self.outcome, t))
        else:
            results.append(
                CheckResult(
                    "downstream",
                    Status.SKIP,
                    "No outcome provided - the score has not been validated against "
                    "any observable consequence of the construct.",
                )
            )
            results.append(
                CheckResult(
                    "leakage", Status.SKIP, "No outcome provided - leakage not assessed."
                )
            )

        if self.segments is not None:
            results.append(check_segments(self.score, self.segments, self.outcome, t))
        else:
            results.append(
                CheckResult(
                    "segments", Status.SKIP, "No segments provided - bias not assessed."
                )
            )

        verdict, reason = self._grade(results)
        return AuditReport(verdict, reason, results)

    def _grade(self, results: list[CheckResult]) -> tuple[Verdict, str]:
        by_name = {r.name: r for r in results}
        fails = [r.name for r in results if r.status is Status.FAIL]
        warns = [r.name for r in results if r.status is Status.WARN]
        # A SKIP on a check whose input was never supplied means "not
        # applicable". A SKIP despite a supplied input means evidence was
        # missing inside the claimed validation scope - that must not be
        # hidden behind a decision-grade verdict.
        supplied = {
            "stability": self.period is not None,
            "downstream": self.outcome is not None,
            "leakage": self.outcome is not None,
            "segments": self.segments is not None,
        }
        unassessable = [
            r.name
            for r in results
            if r.status is Status.SKIP and supplied.get(r.name, False) and r.name != "downstream"
        ]

        if fails:
            return (
                Verdict.NOT_VALIDATED,
                f"failed check(s): {', '.join(fails)}. Fix these before relying on the score.",
            )
        downstream = by_name["downstream"]
        if downstream.status is Status.SKIP:
            return (
                Verdict.NOT_VALIDATED,
                "no downstream validation was possible (no outcome, or too little "
                "overlapping data). Without it the score is an untested hypothesis - "
                "collect a delayed hard outcome and re-audit.",
            )
        if downstream.status is Status.PASS:
            issues = []
            if warns:
                issues.append(f"warnings ({', '.join(warns)})")
            if unassessable:
                issues.append(
                    f"checks that could not be assessed despite supplied inputs "
                    f"({', '.join(unassessable)})"
                )
            if not issues:
                return (
                    Verdict.DECISION_GRADE,
                    "strong downstream signal and every applicable check passed. Suitable "
                    "for per-record decisions within the validated population and horizon.",
                )
            return (
                Verdict.DIRECTIONAL,
                f"strong downstream signal, but with {' and '.join(issues)}. "
                f"Suitable for dashboards and prioritization; resolve these "
                f"before automating decisions on it.",
            )
        return (
            Verdict.DIRECTIONAL,
            "moderate downstream signal. Suitable for directional dashboards and "
            "trend reading; not sharp enough for automated per-record decisions.",
        )
