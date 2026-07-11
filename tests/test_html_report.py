from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from proxyscore import (
    AuditReport,
    CheckResult,
    ProxyAudit,
    Status,
    Verdict,
    analyze_actions,
)
from proxyscore.datasets import make_customer_health

FIXED_TIME = datetime(2026, 7, 11, 15, 30, tzinfo=timezone.utc)


def sample_report() -> AuditReport:
    return AuditReport(
        Verdict.DIRECTIONAL,
        "Strong evidence with a review warning.",
        [
            CheckResult(
                "indicators",
                Status.WARN,
                "Review <unsafe> & unusual labels.",
                metrics={"alpha": 0.81234, "missing": 0},
                details=pd.DataFrame(
                    {
                        "indicator": ["safe", "<script>alert(1)</script>", "third"],
                        "value": [1.0, 2.0, 3.0],
                    }
                ),
                notes=["A note with <b>markup</b>."],
            ),
            CheckResult("stability", Status.SKIP, "No period supplied."),
        ],
        scope={
            "audit_rows": 3,
            "indicator_columns": ["safe", "<unsafe>"],
            "outcome_supplied": True,
        },
    )


def test_html_contains_required_sections_statuses_and_inline_assets():
    document = sample_report().to_html(generated_at=FIXED_TIME)

    assert document.startswith("<!doctype html>")
    assert '<meta charset="utf-8">' in document
    assert "<style>" in document
    assert "<link" not in document
    assert "<script" not in document
    assert '<section aria-labelledby="scope-heading">' in document
    assert '<section aria-labelledby="metadata-heading">' in document
    assert '<section aria-labelledby="summary-heading">' in document
    assert '<section aria-labelledby="checks-heading">' in document
    assert "WARN" in document
    assert "SKIP" in document
    assert "calibration" in document
    assert "prospective performance" in document


def test_html_escapes_title_metadata_report_text_tables_and_notes():
    document = sample_report().to_html(
        title="Audit <script>alert(1)</script>",
        metadata={"organization": "A&B <Corp>", "project": 'Risk "pilot"'},
        generated_at=FIXED_TIME,
    )

    assert "<script>alert(1)</script>" not in document
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in document
    assert "A&amp;B &lt;Corp&gt;" in document
    assert "Risk &quot;pilot&quot;" in document
    assert "Review &lt;unsafe&gt; &amp; unusual labels." in document
    assert "&lt;b&gt;markup&lt;/b&gt;" in document


def test_generation_metadata_is_deterministic_when_timestamp_is_supplied():
    first = sample_report().to_html(generated_at=FIXED_TIME)
    second = sample_report().to_html(generated_at=FIXED_TIME)
    assert first == second
    assert "2026-07-11T15:30:00Z" in first
    assert "proxyscore_version" in first


def test_detail_tables_are_transparently_truncated():
    document = sample_report().to_html(max_detail_rows=2, generated_at=FIXED_TIME)
    assert "Showing first 2 of 3 rows." in document
    assert "third" not in document
    assert "indicators details" in document


def test_detail_table_limit_can_be_disabled():
    document = sample_report().to_html(max_detail_rows=None, generated_at=FIXED_TIME)
    assert "third" in document
    assert "Showing first" not in document


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_detail_table_limit_is_rejected(value):
    with pytest.raises(ValueError, match="max_detail_rows"):
        sample_report().to_html(max_detail_rows=value)


def test_write_html_creates_same_utf8_document(tmp_path):
    destination = tmp_path / "audit.html"
    written = sample_report().write_html(destination, generated_at=FIXED_TIME)

    assert written == destination.resolve()
    assert destination.read_text(encoding="utf-8") == sample_report().to_html(
        generated_at=FIXED_TIME
    )


def test_attached_action_analysis_is_included():
    analysis = analyze_actions(
        pd.Series([1.0, 2.0, 3.0, 4.0]),
        pd.Series([0, 0, 1, 1]),
        cutoffs=[2.5, 3.5],
        segments=pd.Series(["small", "small", "large", "large"]),
        true_positive_benefit=100,
        false_positive_cost=20,
    )
    report = sample_report().attach_action_analysis(analysis)
    document = report.to_html(generated_at=FIXED_TIME)

    assert report.action_analysis is analysis
    assert '<section aria-labelledby="action-analysis-heading">' in document
    assert "Evaluated action policies" in document
    assert "Action analysis by segment" in document
    assert "Business-value assumptions" in document
    assert "true_positive_benefit" in document
    assert "expected_value" in document


def test_attach_action_analysis_rejects_wrong_type():
    with pytest.raises(TypeError, match="ActionAnalysis"):
        sample_report().attach_action_analysis("not analysis")  # type: ignore[arg-type]


def test_proxy_audit_populates_input_scope():
    frame = make_customer_health(n=200, seed=17)
    indicators = ["logins", "feature_depth", "support_tickets", "nps"]
    report = ProxyAudit(
        indicators=frame[indicators],
        score=frame["health_score"],
        outcome=frame["churned"],
        segments=frame["segment"],
        period=frame["month"],
    ).run()

    assert report.scope == {
        "original_rows": 200,
        "audit_rows": 200,
        "indicator_count": 4,
        "indicator_columns": indicators,
        "score_supplied": True,
        "outcome_supplied": True,
        "segments_supplied": True,
        "period_supplied": True,
    }
    document = report.to_html(generated_at=FIXED_TIME)
    assert "original_rows" in document
    assert "logins, feature_depth, support_tickets, nps" in document


def test_empty_report_still_renders_accessible_sections():
    report = AuditReport(Verdict.NOT_VALIDATED, "No checks were run.")
    document = report.to_html(generated_at=FIXED_TIME)
    assert "No scope metadata available" not in document
    assert "None supplied." in document
    assert "Audit check summary" in document
    assert "<tbody>\n  </tbody>" in document
