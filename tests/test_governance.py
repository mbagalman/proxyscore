from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import pytest

from proxyscore import (
    GOVERNANCE_SCHEMA_VERSION,
    GovernanceContext,
    GovernanceManifest,
    GovernanceVersionError,
    ProxyAudit,
    Thresholds,
    configuration_fingerprint,
    create_governance_manifest,
    create_monitoring_baseline,
    monitor_batch,
    redact_secrets,
)
from proxyscore.datasets import make_customer_health

INDICATORS = ["logins", "feature_depth", "support_tickets", "nps", "payment_delay_days"]
FIXED_TIME = datetime(2026, 7, 11, 16, 0, tzinfo=timezone.utc)


def complete_context(**overrides: Any) -> GovernanceContext:
    values: dict[str, Any] = {
        "score_name": "customer_health",
        "score_version": "2026.1",
        "owner": "Customer analytics",
        "intended_uses": ["retention prioritization", "portfolio monitoring"],
        "prohibited_uses": ["automatic account cancellation"],
        "population": "active B2B customers",
        "data_window": "2026-01-01/2026-03-31",
        "outcome_window": "2026-04-01/2026-06-30",
        "decision_owner": "VP Customer Success",
        "reviewer": "Risk committee",
        "tags": ["customer-health", "quarterly"],
        "dataset_id": "warehouse.snapshot.customer_health.2026q1",
        "code_revision_id": "abc1234",
        "metadata": {"ticket": "BR-008"},
    }
    values.update(overrides)
    return GovernanceContext(**values)


def test_manifest_round_trip_schema_version_and_fingerprint_are_stable() -> None:
    context = complete_context(metadata={"api_token": "secret", "ticket": "BR-008"})
    manifest = create_governance_manifest(
        context,
        row_counts={"audit_rows": 100, "outcome_rows": 90},
        checks={"downstream": {"status": "pass"}},
        thresholds={"min_auc_strong": 0.7},
        configuration={"columns": ["a", "b"], "password": "hidden"},
        generated_at=FIXED_TIME,
        strict=True,
    )
    restored = GovernanceManifest.from_json(manifest.to_json())
    parsed = json.loads(manifest.to_json())

    assert manifest.schema_version == GOVERNANCE_SCHEMA_VERSION
    assert manifest.generated_at == "2026-07-11T16:00:00Z"
    assert restored.to_dict() == manifest.to_dict()
    assert parsed["context"]["metadata"]["api_token"] == "[REDACTED]"
    assert "hidden" not in manifest.to_json()
    assert manifest.configuration_fingerprint == create_governance_manifest(
        context,
        row_counts={"audit_rows": 500},
        checks={"different": True},
        thresholds={"min_auc_strong": 0.7},
        configuration={"password": "changed", "columns": ["a", "b"]},
        generated_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
        strict=True,
    ).configuration_fingerprint


def test_unsupported_manifest_schema_is_rejected() -> None:
    manifest = create_governance_manifest(complete_context(), strict=True).to_dict()
    manifest["schema_version"] = "99.0"
    with pytest.raises(GovernanceVersionError, match="unsupported governance schema"):
        GovernanceManifest.from_dict(manifest)


def test_missing_governance_fields_warn_and_strict_mode_fails() -> None:
    manifest = create_governance_manifest({"score_name": "health"})

    assert "Missing governance field: owner" in manifest.warnings
    assert "Missing governance field: intended_uses" in manifest.warnings
    with pytest.raises(ValueError, match="missing required governance field"):
        create_governance_manifest({"score_name": "health"}, strict=True)


def test_redaction_handles_nested_secret_metadata() -> None:
    redacted = redact_secrets(
        {
            "dataset_id": "safe",
            "nested": {"private_key": "abc", "public": "ok"},
            "items": [{"password": "pw"}, {"name": "visible"}],
        }
    )
    assert redacted == {
        "dataset_id": "safe",
        "nested": {"private_key": "[REDACTED]", "public": "ok"},
        "items": [{"password": "[REDACTED]"}, {"name": "visible"}],
    }


def test_configuration_fingerprint_is_key_order_independent_and_redacted() -> None:
    first = configuration_fingerprint({"b": 2, "a": {"token": "one", "x": 1}})
    second = configuration_fingerprint({"a": {"x": 1, "token": "two"}, "b": 2})

    assert first == second


def test_proxy_audit_embeds_governance_manifest_in_markdown_and_html() -> None:
    df = make_customer_health(n=300, seed=71)
    report = ProxyAudit(
        indicators=df[INDICATORS],
        score=df["health_score"],
        outcome=df["churned"],
        segments=df["segment"],
        period=df["month"],
        thresholds=Thresholds(min_audit_rows=100),
        governance=complete_context(),
        governance_strict=True,
    ).run()
    markdown = report.to_markdown()
    html = report.to_html(generated_at=FIXED_TIME)

    assert report.governance_manifest is not None
    assert report.governance_manifest.row_counts["audit_rows"] == 300
    assert report.governance_manifest.checks["downstream"]["status"] in {"pass", "warn", "skip"}
    assert "## Governance manifest" in markdown
    assert "configuration_fingerprint" in markdown
    assert "Governance manifest" in html
    assert "customer_health" in html


def test_monitoring_baseline_and_result_embed_governance_manifest() -> None:
    df = make_customer_health(n=300, seed=72)
    thresholds = Thresholds(min_class_count=5)
    baseline = create_monitoring_baseline(
        df[INDICATORS],
        score_id="customer_health",
        score_version="2026.1",
        score=df["health_score"],
        outcome=df["churned"],
        thresholds=thresholds,
        created_at=FIXED_TIME,
        governance=complete_context(),
        governance_strict=True,
    )
    result = monitor_batch(
        baseline,
        df[INDICATORS],
        score=df["health_score"],
        outcome=df["churned"],
        batch_id="2026-07",
        observed_at=FIXED_TIME,
    )

    assert baseline.governance_manifest is not None
    assert baseline.governance_manifest["row_counts"]["baseline_rows"] == 300
    assert baseline.governance_manifest["thresholds"] == asdict(thresholds)
    assert result.governance_manifest is not None
    assert result.governance_manifest["row_counts"]["batch_rows"] == 300
    assert result.governance_manifest["checks"]["schema"]["status"] == "informational"
    assert "governance_manifest" in result.to_json()
    assert "Governance schema" in result.to_markdown()
    assert "Governance manifest" in result.to_html()
