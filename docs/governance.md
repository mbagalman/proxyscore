# Governance and reproducibility manifests

BR-008 adds a typed governance context and generated manifest so audit and monitoring artifacts
can be approved, reproduced, and reviewed without exposing raw rows or secrets.

## Governance context

Use `GovernanceContext` to describe the business scope of a score:

```python
from proxyscore import GovernanceContext

governance = GovernanceContext(
    score_name="customer_health",
    score_version="2026.1",
    owner="Customer analytics",
    intended_uses=["retention prioritization", "portfolio monitoring"],
    prohibited_uses=["automatic account cancellation"],
    population="active B2B customers",
    data_window="2026-01-01/2026-03-31",
    outcome_window="2026-04-01/2026-06-30",
    decision_owner="VP Customer Success",
    reviewer="Risk committee",
    tags=["customer-health", "quarterly"],
    dataset_id="warehouse.snapshot.customer_health.2026q1",
    code_revision_id="abc1234",
)
```

`dataset_id` and `code_revision_id` are vendor-neutral strings. They can be warehouse snapshot
IDs, data catalog IDs, Git SHAs, model registry revisions, or any other stable identifiers your
workflow already uses.

## Audit reports

Pass the context to `ProxyAudit`. The resulting `AuditReport` includes a `governance_manifest`
and embeds a readable manifest summary in Markdown and HTML exports.

```python
from proxyscore import ProxyAudit

report = ProxyAudit(
    indicators=df[indicator_columns],
    score=df["health_score"],
    outcome=df["churned_next_quarter"],
    period=df["month"],
    governance=governance,
    governance_strict=True,
).run()

print(report.governance_manifest.configuration_fingerprint)
print(report.to_markdown())
```

Strict mode raises when required governance fields are missing. Without strict mode, the manifest
is still created and includes warnings such as `Missing governance field: owner`.

## Monitoring artifacts

Monitoring baselines and run results also carry governance manifests:

```python
from proxyscore import create_monitoring_baseline, monitor_batch

baseline = create_monitoring_baseline(
    reference[indicator_columns],
    score_id="customer_health",
    score_version="2026.1",
    score=reference["health_score"],
    outcome=reference["churned_next_quarter"],
    governance=governance,
    governance_strict=True,
)

result = monitor_batch(
    baseline,
    current[indicator_columns],
    score=current["health_score"],
    batch_id="2026-07",
)
```

The baseline manifest records baseline row counts, score and indicator counts, package version,
thresholds, constructor-state availability, and a deterministic fingerprint. The run manifest
records batch row counts and the monitoring checks that actually ran.

## JSON schema

The manifest schema is versioned with `schema_version: "1.0"`. Unsupported versions raise
`GovernanceVersionError`; no silent migration is attempted.

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-07-11T16:00:00Z",
  "package_version": "0.1.0",
  "context": {
    "score_name": "customer_health",
    "score_version": "2026.1",
    "owner": "Customer analytics",
    "intended_uses": ["retention prioritization"],
    "prohibited_uses": ["automatic account cancellation"],
    "population": "active B2B customers",
    "data_window": "2026-01-01/2026-03-31",
    "outcome_window": "2026-04-01/2026-06-30",
    "decision_owner": "VP Customer Success",
    "reviewer": "Risk committee",
    "tags": ["customer-health"],
    "dataset_id": "warehouse.snapshot.customer_health.2026q1",
    "code_revision_id": "abc1234",
    "metadata": {}
  },
  "row_counts": {
    "audit_rows": 3000,
    "outcome_rows": 3000
  },
  "checks": {
    "downstream": {
      "status": "pass",
      "metrics": {
        "auc_oriented": 0.784
      }
    }
  },
  "thresholds": {
    "min_auc_strong": 0.7
  },
  "configuration_fingerprint": "fca4c7f0...",
  "warnings": []
}
```

The fingerprint is computed from redacted governance context, thresholds, and explicit
configuration. It intentionally excludes runtime, row counts, and check outcomes so the same
approved configuration has the same fingerprint across repeated runs.

## Privacy and redaction

The manifest stores aggregate metadata only. It does not store row-level indicator, score,
outcome, segment, or period values. Keys that look like credentials, tokens, passwords, secrets,
connection strings, or private keys are replaced with `[REDACTED]` before serialization and before
the configuration fingerprint is computed.

Use `redact_secrets()` or `configuration_fingerprint()` directly when you need the same behavior
for a surrounding workflow.
