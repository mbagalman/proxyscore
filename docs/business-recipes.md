# Business recipes and data adapters

Business score audits are only as honest as their input tables. The built-in recipes provide
repeatable patterns for three common proxy-score use cases without shipping first-party
warehouse, CRM, or BI connectors.

Use the recipes when you already have local tables or query results. Keep credentials, network
access, and vendor-specific extraction code outside `proxyscore`; pass only the minimal audit
tables into the library.

## Adapter protocol

Adapters implement one small protocol:

```python
from proxyscore import TabularAdapter, TabularData


def load_audit_tables(adapter: TabularAdapter) -> TabularData:
    return adapter.load()
```

`TabularData` contains:

- `tables`: a mapping of table name to pandas `DataFrame`.
- `provenance`: adapter, source path, format, row count, column names, and load timestamp.

Local CSV and Parquet adapters are included:

```python
from proxyscore import LocalCSVAdapter, LocalParquetAdapter

csv_data = LocalCSVAdapter(
    {
        "snapshots": "exports/customer_health_snapshots.csv",
        "outcomes": "exports/churn_events.csv",
    }
).load()

parquet_data = LocalParquetAdapter(
    {
        "snapshots": "exports/customer_health_snapshots.parquet",
        "outcomes": "exports/churn_events.parquet",
    }
).load()
```

CSV works with the base install. Parquet uses pandas' Parquet engine, so install
`proxyscore[parquet]` or another pandas-supported engine when reading Parquet files.

Database extraction should stay in application code and optional dependencies:

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg://...")
snapshots = pd.read_sql_query(snapshot_sql, engine, params={"as_of": "2026-04-01"})
outcomes = pd.read_sql_query(outcome_sql, engine, params={"as_of": "2026-04-01"})
tables = {"snapshots": snapshots, "outcomes": outcomes}
```

## Customer health

Use `customer_health_recipe()` for account health scores validated against churn events.

Expected snapshot columns:

- `account_id`
- `snapshot_at`
- `snapshot_month`
- `plan_tier`
- `logins_30d`
- `feature_depth`
- `support_tickets_30d`
- `nps`
- `payment_delay_days`
- `health_score`
- optional `updated_at`

Expected outcome columns:

- `account_id`
- `churned_at`
- `churned`
- optional `updated_at`

```python
from proxyscore import ProxyAudit, customer_health_recipe

recipe = customer_health_recipe()
prepared = recipe.prepare(csv_data, as_of="2026-04-01")

report = ProxyAudit(**prepared.audit_inputs()).run()
print(prepared.summary())
print(report.verdict)
```

The default outcome horizon is `90d`. Rows whose 90-day outcome window has not closed at `as_of`
remain in `prepared.data` with `outcome_status == "censored"` and are excluded from
`prepared.audit_inputs()` by default.

## Lead quality

Use `lead_quality_recipe()` for lead scores validated against later conversion or opportunity
creation.

Expected snapshot columns:

- `lead_id`
- `scored_at`
- `snapshot_week`
- `lead_source`
- `firmographic_fit`
- `engagement_score`
- `source_quality`
- `sales_activity_14d`
- `lead_score`
- optional `updated_at`

Expected outcome columns:

- `lead_id`
- `converted_at`
- `converted`
- optional `updated_at`

The default horizon is `30d`, which keeps repeated lead snapshots honest: each snapshot receives
only conversions that happen after that score and within its own maturity window.

## Account risk

Use `account_risk_recipe()` for account or portfolio risk scores validated against default,
serious delinquency, or another hard risk event.

Expected snapshot columns:

- `account_id`
- `snapshot_at`
- `snapshot_month`
- `portfolio`
- `utilization_rate`
- `days_past_due`
- `collateral_ratio`
- `covenant_breaches`
- `risk_score`
- optional `updated_at`

Expected outcome columns:

- `account_id`
- `defaulted_at`
- `defaulted`
- optional `updated_at`

The default horizon is `180d`. Higher risk scores should normally predict higher event rates;
the downstream check still reports observed polarity instead of assuming the naming convention is
correct.

## Point-in-time SQL pattern

Every recipe exposes a `sql_example` string showing the intended extraction shape:

```python
from proxyscore import get_business_recipe

print(get_business_recipe("customer_health").sql_example)
```

The SQL examples use four rules:

1. Select score snapshots at or before the explicit `:as_of` cutoff.
2. Keep one row per entity and snapshot time with `row_number() over (...)` ordered by
   `updated_at desc`.
3. Select only outcome events observed at or before `:as_of`.
4. Load snapshots and outcomes separately, then let `align_delayed_outcomes` assign matched,
   unmatched, and censored labels.

This keeps the score window before the outcome window and avoids sample leakage from future
events.

## Deduplication

Recipes deduplicate locally before alignment:

- Snapshots are keyed by entity and snapshot timestamp.
- Outcomes are keyed by entity and outcome timestamp.
- If an `updated_at` column is present, the latest version wins.
- Otherwise, the last source row wins.

The number of dropped rows is reported in `RecipeResult.deduplicated_rows` and
`RecipeResult.summary()`.

## Data minimization and credentials

Only load columns needed for the audit: entity ID, snapshot time, score, indicators, optional
segment/period, outcome time, outcome value, and optional update timestamps. Do not load free-text
notes, emails, phone numbers, addresses, raw CRM activity bodies, or credentials.

Credential handling belongs outside this package:

- Prefer environment variables, managed identity, or your organization's secret manager.
- Do not put passwords, tokens, connection strings, or private keys in recipe files,
  notebooks, Markdown reports, or governance metadata.
- Store local exports in access-controlled locations and delete temporary raw extracts after
  the audit artifact has been produced.
- Keep provenance metadata descriptive enough for reproducibility, but avoid embedding secrets
  in source paths or metadata values.
