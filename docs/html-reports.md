# Standalone HTML audit reports

`AuditReport.to_html()` produces one self-contained HTML document for reviewers who do not have
a Python environment. `AuditReport.write_html()` writes the same document to disk and returns
the resolved output path.

```python
report = ProxyAudit(
    indicators=df[indicator_columns],
    score=df["health_score"],
    outcome=df["churned_next_quarter"],
    segments=df["plan_tier"],
    period=df["snapshot_month"],
).run()

output = report.write_html(
    "customer-health-audit.html",
    title="Customer health score review",
    metadata={
        "organization": "Example Company",
        "project": "Retention prioritization",
        "review_owner": "Revenue Analytics",
    },
)

print(output)
```

The file contains inline CSS and no external scripts, fonts, images, or network dependencies. It
can be opened locally, attached to a review, or retained as an audit artifact.

## Report contents

The report includes:

- the verdict, reason, and a concise statement of its limitations;
- audit input scope, including row counts, indicator columns, and supplied optional inputs;
- generation time, package version, and caller-supplied project metadata;
- a summary table with visible status text;
- every check's status, explanation, metrics, detail table, and notes;
- attached operating-threshold analysis when supplied.

Statuses always include the words `PASS`, `WARN`, `FAIL`, or `SKIP`; color is only a secondary
signal. The document uses semantic headings, sections, captions, and tables, and its layout
adapts to narrow screens.

## Attaching action analysis

BR-002 output can be attached before rendering:

```python
actions = analyze_actions(
    score=df["health_score"],
    outcome=df["churned_next_quarter"],
    top_n=[50, 100, 250],
    segments=df["plan_tier"],
    true_positive_benefit=500,
    false_positive_cost=40,
    action_cost=15,
)

report.attach_action_analysis(actions)
report.write_html("customer-health-audit-with-actions.html")
```

The HTML then includes evaluated policy and segment tables, polarity, economic assumptions, and
action-analysis notes. Attaching analysis does not change the audit results or verdict.

## Large tables

Detail and action tables show at most 100 rows by default. Truncated tables state exactly how
many rows are shown and how many exist:

```python
report.write_html(
    "customer-health-audit.html",
    max_detail_rows=250,
)
```

Set `max_detail_rows=None` to include all rows. This can create a large document, especially when
many action policies and segments are evaluated.

## Escaping and trusted metadata

Report titles, metadata, check text, table labels and values, and notes are HTML-escaped. Values
such as `<script>` are displayed as text and never interpreted as markup. The renderer does not
include JavaScript.

Caller metadata is descriptive only. It is displayed in the report but does not alter thresholds,
metrics, statuses, or verdicts.

## Deterministic output

Generation time defaults to the current UTC time. Tests or reproducible build pipelines can
supply a fixed timestamp:

```python
from datetime import datetime, timezone

document = report.to_html(
    generated_at=datetime(2026, 7, 11, 15, 30, tzinfo=timezone.utc),
)
```

With the same report, options, package version, and generation timestamp, the resulting HTML is
deterministic.
