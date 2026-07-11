# Time-window and delayed-outcome alignment

Predictive validation is only honest when every outcome is observed after the data used to
calculate its score. `align_delayed_outcomes` turns score snapshots and timestamped outcomes
into an audit-ready table while enforcing that ordering.

## Matching contract

Each observation defines an outcome interval with an exclusive start and inclusive end:

```text
(outcome window start, outcome window end]
```

The start defaults to the score timestamp or observation-window end. Define the end with either
a fixed `horizon` such as `"90d"` or an explicit column in the observations. An explicit outcome
window may start later than the feature window ends, but it may never start earlier.

Every input observation remains in the result and receives one status:

- `matched`: an eligible outcome was selected.
- `unmatched`: the outcome window is complete, but no eligible outcome was found.
- `censored`: the outcome window has not completed at `as_of`.

Matched rows retain the selected outcome timestamp and lag from the feature-window end.
Unmatched rows receive `no_outcome_value`, which defaults to `0` for event-table workflows.
Censored rows always receive a missing outcome and are excluded by `audit_inputs()` unless
`include_censored=True` is requested.

Supply `as_of` in production so the data cutoff is explicit and reproducible. If omitted, the
function uses the latest valid outcome timestamp, or the latest outcome-window end when there
are no valid outcome timestamps.

## Example: churn or renewal event stream

This shape records only events. A mature account without a churn event is therefore a
non-event (`0`).

```python
import pandas as pd

from proxyscore import ProxyAudit, align_delayed_outcomes

snapshots = pd.DataFrame(
    {
        "account_id": ["a-1", "a-2", "a-3"],
        "snapshot_at": pd.to_datetime(["2026-01-01"] * 3),
        "logins": [30, 4, 18],
        "support_tickets": [1, 8, 2],
        "health_score": [82.0, 24.0, 67.0],
    }
)
churn_events = pd.DataFrame(
    {
        "account_id": ["a-2"],
        "churned_at": pd.to_datetime(["2026-02-12"]),
        "churned": [1],
    }
)

aligned = align_delayed_outcomes(
    snapshots,
    churn_events,
    entity="account_id",
    score_time="snapshot_at",
    outcome="churned",
    outcome_time="churned_at",
    horizon="90d",
    as_of="2026-04-15",
    no_outcome_value=0,
)

report = ProxyAudit(
    **aligned.audit_inputs(
        ["logins", "support_tickets"],
        score_column="health_score",
    )
).run()
print(aligned.diagnostics.summary())
print(report.verdict)
```

For renewal events, the same event-table pattern can use `no_outcome_value=0` when no renewal
by the end of a mature window means "did not renew." If the absence of a row has a different
meaning, use an explicit labeled outcome table or set `no_outcome_value=pd.NA`.

## Example: lead conversion with repeated snapshots

Repeated snapshots for the same entity are supported. Each snapshot gets its own outcome
window, so downstream validation can use one row per lead-period.

```python
import pandas as pd

from proxyscore import align_delayed_outcomes

lead_snapshots = pd.DataFrame(
    {
        "lead_id": [101, 101, 202],
        "scored_at": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-02-01"]),
        "fit": [0.7, 0.8, 0.4],
        "intent": [0.5, 0.9, 0.3],
        "lead_score": [61.0, 88.0, 35.0],
    }
)
conversions = pd.DataFrame(
    {
        "lead_id": [101],
        "converted_at": pd.to_datetime(["2026-02-10"]),
        "converted": [1],
    }
)

aligned_leads = align_delayed_outcomes(
    lead_snapshots,
    conversions,
    entity="lead_id",
    score_time="scored_at",
    outcome="converted",
    outcome_time="converted_at",
    horizon="30d",
    as_of="2026-03-15",
    match="first",
)
```

The February conversion is outside the January snapshot's 30-day interval but inside the
February snapshot's interval. The two snapshots therefore receive different labels without a
manual join.

## Multiple outcomes and boundaries

If multiple outcomes fall inside one interval, `match="first"` selects the earliest and
`match="last"` selects the latest. Timestamp ties preserve source-row order. Use
`match="error"` when multiple matches indicate a data-quality problem. Diagnostics always
report the number of affected observations and extra candidate rows.

An event exactly at the window start is excluded because it is not strictly later than the
score window. An event exactly at the window end is included. Events after `as_of` are ignored
and counted as future outcome rows.

## Missing values and timezones

- Observation entity IDs and required observation timestamps may not be missing.
- Outcome rows with a missing entity, timestamp, or value are excluded and counted in
  `invalid_outcome_rows`.
- All naive timestamps are accepted together.
- All timezone-aware timestamps are accepted together and normalized to UTC, even when their
  source timezone names differ.
- Mixing timezone-aware and timezone-naive values is rejected, including a mismatched `as_of`.

The result uses a fresh unique index. `AlignmentResult.audit_inputs()` applies that same index
to indicators, score, outcome, segments, and period columns, satisfying `ProxyAudit`'s strict
alignment contract without manual reindexing.
