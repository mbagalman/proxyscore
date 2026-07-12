# Multi-outcome validation

A construct often has more than one observable consequence. Customer health may precede churn,
expansion, escalation, and payment risk, but those outcomes can have different types, horizons,
availability, and business importance. `validate_outcomes` evaluates them separately instead of
forming one complete-case sample or averaging unlike metrics.

## Configure named outcomes

Each `OutcomeSpec` declares the outcome values and their interpretation:

```python
from proxyscore import OutcomeSpec, validate_outcomes

report = validate_outcomes(
    score=df["health_score"],
    indicators=df[indicator_columns],
    outcomes={
        "churn": OutcomeSpec(
            values=df["churned_in_90d"],
            outcome_type="binary",
            polarity="negative",
            window="90 days after score",
            importance="required",
            mature=df["churn_window_closed"],
        ),
        "expansion": OutcomeSpec(
            values=df["expansion_revenue_180d"],
            outcome_type="continuous",
            polarity="positive",
            window="180 days after score",
            importance="supporting",
            mature=df["expansion_window_closed"],
        ),
    },
)

print(report.verdict)
print(report.summary())
print(report.to_markdown())
```

`outcome_type` is `binary` or `continuous`. Continuous outcomes must be numeric. `polarity`
describes the expected raw relationship: `positive` means a higher score should predict more of
the outcome, `negative` means it should predict less, and `auto` records the detected direction
without testing it against an expectation. `window` is a required human-readable scope label.

At least one outcome must be `required`. Supporting outcomes add evidence but cannot substitute
for a required outcome.

## Samples and maturity

The `mature` boolean mask says whether each row's outcome window has closed. An immature row is
excluded only from that outcome. Missing values are also counted per outcome. The summary reports:

- input, mature, and immature rows;
- observed and missing mature outcomes plus mature-sample missingness;
- complete score/outcome evaluation rows;
- expected and detected polarity;
- separate downstream and leakage statuses.

No intersection is taken across outcomes. For example, a missing 180-day expansion result does
not remove that entity from a mature 30-day escalation assessment.

Point-in-time alignment output can carry its censoring state directly into a specification:

```python
from proxyscore import OutcomeSpec, align_delayed_outcomes

aligned = align_delayed_outcomes(
    observations,
    churn_events,
    entity="account_id",
    score_time="scored_at",
    outcome="churned",
    outcome_time="churned_at",
    horizon="90d",
    as_of="2026-07-01",
)
churn = OutcomeSpec.from_alignment(
    aligned,
    outcome_type="binary",
    polarity="negative",
    window="90 days after score",
)
```

Rows marked `censored` by `align_delayed_outcomes` become immature. Matched and mature unmatched
rows remain eligible.

## Overall verdict policy

The overall verdict is a named gate, never an average:

| Evidence | Overall result |
| --- | --- |
| Any required outcome fails downstream or leakage, or contradicts configured polarity | `not_validated` |
| Any required outcome cannot be assessed | `not_validated` |
| Required evidence has warnings | `directional` |
| A supporting outcome fails or contradicts configured polarity | `directional` with the outcome named |
| Every required outcome passes strongly and supporting evidence introduces no conflict | `decision_grade` |

An immature or otherwise unassessable supporting outcome remains visible but does not by itself
downgrade the verdict. This distinction lets teams state what evidence is mandatory while keeping
early or secondary evidence honest.

`report.tables()` returns the outcome summary plus namespaced downstream lift and leakage tables,
such as `churn.downstream` and `churn.leakage`, for report templates.

## Compare score versions

`compare_outcomes` applies the existing paired score-comparison workflow independently to each
outcome's mature rows:

```python
from proxyscore import compare_outcomes

comparison = compare_outcomes(
    baseline_score=df["health_v1"],
    candidate_score=df["health_v2"],
    outcomes={"churn": churn, "expansion": expansion},
    n_bootstrap=500,
)

print(comparison.outcome_summary)
print(comparison["churn"].performance)
```

The cross-outcome summary preserves each outcome's type, window, importance, maturity counts,
evaluation size, metric, delta, and assessment. `comparison.tables()` and
`comparison.to_markdown()` expose namespaced details from every `ScoreComparison`; score versions
are still paired within an outcome, never across outcomes.

## Interpretation limits

Multi-outcome validation organizes criterion evidence; it does not prove causality or make unlike
outcomes interchangeable. Outcome definitions, point-in-time correctness, business costs, segment
behavior, calibration, and permitted use still need explicit review. A favorable verdict applies
only to the declared outcomes, populations, and windows.
