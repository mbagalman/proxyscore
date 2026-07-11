# Operating-threshold and action analysis

A score can rank records well while still performing poorly at the cutoff a business intends to
use. `analyze_actions` evaluates concrete action policies after downstream validation, exposing
their workload, errors, segment behavior, and optional economic value.

This analysis does not change a `ProxyAudit` verdict. It supplies additional evidence for a
specific operating policy and sample.

## Binary outcomes

```python
from proxyscore import analyze_actions

analysis = analyze_actions(
    score=df["health_score"],
    outcome=df["churned_next_quarter"],
    cutoffs=[25, 35, 45],
    percentiles=[5, 10, 20],
    top_n=[50, 100, 250],
    segments=df["plan_tier"],
)

print(analysis.polarity)
print(analysis.table)
print(analysis.segment_table)
```

For each policy, the overall table reports selected count and rate, confusion-matrix counts,
precision, recall, specificity, false-positive and false-negative rates, accuracy, F1, and
Youden's J statistic.

Automatic polarity detects whether more of the outcome occurs at high or low scores. With a
health score predicting churn, polarity is usually `-1`, so an explicit cutoff of `35` selects
scores at or below 35. The result's notes state the resolved direction.

Binary labels follow the same convention as downstream validation: the larger or later-sorting
of two values is the positive class. Encoding the event as `1` remains the safest choice.

## Policy types

Four policy types can be evaluated together:

- `cutoffs`: raw score cutoffs. Ties are included.
- `percentiles`: percentage of the usable population selected after orienting the score. A value
  of `10` means the highest-priority 10 percent. Ties at the cutoff are included, so realized
  selection may be slightly larger.
- `top_n`: an exact action capacity. A value of `100` selects exactly 100 records. Boundary ties
  are resolved by original row order.
- `grid_size`: number of empirical candidate cutoffs generated across the score distribution.

When none is supplied, a 20-point grid is generated. Every row records its strategy, original
parameter, resolved raw score cutoff, and stable policy ID.

## Business value

For binary outcomes, optional economics convert classification counts into sample-level value:

```python
valued = analyze_actions(
    score=df["health_score"],
    outcome=df["churned_next_quarter"],
    top_n=[50, 100, 250, 500],
    true_positive_benefit=500,
    false_positive_cost=40,
    false_negative_cost=300,
    action_cost=15,
)
```

Expected value is calculated as:

```text
TP * true-positive benefit
- FP * false-positive cost
- FN * false-negative cost
- selected records * action cost
```

The table also reports value per record, value relative to taking no action, and the
true-positive benefit required to break even. If at least one economic input is supplied,
unspecified inputs are recorded explicitly as zero in `analysis.assumptions`.

All monetary values must use one consistent unit and time horizon. The library does not infer
currency, intervention effectiveness, downstream side effects, or whether a true positive would
actually respond to the action.

## Explicit candidate recommendations

No policy is recommended automatically. Call `recommend` with an objective and optional
constraints when you want the library to select among the policies you deliberately evaluated:

```python
candidate = valued.recommend(
    "expected_value",
    max_actions=250,
    min_recall=0.40,
)

print(candidate.statement)
```

Binary objectives are `expected_value`, `f1`, `precision`, `recall`, and `youden_j`.
`expected_value` requires explicit economic assumptions. The returned recommendation records
the objective, constraints, assumptions, sample size, and a prospective-validation warning. It
is a candidate operating rule, not production authorization.

## Segment safeguards

Supplying `segments` evaluates each global policy inside each segment. The policy itself is not
retuned per segment. Segment rows are marked unassessed when they have fewer than
`Thresholds.min_segment_size` usable records or, for binary outcomes, fewer than
`Thresholds.min_class_count` records in either class.

Unassessed segments remain in `segment_table` with their sample size, selected count, selected
rate, and reason. This prevents a favorable aggregate result from silently claiming coverage in
a sparse or one-class segment.

## Continuous outcomes

For continuous outcomes, action analysis reports the overall outcome mean, selected and
unselected outcome means, and their difference:

```python
continuous = analyze_actions(
    score=df["expansion_score"],
    outcome=df["expansion_revenue_next_quarter"],
    top_n=[25, 50, 100],
    segments=df["region"],
)

print(continuous.table)
```

Business-value inputs and automatic recommendations are intentionally unavailable for
continuous outcomes. Translating an outcome-unit difference into decision value requires a
domain-specific intervention and cost model that the library cannot safely assume.

## Recommended workflow

1. Use [time-window alignment](time-window-alignment.md) to create point-in-time-correct score
   and outcome rows.
2. Run `ProxyAudit` to establish association, stability, leakage, and segment evidence.
3. Use `analyze_actions` on a prospective or untouched evaluation sample.
4. Choose an objective and constraints with the decision owner, then request a candidate.
5. Validate the chosen policy forward in time before production use and monitor realized costs,
   workload, and outcomes afterward.
