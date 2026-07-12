# Comparing score versions

`compare_scores` evaluates a baseline and candidate score on the same entities and outcomes. It
keeps coverage differences visible, quantifies downstream-performance uncertainty with a paired
bootstrap, and exposes rank, band, segment, stability, and action-policy changes without reducing
all tradeoffs to one number.

```python
from proxyscore import compare_scores

comparison = compare_scores(
    baseline_score=evaluation["health_score_v1"],
    candidate_score=evaluation["health_score_v2"],
    outcome=evaluation["churned_next_quarter"],
    segments=evaluation["plan_tier"],
    period=evaluation["snapshot_month"],
    baseline_name="v1",
    candidate_name="v2",
    action_top_n=[50, 100, 250],
)

print(comparison.performance)
print(comparison.dimensions)
print(comparison.actions)
```

## Paired entity coverage

Pandas Series may have different entity indexes. The comparison reports:

- rows supplied for each version;
- overlapping entities;
- entities present in only one version;
- missing baseline, candidate, or outcome values in the overlap;
- complete paired rows retained for evaluation.

Every performance, lift, distribution, rank, migration, segment, stability, and action table is
then calculated from that same complete paired entity/outcome sample. Optional missing segment or
period labels affect only their corresponding tables.

When either score is a Series, both scores and the outcome must be Series. Optional segment and
period inputs must also be Series. This prevents an unindexed array from being silently attached
to the wrong entity after indexes differ. Plain array-like inputs are accepted when all required
inputs have the same length.

```python
print(comparison.coverage.summary())
```

## Downstream performance and uncertainty

For binary outcomes, the compared metric is oriented AUC. For continuous outcomes, it is
polarity-oriented Spearman correlation. Each version detects polarity independently, so a
candidate that reverses the score scale is not mistaken for a regression.

The performance table reports baseline value, candidate value, candidate-minus-baseline delta,
confidence interval, valid bootstrap count, method, and assessment. The default is 500 paired
bootstrap samples with a 95 percent interval:

```python
comparison = compare_scores(
    old_score,
    new_score,
    delayed_outcome,
    n_bootstrap=2000,
    confidence_level=0.95,
    random_state=42,
)
```

The candidate is labeled `improved` only when the entire interval is above zero and `regressed`
only when it is below zero. Otherwise the result is `inconclusive`. Set `n_bootstrap=0` for a
descriptive-only comparison; it is always labeled inconclusive regardless of the point estimate.

## Report-ready evidence

`ScoreComparison` exposes these DataFrames:

- `performance`: paired downstream metric and confidence interval.
- `dimensions`: concise improved, regressed, or inconclusive judgments with their stated basis.
- `distributions`: paired-sample count, overlap missingness, mean, spread, range, and quantiles.
- `lift`: version-specific lift/capture or continuous-outcome band tables.
- `rank_movements`: entity-level scores, oriented rank percentiles, absolute rank movement, and
  bands.
- `migration`: baseline-to-candidate band migration counts and rates.
- `stability`: PSI over time for each version when `period` is supplied.
- `segments`: score and validity summaries for each version when `segments` is supplied.
- `actions`: changed action assignments when cutoffs, percentiles, or capacities are supplied.

`comparison.tables()` returns all available tables by section name for HTML/template systems.
`comparison.to_markdown()` produces a portable review document and transparently limits long
tables:

```python
markdown = comparison.to_markdown(max_rows=50)
```

## Rank and band migration

Raw score correlation can be misleading when the scale changes. The comparison therefore
reports raw Pearson and Spearman correlation plus polarity-oriented Spearman correlation.
Entity-level rank movement uses oriented percentile ranks, and band migration assigns band 1 to
the highest-priority records for each version. Bands use score-value quantile boundaries: tied
scores always remain together, so migration is invariant to row order and fewer than the
requested number of bands may be present.

Cross-version PSI is also reported, but it is scale-dependent. Treat it as descriptive if the
candidate changes score units or range.

## Action-assignment changes

Supply the same kinds of policies accepted by `analyze_actions`:

```python
comparison = compare_scores(
    old_score,
    new_score,
    delayed_outcome,
    action_cutoffs=[25, 35, 45],
    action_percentiles=[5, 10, 20],
    action_top_n=[50, 100, 250],
)
```

For each policy, the action table reports version-specific cutoffs and selected counts, records
selected by both versions, records selected by only one version, total changed assignments,
change rate, Jaccard overlap, and available outcome metrics. Top-N policies are especially useful
when the business has a fixed review capacity.

Raw cutoffs should only be compared when both score versions use meaningfully comparable units.
Percentile and top-N policies remain interpretable across scale changes.

## Interpreting dimensions

Only downstream performance receives a paired inferential assessment. Missingness, maximum
period PSI, and weakest-segment validity are labeled improved or regressed descriptively, with
the direction and basis recorded in the dimensions table. A descriptive label is evidence about
this evaluation sample, not proof that the change will persist.

Review the full tables when a candidate improves one dimension and regresses another. The API
deliberately does not calculate a single opaque winner score or authorize deployment.
