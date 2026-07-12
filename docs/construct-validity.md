# Multi-construct validity

`assess_construct_validity` provides exploratory convergent and discriminant validity diagnostics
for two or more named reflective constructs. It reports average variance extracted (AVE) for each
construct and the absolute-correlation heterotrait-monotrait ratio (HTMT+) for each construct
pair, with row-bootstrap percentile confidence intervals and raw-correlation polarity safeguards.

This is useful when a business scorecard claims to measure several related but distinct concepts,
such as customer trust, perceived value, and product engagement. It is not appropriate for
formative composites whose indicators define, rather than reflect, the construct.

## Basic use

```python
from proxyscore import assess_construct_validity

assessment = assess_construct_validity(
    survey,
    {
        "trust": ["reliable", "transparent", "keeps_promises"],
        "value": ["worth_price", "useful", "good_investment"],
    },
    n_bootstrap=1000,
    random_state=42,
)

print(assessment.ave)
print(assessment.htmt)
print(assessment.polarity)
print(assessment.loadings)
print(assessment.to_markdown())
```

All selected indicators must be numeric, finite where observed, uniquely assigned to one
construct, and variable in the analysis sample. Each construct needs at least two indicators.
The default minimum is 100 complete rows; change `min_sample_size` only with a substantive reason.

## What is calculated

The analysis first selects all named indicators and removes any row missing one of them. AVE and
HTMT therefore use exactly the same sample, which makes construct-pair comparisons auditable. The
result records input, complete, and dropped row counts and warns when rows were excluded.

For each construct, the function extracts the leading component from its indicator correlation
matrix. The standardized loading for indicator `i` is the leading eigenvector coefficient times
the square root of the leading eigenvalue. The exploratory AVE estimate is:

```text
AVE = mean(standardized loading squared)
```

For constructs A and B, HTMT+ uses absolute correlations:

```text
HTMT+(A, B) = mean(absolute cross-construct correlations)
              / sqrt(mean(absolute within-A correlations)
                     * mean(absolute within-B correlations))
```

Absolute correlations prevent positive and negative relationships from cancelling in the ratio,
but they must not hide incorrectly oriented indicators. The `polarity` table therefore reports
every raw within-construct indicator correlation. By default, any correlation below `0` marks the
construct unresolved. Set `min_within_correlation` to a different documented floor when theory
justifies it.

The AVE and HTMT+ tables preserve the estimate-only threshold result and a `polarity_aligned`
field. Their headline `meets_threshold` / `below_threshold` flags are favorable only when both the
numeric threshold and polarity safeguard pass. An apparently low HTMT+ can therefore never
authorize a discriminant-validity claim while a participating construct contains a negatively
correlated indicator pair.

The default display flags are AVE at least `0.50` and HTMT below `0.85`. These are conventional
screening thresholds, not universal laws or an overall verdict. Inspect estimates, intervals,
loadings, theory, data quality, and downstream evidence together. A bootstrap interval that
crosses a threshold is uncertain evidence even when the point estimate is on the preferred side.

Set `n_bootstrap=0` for descriptive-only output. Confidence limits are then missing and the result
contains an explicit warning. Bootstrap resampling is deterministic when `random_state` is fixed.

## Sample safeguards

- The API rejects fewer than two constructs, fewer than two indicators per construct, duplicate
  assignments, missing columns, nonnumeric or infinite inputs, constant indicators, duplicate row
  indexes, invalid thresholds, and an undersized complete sample.
- Missingness is handled listwise across every selected indicator and is reported. Impute only in
  a documented preprocessing pipeline; do not let this diagnostic silently invent values.
- Two-indicator constructs are accepted but warned because their measurement models have limited
  identification and their estimates are less stable.
- HTMT+ is reported as unassessed when either construct has zero mean within-construct association.
- Raw within-construct correlations below `min_within_correlation` are named in warnings, and
  favorable AVE/HTMT+ flags involving those constructs are withheld until item orientation or the
  construct definition is resolved.
- Every table retains construct names, sample information, interval bounds, and valid bootstrap
  counts. Results are not averaged into a single pass/fail score.

## When to use SEM/CFA instead

The AVE here is based on a one-factor correlation/PCA extraction. It is a practical screening
estimate, not a confirmatory factor analysis and not a fitted structural equation model. Use a
specialized package such as `semopy` or R's `lavaan` when decisions depend on any of the following:

- global or comparative model-fit statistics;
- standard errors, hypothesis tests, or latent-variable confidence intervals;
- ordinal indicators and polychoric correlations;
- cross-loadings, correlated measurement errors, method factors, or hierarchical factors;
- formal tests comparing alternative measurement models;
- latent regressions, mediation, or other structural paths;
- measurement invariance that needs ordinal or robust estimators, partial invariance,
  longitudinal dependence, or a model outside the built-in continuous simple-structure ladder.

For consequential survey instruments, the usual workflow is to specify the measurement model
from theory, fit and diagnose it in SEM/CFA, evaluate invariance where relevant, and use these
lightweight metrics only as reproducible screening and monitoring companions.

For the supported continuous simple-structure case, see
[Measurement invariance across segments](measurement-invariance.md).
