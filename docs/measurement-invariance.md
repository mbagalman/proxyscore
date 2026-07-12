# Measurement invariance across segments

`assess_measurement_invariance` tests whether named reflective constructs have a comparable
measurement model across supplied segments. It fits four nested multigroup confirmatory factor
analysis (CFA) models and reports each level separately:

| Level | Equality constraints | What support permits |
|---|---|---|
| Configural | Same indicator-to-construct pattern | The declared factor structure is plausible in every segment. |
| Metric | Configural plus equal factor loadings | Compare latent covariances and relationships across segments. |
| Scalar | Metric plus equal indicator intercepts | Compare latent means across segments. |
| Strict | Scalar plus equal residual variances | Treat indicator-specific measurement error as equal too. |

The ladder is prerequisite-gated. If metric invariance fails, scalar and strict models are still
reported, but neither can be marked supported. In particular, a later model's favorable change
statistics never revive a failed prerequisite.

## Quick start

```python
from proxyscore import assess_measurement_invariance

invariance = assess_measurement_invariance(
    survey,
    segments=survey["customer_tier"],
    constructs={
        "trust": ["reliable", "transparent", "keeps_promises"],
        "value": ["worth_price", "useful", "good_investment"],
    },
)

print(invariance.levels)
print(invariance.highest_supported_level)
print(invariance.parameters)
print(invariance.to_markdown())
```

Rows in `levels` include convergence, chi-square, degrees of freedom, CFI, RMSEA, SRMR,
successive chi-square differences and p-values, fit-index changes, prerequisite state, support,
and a plain-language interpretation. `parameters` retains the loading, intercept, residual
variance, and latent mean for every indicator, segment, and level. `group_sizes` records all
supplied segments and whether each met the sample floor.

## Decision rules

The configural model defaults to CFI at least `0.90`, RMSEA at most `0.08`, and SRMR at most
`0.08`. Each later level defaults to no more than a `0.01` CFI decrease or `0.015` RMSEA increase.
The allowed SRMR increase is `0.030` for metric and `0.010` for scalar and strict. All thresholds
are configurable.

These are screening rules, not universal laws. Chi-square differences are reported for
transparency but are not the sole gate because they are sensitive to sample size. The change
limits follow Fang Fang Chen's simulation study, whose results also show that sample size,
group balance, and the pattern of noninvariance affect sensitivity. Read estimates, absolute
fit, change statistics, theory, and practical consequences together.

`highest_supported_level` returns only the last consecutively supported stage. It is `None` when
configural fit is unsupported or the assessment is unavailable. Do not use metric support to
justify latent-mean comparisons; that requires scalar support.

## Model and sample safeguards

The implementation is intentionally narrow and auditable:

- maximum-likelihood multigroup CFA for continuous, approximately multivariate-normal data;
- simple structure: every indicator belongs to exactly one named construct;
- the first indicator listed for each construct is its marker, with loading fixed to one;
- freely correlated latent factors and group-specific factor covariance matrices;
- one shared listwise-complete sample across every level;
- at least two constructs and two indicators per construct, matching the construct-validity API;
- at least `min_group_size=100` complete rows in every supplied segment by default;
- varying indicators and a nonsingular covariance matrix within every segment.

If any supplied segment is below the sample floor, no segment is silently discarded. The result
contains all four levels as unassessed, names the sparse segment, and makes no comparability claim.
Missing rows and optimizer nonconvergence are also explicit.

## Published reference validation

The regression suite reconstructs the group means and covariance matrices from the classic
Holzinger-Swineford two-school example used in lavaan documentation. It reproduces the published
configural, metric, and scalar chi-square results and degrees of freedom within numerical
tolerance: metric invariance is supported, while scalar invariance is not. This is the same
substantive conclusion shown in lavaan's example, where direct latent-mean comparison is called
unwise.

References:

- [lavaan multigroup measurement-invariance example](https://lavaan.ugent.be/tutorial/groups.html)
- [Chen (2007), *Sensitivity of Goodness of Fit Indexes to Lack of Measurement Invariance*](https://doi.org/10.1080/10705510701301834)
- [Putnick and Bornstein (2016), *Measurement Invariance Conventions and Reporting*](https://pmc.ncbi.nlm.nih.gov/articles/PMC5145197/)

## When to use specialized SEM software

Use an established package such as R's `lavaan` when the analysis needs ordinal indicators and
threshold invariance, robust or weighted estimators, full-information missing-data estimation,
standard errors or parameter confidence intervals, survey weights, complex sampling,
cross-loadings, correlated errors, longitudinal dependence, partial invariance, modification
indices, or equality constraints beyond this fixed ladder.

Partial invariance is deliberately not automated here. Choosing constraints after inspecting the
same data is a model-development decision that needs theory, transparent reporting, and usually
independent validation. This API provides a reproducible first full-invariance test, not a license
to search until a desired comparison becomes available.
