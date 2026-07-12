# proxyscore

**Construct, validate, and monitor proxy scores for latent business constructs.**

*For when the business needs a number for something it cannot directly observe.*

[![CI](https://github.com/mbagalman/proxyscore/actions/workflows/ci.yml/badge.svg)](https://github.com/mbagalman/proxyscore/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/proxyscore.svg)](https://pypi.org/project/proxyscore/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Customer health, lead quality, engagement, account risk, brand strength, demand — businesses
constantly need numbers for things nobody can measure directly. Analysts answer with proxy
scores: weighted composites of observable indicators. What's usually missing is any rigorous
answer to the question that follows: **is this score real, or are we just compounding noise?**

`proxyscore` is a validation toolkit for exactly that question. It audits a proxy score for:

| Check | Question it answers |
| --- | --- |
| **indicators** | Are the inputs healthy? Missingness, dead columns, weak items, reliability (Cronbach's alpha), redundancy (correlation pairs, VIF), single-indicator dominance. |
| **stability** | Does the score distribution hold still over time? Population Stability Index per period with the standard 0.10 / 0.25 bands. |
| **downstream** | Does the score predict a *delayed hard outcome* it should predict (renewal, churn, conversion)? AUC / rank correlation, lift and capture by score band, automatic polarity detection. |
| **leakage** | Is the score secretly built from the outcome? Flags indicators with implausibly strong standalone association with the outcome, and leak-suggestive column names. |
| **segments** | Does the score mean the same thing for every segment? Score-level gaps (standardized mean difference) and per-segment validity divergence. |

The audit ends in a verdict:

- **`decision_grade`** — strong downstream signal, no failures or warnings, and every supplied
  check assessable: a strong *candidate* for per-record decisions (prioritization, alerting,
  automation) within the validated scope. It confirms association strength and the absence of
  problems among the checks you supplied — it does not by itself establish calibration,
  operating-threshold performance, prospective validation, or governance clearance, so treat it
  as "cleared to evaluate for automation," not an automatic green light.
- **`directional`** — real but moderate signal, or unresolved warnings, or a supplied check that
  couldn't be assessed: usable for dashboards and trend reading, not for automated per-record
  action.
- **`not_validated`** — a failed check or no outcome validation: the score is an untested
  hypothesis, not a measurement.

## Install

```bash
pip install proxyscore
```

Requires Python ≥ 3.10. Depends only on `numpy`, `pandas`, and `scipy`.

The installed `proxyscore` command exposes `audit`, `compare`, `baseline`, and `monitor`
workflows. See the [CLI guide](docs/cli.md). Install `proxyscore[parquet]` when reading Parquet
files from the command line.

## Quick start

```python
from proxyscore import ProxyAudit

report = ProxyAudit(
    indicators=df[["logins", "feature_depth", "support_tickets", "nps", "payment_delay_days"]],
    score=df["health_score"],            # omit to have one built for you
    outcome=df["churned_next_quarter"],  # delayed hard outcome (observed AFTER the score window)
    segments=df["plan_tier"],            # optional
    period=df["month"],                  # optional
).run()

print(report.verdict)        # Verdict.DECISION_GRADE / DIRECTIONAL / NOT_VALIDATED
print(report.summary())      # one row per check: status + plain-language summary
print(report.to_markdown())  # full audit report, ready for a PR or wiki page
report.write_html("proxy-audit.html")  # portable report for business review

report["downstream"].metrics    # {'auc': ..., 'auc_oriented': ..., 'spearman': ..., ...}
report["segments"].details      # per-segment DataFrame
```

Evaluate concrete action cutoffs only after the outcome is aligned and the score is audited:

```python
from proxyscore import analyze_actions

actions = analyze_actions(
    df["health_score"],
    df["churned_next_quarter"],
    top_n=[50, 100, 250],
    false_positive_cost=40,
    action_cost=15,
)
print(actions.table)
```

See [Operating-threshold and action analysis](docs/action-analysis.md) for cutoff, percentile,
capacity, segment, economic-value, and explicit recommendation workflows.

Compare a proposed score version on the same entities and delayed outcomes:

```python
from proxyscore import compare_scores

comparison = compare_scores(
    df["health_score_v1"],
    df["health_score_v2"],
    df["churned_next_quarter"],
    segments=df["plan_tier"],
    action_top_n=[50, 100, 250],
)
print(comparison.performance)
print(comparison.migration)
```

See [Comparing score versions](docs/score-comparison.md) for paired uncertainty, coverage,
stability, segment, rank-migration, and changed-action analysis.

Validate the construct against several business consequences without mixing their samples:

```python
from proxyscore import OutcomeSpec, validate_outcomes

evidence = validate_outcomes(
    df["health_score"],
    df[indicator_columns],
    {
        "churn": OutcomeSpec(
            df["churned_90d"], "binary", polarity="negative", window="90d"
        ),
        "expansion": OutcomeSpec(
            df["expansion_180d"],
            "continuous",
            polarity="positive",
            window="180d",
            importance="supporting",
        ),
    },
)
print(evidence.summary())
```

See [Multi-outcome validation](docs/multi-outcome-validation.md) for maturity masks, required
versus supporting evidence, non-averaging verdicts, and multi-outcome score comparison.

Persist an approved baseline and monitor later batches without refitting:

```python
from proxyscore import create_monitoring_baseline, monitor_batch

baseline = create_monitoring_baseline(
    reference[indicator_columns],
    score_id="customer-health",
    score_version="2026.1",
    score=reference["health_score"],
    outcome=reference["churned_next_quarter"],
)
baseline.save("customer-health-baseline.json")

result = monitor_batch(
    baseline,
    current[indicator_columns],
    score=current["health_score"],
    batch_id="2026-07",
)
print(result.alert_state, result.exit_code)
```

See [Repeatable batch monitoring](docs/monitoring.md) for fitted-constructor artifacts, fixed
drift bins, matured outcomes, alert states, exit codes, and monthly result retention.

Attach governance and reproducibility metadata to audit and monitoring artifacts when a score needs
business approval:

```python
from proxyscore import GovernanceContext, ProxyAudit

governance = GovernanceContext(
    score_name="customer_health",
    score_version="2026.1",
    owner="Customer analytics",
    intended_uses=["retention prioritization"],
    prohibited_uses=["automatic account cancellation"],
    population="active B2B customers",
    data_window="2026-01-01/2026-03-31",
    outcome_window="2026-04-01/2026-06-30",
    decision_owner="VP Customer Success",
    reviewer="Risk committee",
    dataset_id="warehouse.snapshot.customer_health.2026q1",
    code_revision_id="abc1234",
)

report = ProxyAudit(
    indicators=df[indicator_columns],
    score=df["health_score"],
    outcome=df["churned_next_quarter"],
    governance=governance,
    governance_strict=True,
).run()
print(report.governance_manifest.configuration_fingerprint)
```

See [Governance and reproducibility manifests](docs/governance.md) for the versioned JSON schema,
strict-mode workflow, redaction behavior, and monitoring-artifact integration.

When a score must represent an event probability, fit and evaluate an explicit calibration
mapping on separate samples:

```python
from proxyscore import fit_and_assess_calibration

calibration = fit_and_assess_calibration(
    df["health_score"],
    df["churned_next_quarter"],
    method="logistic",  # or "isotonic"
)
print(calibration.metrics)
```

See [Probability calibration](docs/calibration.md) for explicit probability opt-in, reusable
mapping artifacts, held-out evaluation, curve binning, uncertainty, and sparse-data warnings.

No data handy? There's a synthetic example with a known latent construct, a plantable leak,
and plantable drift:

```python
from proxyscore.datasets import make_customer_health

df = make_customer_health(n=3000, include_leak=True)
```

> **New to the library?** The [Getting started guide](docs/getting-started.md) is a
> hands-on walkthrough that runs a full audit, reads the verdict, interprets every check,
> shows what failures look like, and covers bringing your own data — with real output throughout.

## Building a score

If you don't have a score yet, two pragmatic constructors are included, both with a
fit/transform API so weights learned on a development sample can be applied to later
periods without re-fitting (which keeps stability monitoring honest):

```python
from proxyscore import CompositeScore, PCAScore

# industry-scorecard style: normalize, weight, sum (negative weight = reverse-oriented)
score = CompositeScore(
    weights={"logins": 1, "support_tickets": -1, "payment_delay_days": -0.5},
    scaling="zscore",  # or "minmax", "rank"
).fit_transform(df[indicator_cols])

# data-driven: first principal component of the standardized indicators
score = PCAScore().fit_transform(df[indicator_cols])
```

Missing data never silently becomes a number: `CompositeScore` renormalizes partially
observed rows over the weights actually present and returns `NaN` once observed weight
falls below `min_coverage` (default 0.5); `PCAScore` returns `NaN` for any incomplete row.

All multi-input APIs enforce row alignment: a `Series` must carry the same index as the
indicators (same labels, same order), and plain arrays must match their length exactly.

## Using checks individually

Every check is also a standalone function returning a `CheckResult` with a status
(`pass` / `warn` / `fail` / `skip`), headline metrics, a details DataFrame, and
interpretation notes:

```python
from proxyscore import (
    check_indicators, check_stability, check_downstream, check_segments, check_leakage,
    psi, lift_table, cronbach_alpha, vif,
)

check_stability(df["score"], df["month"]).metrics      # {'max_psi': ..., 'n_periods': ...}
lift_table(df["score"], df["churned"], n_bands=10)     # decile lift / capture table
```

All thresholds are overridable:

```python
from proxyscore import ProxyAudit, Thresholds

strict = Thresholds(min_auc_strong=0.75, psi_unstable=0.2, leak_auc=0.85)
report = ProxyAudit(indicators=X, score=s, outcome=y, thresholds=strict).run()
```

## What this library deliberately is not

- **Not an SEM/CFA package.** If you have survey-style reflective constructs and want full
  structural equation modeling, use `semopy` or `lavaan`. `proxyscore` targets the messy
  behavioral/transactional data of real business stacks, where those assumptions rarely hold.
- **Not an ML observability platform.** Tools like Evidently monitor model inputs and outputs;
  `proxyscore` validates whether an engineered score measures the *hidden concept* it claims to.
- **Not a guarantee.** Leakage and bias checks are heuristics. The only hard guarantee against
  leakage is a pipeline that snapshots indicators strictly before the outcome window opens.

## Validation philosophy

A proxy for a latent construct can never be validated against the construct itself — only
against observable consequences the construct is supposed to drive. That's why the audit
treats **downstream validation against a delayed hard outcome as the gate** to decision-grade
status: a score that has never been confronted with reality is an opinion with units.

For the full picture — reflective vs. formative constructs, the stack of validity types,
leakage, PSI monitoring, Goodhart's Law, and decision-grade thinking, each mapped to what the
tool does — see **[A practitioner's guide to proxy metrics](docs/proxy-metrics-guide.md)**.

## Roadmap

- Convergent/discriminant validity for multi-construct setups (AVE, HTMT)
- Measurement invariance testing across segments
- Eigenvector/loading drift for PCA-based scores
- Survival-style validation for right-censored time-to-event outcomes

Contributions and issue reports are welcome.

## License

[MIT](LICENSE)
