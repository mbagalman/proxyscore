# Repeatable batch monitoring

BR-005 adds a storage-agnostic monitoring workflow with two explicit steps:

1. `create_monitoring_baseline` captures validated reference state in a versioned JSON artifact.
2. `monitor_batch` evaluates each later batch against that fixed state without refitting or
   redefining bins.

The library does not schedule jobs or deliver notifications. It emits stable JSON, Markdown, and
HTML results plus an alert state and process exit code that schedulers and notification systems
can consume.

## Creating a baseline

Create the baseline from the development or approved reference sample. A fitted
`CompositeScore` or `PCAScore` can be included so later batches can be scored from indicators
without refitting:

```python
from proxyscore import CompositeScore, create_monitoring_baseline

constructor = CompositeScore(
    weights={"usage": 1, "depth": 1, "tickets": -0.5},
    scaling="zscore",
).fit(reference[indicator_columns])

baseline = create_monitoring_baseline(
    reference[indicator_columns],
    score_id="customer-health",
    score_version="2026.1",
    score_constructor=constructor,
    outcome=reference["churned_next_quarter"],
    metadata={
        "owner": "Revenue Analytics",
        "approved_population": "Active North American accounts",
    },
)

baseline.save("customer-health-2026.1-baseline.json")
```

If scores are produced by another system, supply `score=` instead of a constructor. That artifact
still stores fixed bins and reference distributions, but every monitoring run must then receive
its externally generated score.

## Artifact contents

The JSON artifact contains:

- artifact format, package, score, and model versions;
- creation timestamp and user metadata;
- required indicator columns and dtypes;
- baseline row volume and missingness rates;
- fixed score and indicator bin cuts with baseline proportions;
- audit thresholds and operational monitoring limits;
- baseline score summary and optional delayed-outcome performance;
- fitted construction state for supported score constructors.

`CompositeScore` state includes fitted scaling parameters, weights, coverage rules, and the rank
reference sample when rank scaling is used. `PCAScore` state includes fitted means, standard
deviations, loadings, and explained variance. Monitoring calls `transform`; it never calls `fit`.

Artifacts use format version `1.0`. An unsupported or malformed version raises
`ArtifactVersionError` before monitoring begins. There is no automatic migration yet; recreate
the baseline with a compatible package when that error occurs.

## Monthly monitoring run

```python
from proxyscore import MonitoringBaseline, monitor_batch

baseline = MonitoringBaseline.load("customer-health-2026.1-baseline.json")

result = monitor_batch(
    baseline,
    july[indicator_columns],
    outcome=july["churned_next_quarter"],  # omit until this outcome has matured
    score_version="2026.1",
    batch_id="2026-07",
)

result.write_json("monitoring/customer-health-2026-07.json")
result.write_html("monitoring/customer-health-2026-07.html")
print(result.to_markdown())
```

Saving each result under its batch ID creates an append-only monitoring history without tying the
library to a database, object store, scheduler, or cloud vendor.

For an externally generated score:

```python
result = monitor_batch(
    baseline,
    july[indicator_columns],
    score=july["health_score"],
    score_version="2026.1",
    batch_id="2026-07",
)
```

## Preflight validation

Monitoring validates compatibility before calculating drift or performance:

- score version must match when supplied;
- every required indicator must be present;
- required indicators must remain numeric and finite apart from allowed missing values;
- indexes must be unique and aligned;
- a score must be supplied when no fitted constructor was stored.

A missing column, incompatible score version, invalid value, or unavailable scoring state returns
a `failure` result containing only the schema check and
`validation_stopped_before_metrics=true`. Extra columns are ignored but produce a warning.

## Checks

Each valid run evaluates:

- **Volume:** current rows relative to baseline rows.
- **Score drift:** PSI using the baseline's stored score cuts and proportions.
- **Indicator drift:** per-indicator PSI using each stored reference distribution.
- **Missingness:** current rates and change from baseline.
- **Outcome performance:** oriented AUC or absolute Spearman once delayed outcomes mature.

Outcome performance compares with the baseline metric when one was captured. It is marked
not-assessable when outcomes are absent, immature, constant, or underpowered. Missing outcome
evidence does not fabricate a pass.

## Alert states and exit codes

The overall run takes the most actionable state present:

| State | Exit code | Meaning |
| --- | ---: | --- |
| `informational` | 0 | Checks are available and within configured limits. |
| `warning` | 1 | Review a moderate shift, volume change, missingness increase, extra schema, or performance decline. |
| `failure` | 2 | Stop or escalate: incompatible input, severe drift, severe volume change, excessive missingness, or material performance failure. |
| `not_assessable` | 3 | No warning or failure exists, but required evidence such as matured outcomes is not yet available. |

Notification delivery remains the caller's responsibility. A scheduler can branch on
`result.exit_code`, while the JSON record provides the check-level explanation and metrics.

## Thresholds

Statistical limits come from the persisted `Thresholds`, including PSI bands, maximum missing
rate, minimum outcome-class count, and weak/strong downstream performance levels.

`MonitoringLimits` controls operational changes:

```python
from proxyscore import MonitoringLimits, Thresholds

baseline = create_monitoring_baseline(
    reference[indicator_columns],
    score_id="customer-health",
    score_version="2026.1",
    score=reference["health_score"],
    thresholds=Thresholds(psi_stable=0.08, psi_unstable=0.20),
    monitoring_limits=MonitoringLimits(
        missing_rate_warning_delta=0.03,
        volume_warning_low=0.70,
        volume_warning_high=1.50,
        performance_warning_drop=0.03,
        performance_failure_drop=0.08,
    ),
)
```

These settings are part of the artifact, so later runs cannot silently pick up different defaults.

## Result retention

`MonitoringResult.to_dict()` and `to_json()` convert enums, timestamps, numpy values, DataFrames,
and missing metrics to standards-compliant JSON. Output never contains non-standard `NaN` or
`Infinity` tokens. Fixed inputs, batch metadata, and observation timestamp produce deterministic
JSON, which makes result files suitable for versioned storage and change review.
