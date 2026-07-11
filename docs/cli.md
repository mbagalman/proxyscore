# Command-line interface

The `proxyscore` command runs audits, score comparisons, monitoring-baseline creation, and batch
monitoring without custom Python code.

```text
proxyscore audit
proxyscore compare
proxyscore baseline
proxyscore monitor
```

Every command accepts CSV or Parquet input, explicit column mappings, a versioned TOML config,
machine-readable JSON output, and optional Markdown and standalone HTML reports.

## Installation

```bash
pip install proxyscore
proxyscore --help
```

CSV works with the base installation. Parquet requires a pandas Parquet engine:

```bash
pip install "proxyscore[parquet]"
```

## Audit

```bash
proxyscore audit \
  --input customer-health.csv \
  --indicators logins feature_depth support_tickets nps \
  --score health_score \
  --outcome churned_next_quarter \
  --segments plan_tier \
  --period snapshot_month \
  --json-output reports/audit.json \
  --markdown-output reports/audit.md \
  --html-output reports/audit.html
```

Omit `--score` to let `ProxyAudit` construct its default equal-weight score. Omit optional outcome,
segment, or period mappings only when their corresponding checks are intentionally out of scope.

## Compare

```bash
proxyscore compare \
  --input score-versions.parquet \
  --baseline-score health_score_v1 \
  --candidate-score health_score_v2 \
  --outcome churned_next_quarter \
  --segments plan_tier \
  --period snapshot_month \
  --action-top-n 50 100 250 \
  --n-bootstrap 1000 \
  --json-output reports/comparison.json \
  --html-output reports/comparison.html
```

The comparison uses the same paired rows for both versions. The CLI exposes raw cutoffs through
`--action-cutoffs`, oriented population shares through `--action-percentiles`, and exact
capacities through `--action-top-n`.

## Baseline

The CLI baseline command captures externally generated scores. Python callers can additionally
persist fitted `CompositeScore` or `PCAScore` state.

```bash
proxyscore baseline \
  --input approved-reference.csv \
  --indicators usage depth tickets \
  --score health_score \
  --outcome churned_next_quarter \
  --score-id customer-health \
  --score-version 2026.1 \
  --artifact baselines/customer-health-2026.1.json \
  --metadata owner=analytics
```

The artifact itself is the required machine-readable JSON output. Optional Markdown and HTML
paths produce a short creation summary.

## Monitor

```bash
proxyscore monitor \
  --input monthly/2026-07.csv \
  --artifact baselines/customer-health-2026.1.json \
  --score health_score \
  --outcome churned_next_quarter \
  --score-version 2026.1 \
  --batch-id 2026-07 \
  --json-output monitoring/2026-07.json \
  --markdown-output monitoring/2026-07.md \
  --html-output monitoring/2026-07.html
```

Omit `--outcome` until delayed outcomes mature. That produces a not-assessable performance check
and warning-class CLI exit rather than inventing a successful result.

## TOML configuration

All commands accept `--config`. A config must declare `config_version = 1`. Command-line options
override values from the corresponding command table.

```toml
config_version = 1

[audit]
input = "customer-health.csv"
indicators = ["logins", "feature_depth", "support_tickets", "nps"]
score = "health_score"
outcome = "churned_next_quarter"
segments = "plan_tier"
period = "snapshot_month"
json_output = "reports/audit.json"
html_output = "reports/audit.html"

[thresholds]
min_auc_strong = 0.70
psi_unstable = 0.20

[metadata]
organization = "Example Company"
project = "Customer health review"
```

```bash
proxyscore audit --config examples/configs/audit.toml
```

Supported sections are `[common]`, the command table, `[output]`, `[thresholds]`,
`[monitoring_limits]`, and `[metadata]`. Use repeated `--threshold NAME=VALUE` arguments to
override statistical thresholds, repeated `--limit NAME=VALUE` arguments on `baseline` for
monitoring limits, and `--metadata KEY=VALUE` for metadata.

Complete examples live in [`examples/configs`](../examples/configs/audit.toml).

## Output and privacy

If `--json-output` is omitted, JSON is written to standard output. Markdown and HTML are written
only when their paths are supplied. Parent output directories are created automatically.

The CLI never logs DataFrame rows or raw values by default. Input errors identify file, option,
or column names; analytical detail belongs in structured output files.

## Exit codes

| Code | Meaning |
| ---: | --- |
| `0` | Successful or informational result. |
| `1` | Warning, inconclusive comparison, directional audit, or not-yet-assessable monitoring evidence. |
| `2` | Failed audit/monitoring validation or a comparison containing a descriptive regression. |
| `3` | Invalid config, missing file/column, incompatible artifact, or malformed input. |
| `4` | Unexpected internal error at the process boundary. |

Schedulers should retain the JSON result before branching on the exit code. Exit status is a
compact routing signal, not a substitute for the check-level explanation.
