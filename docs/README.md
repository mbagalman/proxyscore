# Documentation

Project documentation for `proxyscore`.

## Guides

- **[Getting started](getting-started.md)** — a hands-on, end-to-end walkthrough: run a
  full audit, read the verdict, interpret every check, see what failures look like, and
  point the tool at your own data. Every example uses real output.
- **[A practitioner's guide to proxy metrics](proxy-metrics-guide.md)** — the *why* behind
  the tool: the full lifecycle of constructing, validating, and using scores for things you
  can't directly measure (reflective vs. formative, criterion validity, leakage, PSI,
  Goodhart's Law, decision-grade thinking), with each concept mapped to what the library does,
  what's on the roadmap, and what belongs to a specialized tool.

- **[Business-readiness backlog](business-readiness-backlog.md)** - prioritized,
  implementation-ready tickets for taking the library from focused score validation to
  repeatable business operation and governance.
- **[Time-window alignment](time-window-alignment.md)** - build point-in-time-correct audit
  inputs from score snapshots and delayed event or outcome tables.
- **[Operating-threshold and action analysis](action-analysis.md)** - evaluate concrete score
  cutoffs, capacities, error rates, segment behavior, and optional business value.
- **[Standalone HTML reports](html-reports.md)** - create portable, accessible audit artifacts
  with project metadata and optional action-analysis results.
- **[Comparing score versions](score-comparison.md)** - compare paired downstream performance,
  uncertainty, rankings, migrations, segments, stability, and action assignments.
- **[Repeatable batch monitoring](monitoring.md)** - persist fixed baseline state and evaluate
  later batches with drift, missingness, volume, and matured-outcome checks.
- **[Command-line interface](cli.md)** - run audit, comparison, baseline, and monitoring
  workflows from CSV or Parquet with versioned TOML configuration.
- **[Probability calibration](calibration.md)** - map arbitrary scores to probabilities and
  evaluate levels with held-out Brier, intercept/slope, ECE, curve, and uncertainty results.
- **[Governance and reproducibility manifests](governance.md)** - attach typed ownership,
  permitted-use, data-window, revision, row-count, threshold, check, and fingerprint metadata to
  audit and monitoring artifacts.
- **[Multi-outcome validation](multi-outcome-validation.md)** - evaluate required and supporting
  binary or continuous outcomes on separate mature samples, detect polarity contradictions, and
  compare score versions without averaging evidence away.
- **[Business recipes and data adapters](business-recipes.md)** - prepare customer-health,
  lead-quality, and account-risk audit inputs with point-in-time SQL patterns, local CSV/Parquet
  adapters, deduplication, provenance, and credential-handling guidance.
- **[Multi-construct validity](construct-validity.md)** - screen named reflective constructs with
  exploratory AVE, HTMT, shared-sample safeguards, bootstrap intervals, and clear SEM/CFA
  escalation guidance.

The top-level [`README.md`](../README.md) has the quick reference and the project's
positioning; [`CHANGELOG.md`](../CHANGELOG.md) tracks releases.

## Layout

- **`internal/`** — working notes that are *not* published with the package
  (kept out of version control via `.gitignore`): the original project brief,
  background research, and code-review records.

Additional public docs (deeper guides, API reference) will be added to this folder as
the project grows.
