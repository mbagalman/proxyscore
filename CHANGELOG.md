# Changelog

## Unreleased

- Added `align_delayed_outcomes`, `AlignmentResult`, and `AlignmentDiagnostics` for strict
  point-in-time alignment of score snapshots with delayed outcomes, including repeated entities,
  explicit outcome windows, censoring, duplicate-event policies, timezone handling, diagnostics,
  and direct `ProxyAudit` input construction.
- Added `analyze_actions`, `ActionAnalysis`, and `ActionRecommendation` for operating-cutoff,
  percentile, top-N capacity, and candidate-grid evaluation, including classification metrics,
  continuous outcome bands, segment safeguards, explicit business-value assumptions, and
  opt-in constrained candidate recommendations.
- Added self-contained `AuditReport.to_html()` and `write_html()` exports with input scope,
  generation and project metadata, accessible status and table markup, safe escaping, transparent
  detail-table truncation, and optional attached action-analysis results.
- Added `compare_scores`, `ScoreComparison`, and `ComparisonCoverage` for paired score-version
  evaluation with entity coverage diagnostics, bootstrap downstream deltas, distribution and
  lift tables, polarity-aware rank and band migration, stability and segment comparisons, and
  changed action assignments at supplied policies.
- Added versioned `MonitoringBaseline` artifacts and `monitor_batch` runs with fixed score and
  indicator bins, fitted `CompositeScore`/`PCAScore` state, schema preflight, drift, missingness,
  volume and matured-outcome checks, stable JSON/Markdown/HTML output, alert states, and scheduler
  exit codes.
- Added the `proxyscore` command with `audit`, `compare`, `baseline`, and `monitor` workflows,
  versioned TOML configuration, CSV and optional Parquet input, explicit column mappings,
  threshold overrides, JSON/Markdown/HTML output, privacy-safe errors, and stable exit codes.
- Added explicit probability calibration with reusable logistic and isotonic mapping artifacts,
  default stratified holdout evaluation, calibration curves with Wilson intervals, Brier score
  with bootstrap uncertainty, calibration intercept/slope, expected calibration error, and
  sparse-bin, constant-score, and severe-imbalance warnings.
- Added governance and reproducibility manifests with typed ownership and permitted-use context,
  dataset/code revision IDs, strict mode, redaction, deterministic configuration fingerprints,
  and embedding in audit reports plus monitoring artifacts.

## 0.1.0 (2026-06-10)

Initial release.

- `ProxyAudit` orchestrator with decision-grade verdict (`decision_grade` / `directional` / `not_validated`) and markdown report export.
- Score construction: `CompositeScore` (weighted, normalized composite) and `PCAScore` (first principal component).
- Indicator quality and redundancy checks: missingness, variance, item-rest correlation, Cronbach's alpha, VIF, high-correlation pairs, single-indicator dominance.
- Temporal stability: Population Stability Index (PSI) per period against a baseline, with standard 0.10 / 0.25 thresholds.
- Downstream validation against delayed hard outcomes: AUC (binary) or rank correlation (continuous), lift / decile tables, automatic polarity detection.
- Segment bias audit: standardized mean differences across segments and per-segment validity divergence.
- Leakage scan: indicators with suspiciously strong association with the outcome, and leak-suggestive column names.
- Synthetic example dataset: `proxyscore.datasets.make_customer_health()`.

Hardening (post-review, pre-publication):

- Strict row-alignment contract: Series inputs must carry the indicator index (same labels, same order), array-likes must match its length exactly, and duplicate index labels are rejected — silent label-based misalignment between checks is no longer possible.
- Explicit missing-data policy in score construction: missing indicators never become numbers. `CompositeScore` renormalizes partial rows over observed weights with a configurable `min_coverage` floor (NaN below it); `PCAScore` returns NaN for incomplete rows; rank scaling maps missing to NaN instead of percentile 1.0.
- Binary downstream validation requires a minimum count of each outcome class (`min_class_count`, default 10) instead of accepting a single-event AUC.
- Segment validity is only graded where there is enough outcome evidence per segment; segments without it are reported as unassessed (WARN), never folded into a "consistent" PASS.
- Leakage check distinguishes clean, flagged, and unassessed indicators; it SKIPs when nothing was assessable and WARNs when some indicators were.
- `cronbach_alpha` and `vif` are now genuinely listwise-complete (the internal z-score no longer mean-imputes missing values).
- String/bool-labeled binary outcomes work end to end (lift tables included).
- `CompositeScore` weights validated at fit: unknown keys, non-finite values, and zero total weight raise immediately.
- `Thresholds.__post_init__` validates ranges and ordering; `bins`/`n_bands` must be >= 2.
- PSI grading guards against undersized periods (`min_period_rows`, default 50): undersized baselines skip the check, undersized comparison periods are excluded and listed.
- Infinite values are rejected at the indicator boundary with the offending columns named.

Hardening, round 2:

- The verdict now distinguishes "not applicable" from "supplied but unassessable": a check that SKIPs despite its input being provided (underpowered stability baseline, unassessable leakage, too-small segments) caps the verdict at `directional` and is named in the verdict reason.
- Leakage assessability now requires a computable, finite association - enough overlapping rows AND variation on both sides (e.g. both outcome classes present where the indicator is populated). A constant outcome SKIPs the whole check.
- `check_stability` raises on an unknown `baseline_period` instead of reporting it as underpowered.
- `check_downstream` validates `n_bands` at its boundary and no longer swallows `ValueError`s from lift-table construction.
- One consistent `TypeError` for non-numeric outcomes with more than two values across downstream, segment, and leakage checks.
- Infinite values are rejected in scores and numeric outcomes everywhere (`psi`, stability, downstream, segments, `ProxyAudit`), not just in indicator frames.

Hardening, round 3:

- PSI's near-constant-baseline fallback now brackets the midpoint with three bins, so a shift in either direction registers - previously an arbitrarily large upward shift from a constant baseline scored PSI 0.0 while a downward shift alarmed.
- `leakage_scan` no longer crashes when an indicator column is named `outcome` (internal columns are renamed before pairing).
- Segment SMD now uses the sample-size-weighted pooled standard deviation instead of an unweighted variance average that assumed equal segment sizes.
- A segment whose validity is not computable (e.g. constant score within the segment) is reported as unassessed with a WARN instead of being silently dropped from the comparison.
- `check_downstream` SKIPs with a clear message on a constant outcome instead of failing with a misleading "no usable downstream signal".

Hardening, round 4:

- Segments excluded for size (`n < min_segment_size`) now produce a WARN naming them instead of a footnote, so a partially-assessed segment audit caps the overall verdict at `directional`.
- Rank scaling returns NaN for a column that had no observed values at fit time instead of fabricating percentile 0.5; an all-missing indicator now FAILs the indicator quality check.
- `check_leakage` aligns and validates inputs (index, length, outcome type, finiteness) before its constant-outcome early return, and leakage now rejects infinite outcomes like every other check.
- Scores must be real-valued numeric at every public boundary (`ProxyAudit`, downstream, stability, segments, indicator dominance) - one clear TypeError instead of pandas/SciPy internals.
- Two-valued outcomes with mutually unorderable labels (e.g. `1` and `"yes"`) are rejected with a clear error at validation time instead of crashing in `sorted()`.
- `bins` / `n_bands` are validated as integers (>= 2, numpy integers accepted, bool rejected) consistently across `psi`, `psi_over_time`, `check_stability`, `lift_table`, and `check_downstream`.

Hardening, round 5:

- Complex values are rejected across all quantitative inputs (indicators, outcomes, PSI samples) instead of being silently cast to their real component.
- `Thresholds` validation completed: every numeric threshold must be finite, count fields reject booleans and floats, segment gap thresholds are range-checked, and `leak_name_patterns` must be an iterable of non-empty strings (copied defensively).
- `PCAScore.fit` raises a clear error when no indicator varies in the fitted rows, instead of storing an arbitrary direction from a degenerate SVD.
