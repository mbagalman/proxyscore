# Business-readiness backlog

This backlog describes the work needed to turn `proxyscore` from a focused validation
library into a practical score-governance tool for business analytics teams. It assumes the
existing audit remains the core product: construct a proxy score, test its indicators,
validate it against a delayed outcome, check leakage and stability, and inspect performance
across segments.

Priorities reflect operational risk: **P0** closes gaps that commonly prevent real-world use,
**P1** makes repeated use and governance practical, and **P2** broadens statistical depth.

## Already working

- Composite-score construction with reusable fitted transformations and PCA-based scores.
- Indicator diagnostics: missingness, redundancy, VIF, item-rest correlation, and internal
  consistency.
- Downstream validation for binary or continuous delayed outcomes, including AUC or rank
  correlation, polarity detection, and lift/capture tables.
- Statistical and name-based leakage checks.
- Score-distribution stability checks over periods using PSI.
- Segment-level score and outcome checks.
- Configurable thresholds and a typed `AuditReport` with summary and Markdown output.
- Explicit `decision_grade`, `directional`, and `not_validated` verdicts.
- Point-in-time alignment of score snapshots with delayed outcomes, including explicit windows,
  repeated entities, censoring, duplicate-event policies, timezone handling, diagnostics, and
  direct `ProxyAudit` inputs.
- Operating-threshold and action analysis for cutoffs, population percentiles, exact capacities,
  candidate grids, segment performance, and explicit binary-outcome economics.
- Self-contained HTML audit reports with input scope, project metadata, accessible status text,
  escaped content, transparent table truncation, and attached action-analysis output.
- Paired score-version comparison with coverage diagnostics, bootstrap downstream uncertainty,
  lift, stability, segments, polarity-aware rank migration, and changed action assignments.
- Versioned monitoring baselines with fixed bins, fitted constructor state, schema preflight,
  score and indicator drift, missingness, volume, matured outcomes, and operator artifacts.
- Shell-friendly `proxyscore` CLI commands for audits, score comparisons, monitoring baselines,
  and batch monitoring, with TOML configuration, output artifacts, and stable exit codes.
- Explicit probability calibration with reusable logistic and isotonic mappings, held-out
  evaluation, calibration curves, Brier score, intercept/slope, ECE, uncertainty, and
  sparse-data warnings.
- Governance and reproducibility manifests with score ownership, permitted-use context,
  dataset/code revisions, row counts, checks, thresholds, strict mode, redaction, and
  deterministic configuration fingerprints.
- Named multi-outcome validation for binary and continuous criteria with separate mature
  samples, required/supporting policy, polarity contradiction detection, and non-averaging
  verdicts.
- Business recipes and a tabular adapter protocol with customer-health, lead-quality, and
  account-risk preparation patterns, local CSV/Parquet adapters, provenance metadata,
  point-in-time SQL examples, deduplication, and credential-handling guidance.
- Exploratory multi-construct validity assessment with named one-factor loadings, AVE, HTMT,
  bootstrap intervals, shared complete-case safeguards, and SEM/CFA escalation guidance.

## P0: required for operational use

### BR-001: Time-window and delayed-outcome alignment - Complete

**Completed:** Implemented and documented in the Unreleased version. The focused alignment and
documentation tests pass as part of the full test suite.

**Problem:** Users supply already-aligned rows. Although the API says outcomes must occur after
the score window, it cannot verify that claim. A same-window or future-data join can therefore
produce a convincing but invalid audit.

**Deliverable:** Add a window-aware alignment API that builds audit-ready data from entity IDs,
score timestamps or observation-window boundaries, outcome timestamps, and a prediction
horizon. It must report exclusions and reject temporal overlap.

**Acceptance criteria**

- Supports one row per entity and repeated entity-period observations.
- Accepts entity key, score/observation timestamp, outcome timestamp, and horizon or explicit
  outcome-window boundaries.
- Enforces that each selected outcome occurs strictly after its score observation window.
- Documents policies for multiple eligible outcomes, censoring, missing outcomes, timezones,
  and boundary inclusivity.
- Reports input, matched, unmatched, censored, and duplicate-candidate row counts plus the
  observed lag distribution.
- Produces data directly usable by `ProxyAudit` without manual index repair.
- Includes churn/renewal and lead-conversion examples.
- Tests overlap, duplicate events, unsorted input, null timestamps, timezones, boundaries, and
  repeated entities.

**Dependencies:** None.

**Non-goal:** Full survival analysis; this ticket establishes honest temporal alignment and
labels censored observations.

### BR-002: Operating-threshold and action analysis - Complete

**Completed:** Implemented and documented in the Unreleased version. The focused action-analysis
and documentation tests pass as part of the full test suite.

**Problem:** A strong overall AUC or correlation does not show whether a specific business
action is viable. Teams need workload, error-rate, and economic consequences at an action
cutoff.

**Deliverable:** Add threshold analysis for binary outcomes and band/action analysis for
continuous outcomes. Support user-supplied cutoffs and candidate-cutoff exploration.

**Acceptance criteria**

- For binary outcomes, reports confusion counts, precision, recall, specificity, false-positive
  and false-negative rates, selection rate, and selected-record count.
- Correctly respects detected score polarity.
- Evaluates explicit cutoffs, percentile cutoffs, top-N capacity, or a generated cutoff grid.
- Accepts optional true-positive benefit, false-positive cost, false-negative cost, and
  per-action cost; reports expected value and break-even information.
- Supports fixed-capacity recommendations such as a review queue size.
- Reports metrics by segment with minimum-sample safeguards.
- Never silently chooses a production cutoff; recommendations state objective, constraints,
  sample, and assumptions.
- Returns structured tables suitable for serialization and reporting.
- Tests class imbalance, reversed polarity, ties, degenerate outcomes, costs, sparse segments,
  and top-N boundaries.

**Dependencies:** BR-001 is recommended so evaluation windows are trustworthy.

### BR-003: Standalone HTML audit report - Complete

**Completed:** Implemented and documented in the Unreleased version. Focused HTML-report and
documentation tests pass as part of the full test suite.

**Problem:** Markdown and DataFrames work for analysts, but business reviewers need a portable
report that opens without Python and can be retained as review evidence.

**Deliverable:** Add `AuditReport.to_html()` and `AuditReport.write_html(path)` methods that
generate a self-contained, accessible report.

**Acceptance criteria**

- Includes verdict and limitations, input scope, statuses, metrics, detail tables, notes, and
  generation metadata.
- Includes BR-002 action analysis when attached to the report.
- Produces one portable HTML file and escapes all user-controlled labels and values.
- Handles skipped checks and truncates large tables transparently.
- Supports report title and organization/project metadata without changing results.
- Uses semantic headings and tables, sufficient contrast, and status labels that do not rely on
  color alone.
- Has deterministic structural tests plus tests for escaping and missing sections.

**Dependencies:** None; integrate BR-002 when available.

**Non-goal:** A hosted dashboard or PDF rendering service.

### BR-004: Score-version comparison - Complete

**Completed:** Implemented and documented in the Unreleased version. Focused comparison,
action-assignment, and documentation tests pass as part of the full test suite.

**Problem:** Businesses rarely assess one immutable score. They need to show whether a proposed
version is better, how population behavior changed, and which records receive different actions.

**Deliverable:** Add a comparison API for baseline and candidate scores evaluated on the same
entities and outcomes.

**Acceptance criteria**

- Validates aligned entity/index coverage and reports dropped or mismatched rows.
- Compares downstream performance, lift/capture, stability, segment results, missingness, and
  distributions on the same sample where applicable.
- Reports score correlation, rank movement, band migration, and changed action assignments at
  supplied cutoffs or capacities.
- Uses paired uncertainty estimates or an appropriate paired test for performance deltas where
  feasible; otherwise labels comparisons descriptive-only.
- Marks each dimension improved, regressed, or inconclusive without hiding tradeoffs in one
  opaque number.
- Produces a structured comparison object with Markdown- and HTML-ready tables.
- Tests reversed scales, ties, missing entities, identical scores, sparse segments, and sample
  mismatch.

**Dependencies:** BR-002 for action comparisons; BR-001 is recommended.

### BR-005: Repeatable monitoring runs - Complete

**Completed:** Implemented and documented in the Unreleased version. Focused monitoring and
documentation tests pass as part of the full test suite.

**Problem:** The library calculates period PSI inside one audit but cannot save a validated
baseline, evaluate a new batch, retain compatible history, and expose an operator alert state.

**Deliverable:** Add a local, storage-agnostic workflow around a versioned baseline artifact and
structured monitoring results.

**Acceptance criteria**

- Persists score bins, fitted construction state, thresholds, schema, package version, creation
  time, and a user-supplied score/model ID.
- Evaluates a new batch without refitting or redefining bins.
- Detects schema mismatch, missing indicators, incompatible versions, and invalid data first.
- Reports score PSI, indicator drift, missingness and volume changes, and delayed-outcome
  performance once outcomes mature.
- Distinguishes informational, warning, failure, and not-yet-assessable states.
- Emits stable JSON-serializable results and Markdown/HTML operator reports.
- Leaves notifications to callers but returns a documented alert state and process exit status.
- Includes artifact versioning, migration/error behavior, round-trip tests, and a monthly batch
  example.

**Dependencies:** BR-004 should define comparison semantics; BR-003 is recommended.

**Non-goal:** Scheduling jobs, storing credentials, or becoming a full observability platform.

## P1: adoption and governance

### BR-006: Command-line interface - Complete

**Completed:** Implemented and documented in the Unreleased version. The focused CLI tests,
Ruff, mypy, and the full test suite pass.

**Problem:** Requiring custom Python raises adoption cost and makes scheduled runs less
standardized.

**Deliverable:** Provide a `proxyscore` CLI for audits, comparisons, baselines, and monitoring.

**Acceptance criteria**

- Supports CSV and Parquet with explicit column mappings.
- Reads versioned YAML or TOML configuration with documented CLI overrides.
- Provides `audit`, `compare`, `baseline`, and `monitor` commands as their APIs become available.
- Writes JSON and optional Markdown or HTML.
- Defines exit codes for success, warning, failed validation, bad input, and internal error.
- Never logs raw row values by default.
- Includes end-to-end tests, example configs, and shell-friendly documentation.

**Dependencies:** BR-003 through BR-005.

### BR-007: Calibration assessment - Complete

**Completed:** Implemented and documented in the Unreleased version. Focused calibration and
documentation tests, Ruff, mypy, and the full test suite pass.

**Problem:** When a score represents a probability or expected risk, rank discrimination is not
enough; predicted levels must agree with observed rates.

**Deliverable:** Add optional calibration checks for probability-like binary scores and a
documented calibration mapping for arbitrary proxy scores.

**Acceptance criteria**

- Never treats arbitrary scores as probabilities without explicit opt-in or a fitted mapping.
- Reports calibration-curve data, Brier score, calibration intercept/slope, and expected
  calibration error with documented binning.
- Separates calibration fitting and evaluation samples by default.
- Supports logistic and isotonic calibration through an optional dependency or justified local
  implementation.
- Warns on sparse bins and reports sample size and uncertainty where feasible.
- Tests perfect calibration, overconfidence, constant scores, and severe imbalance.

**Dependencies:** BR-001 is recommended.

### BR-008: Governance and reproducibility metadata - Complete

**Problem:** An audit is difficult to approve or reproduce without knowing which score, data
window, configuration, and code version produced it.

**Deliverable:** Introduce a typed audit context and manifest attached to reports and monitoring
artifacts.

**Acceptance criteria**

- Supports score name/version, owner, intended and prohibited uses, population, data and outcome
  windows, decision owner, reviewer, and tags.
- Automatically records package version, run time, row counts, checks, thresholds, and a
  deterministic configuration fingerprint.
- Accepts dataset and code revision IDs without requiring a specific vendor.
- Serializes to a documented, versioned JSON schema and embeds in Markdown/HTML reports.
- Excludes secrets and raw row-level data.
- Warns when governance fields are absent and offers strict mode for controlled workflows.
- Tests round trips, schema versions, redaction, and deterministic fingerprints.

**Dependencies:** Coordinate formats with BR-003 and BR-005.

**Completion notes:** Added `GovernanceContext`, `GovernanceManifest`, strict-mode validation,
redaction, deterministic fingerprints, and public helpers. Audit Markdown/HTML reports and
monitoring baseline/result artifacts now embed versioned manifests. Added schema documentation and
focused tests for round trips, schema rejection, redaction, warnings, fingerprints, audit reports,
and monitoring artifacts.

### BR-009: Multi-outcome validation - Complete

**Completed:** Implemented and documented in the Unreleased version. Focused multi-outcome tests
cover mixed types, independent missingness and maturity samples, conflicting polarity, a failed
required outcome, BR-001 alignment handoff, report tables/Markdown, and score comparisons.

**Problem:** A construct may need to predict churn, expansion, escalation, and payment risk.
Separate audits make conflicting evidence and sample differences hard to see.

**Deliverable:** Evaluate named outcomes with their own type, polarity, window, and importance.

**Acceptance criteria**

- Accepts named binary and continuous outcomes with per-outcome configuration.
- Produces separate downstream and leakage results without silently mixing samples.
- Reports sample sizes, maturity windows, missingness, and detected polarity per outcome.
- Defines a documented overall-verdict policy that never hides an outcome by averaging.
- Supports required versus supporting outcomes and identifies contradictory evidence.
- Integrates with comparison and reporting outputs.
- Tests mixed types, differing missingness, conflicting polarity, immature outcomes, and a failed
  required outcome.

**Dependencies:** BR-001 for per-outcome alignment.

**Completion notes:** Added `OutcomeSpec`, `validate_outcomes`, and structured per-outcome results
with explicit type, polarity, window, importance, maturity, sample diagnostics, downstream checks,
and leakage checks. Added a documented non-averaging verdict policy, alignment-aware construction,
namespaced reporting tables, Markdown output, and `compare_outcomes` integration.

### BR-010: Business recipes and data-adapter protocol - Complete

**Completed:** Implemented and documented in the Unreleased version. Focused recipe and adapter
tests pass locally.

**Problem:** Teams need concrete patterns for creating honest audit inputs, while built-in vendor
connectors would create a large security and maintenance burden.

**Deliverable:** Publish tested recipes and define a small protocol for tabular audit inputs.

**Acceptance criteria**

- Includes customer-health, lead-quality, and account-risk recipes.
- Demonstrates point-in-time-correct SQL, snapshots, delayed outcomes, and deduplication.
- Defines an adapter returning pandas DataFrames plus provenance metadata.
- Provides local CSV and Parquet adapters; database examples use optional dependencies.
- Documents data minimization and credential handling.
- Tests recipe code and adapter conformance without live services.

**Dependencies:** BR-001 and BR-008.

**Non-goal:** First-party connectors for every warehouse, CRM, or BI product.

**Completion notes:** Added `TabularAdapter`, `TabularData`, adapter provenance metadata, local
CSV and Parquet adapters, customer-health, lead-quality, and account-risk `BusinessRecipe`
builders, recipe preparation through BR-001 alignment, local deduplication counts, SQL examples,
and a public guide covering optional database extraction, data minimization, and credential
handling.

## P2: statistical depth

### BR-011: Multi-construct convergent and discriminant validity - Complete

**Completed:** Implemented and documented in the Unreleased version. Focused synthetic-reference,
sample-safeguard, documentation, Ruff, mypy, and full-suite checks pass locally.

Implement average variance extracted (AVE) and heterotrait-monotrait ratio (HTMT) for named
constructs, with bootstrap confidence intervals, sample safeguards, synthetic reference tests,
and guidance on when to use a structural-equation-modeling package.

**Completion notes:** Added `ConstructValidityAssessment` and `assess_construct_validity` with
named reflective constructs, exploratory one-factor correlation/PCA loadings, per-construct AVE,
pairwise HTMT, row-bootstrap percentile intervals, one shared complete-case sample, explicit
threshold flags, sample and identifiability safeguards, report-ready tables/Markdown, analytically
known synthetic reference tests, and a guide defining when SEM/CFA software is required.

### BR-012: Measurement invariance across segments

Add staged configural, metric, scalar, and strict invariance testing, or a clearly documented
subset. Report every level separately, handle sparse groups, and never claim cross-group
comparability after a prerequisite level fails. Validate against published reference examples.

**Dependencies:** BR-011 and existing segment checks.

### BR-013: PCA loading-drift monitoring

Store PCA loadings and compare later samples using sign-aligned vector similarity, loading
deltas, explained-variance changes, and uncertainty where appropriate. Integrate with BR-005
without silently refitting the baseline PCA.

**Dependencies:** BR-005 and existing `PCAScore` fitted state.

### BR-014: Survival-style validation

Support right-censored time-to-event outcomes using concordance and time-dependent evaluation at
declared horizons. Require censoring information, separate ranking from calibration, and use a
proven optional survival-analysis dependency rather than implementing estimators from scratch.

**Dependencies:** BR-001.

## Suggested release sequence

1. **0.2 - Honest decisions:** BR-001, BR-002, and BR-003.
2. **0.3 - Change control:** BR-004, BR-005, and BR-008.
3. **0.4 - Team workflows:** BR-006, BR-007, BR-009, and BR-010.
4. **Later statistical releases:** BR-011 through BR-014, independently gated by API design and
   reference-validation work.

Every release should retain the current rule that a favorable verdict is evidence within a
declared validation scope, not blanket authorization to automate a business decision.
