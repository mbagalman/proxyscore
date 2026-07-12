# PCA loading-drift monitoring

`assess_pca_loading_drift` checks whether the first principal component in a later sample still
has the same indicator-weight pattern as an approved fitted `PCAScore`. It compares against stored
baseline state and never mutates or refits the baseline constructor.

## Direct assessment

```python
from proxyscore import PCAScore, assess_pca_loading_drift

approved_pca = PCAScore().fit(reference[indicator_columns])

drift = assess_pca_loading_drift(
    approved_pca,
    july[indicator_columns],
    n_bootstrap=500,
    random_state=42,
)

print(drift.metrics())
print(drift.loadings)
print(drift.to_markdown())
```

The current sample is standardized using its own means and standard deviations, and a diagnostic
first component is extracted. That diagnostic component is used only for comparison. Production
scores must continue to come from `approved_pca.transform(...)`, which uses the approved baseline
means, standard deviations, and loadings.

## Reported evidence

The assessment reports:

- sign-aligned cosine similarity between baseline and current loading vectors;
- current-minus-baseline loading deltas and the maximum absolute delta;
- baseline and current first-component explained-variance ratios and their difference;
- percentile intervals for similarity, maximum loading delta, current loadings, loading deltas,
  and explained-variance change;
- input, complete-case, dropped-row, and valid-bootstrap counts.

PCA component signs are arbitrary. The implementation therefore flips the entire current vector
when necessary to maximize alignment with the baseline before calculating differences. It never
flips individual indicators, which would conceal a genuine structural change.

Bootstrap resampling draws rows from the current batch and re-estimates only its diagnostic PCA.
The intervals represent current-batch sampling uncertainty. A baseline artifact contains fitted
state rather than the original rows, so baseline-fit uncertainty cannot be reconstructed and is
not implied by these intervals.

## Sample safeguards

- The baseline must be a fitted `PCAScore` with columns, loadings, and explained variance.
- Every baseline indicator must exist in the current frame; extra columns are ignored.
- Inputs must be numeric and finite where observed, with a unique row index.
- All metrics use one complete-case current sample and report dropped rows.
- The default minimum is 100 complete rows.
- A batch with no varying principal direction is rejected as unassessable.
- Set `n_bootstrap=0` for point estimates only; intervals are then missing with an explicit warning.

## Batch-monitoring integration

`create_monitoring_baseline` already persists fitted PCA means, standard deviations, loadings, and
explained variance. When `monitor_batch` restores a `PCAScore`, it now adds a
`pca_loading_drift` check and a per-indicator detail table automatically. Composite and externally
scored baselines do not receive an irrelevant PCA check.

Default monitoring thresholds are:

| Metric | Warning | Failure |
|---|---:|---:|
| Cosine similarity | below `0.98` | below `0.95` |
| Maximum absolute loading delta | at least `0.10` | at least `0.20` |
| Explained-variance drop | at least `0.05` | at least `0.10` |

The persisted `MonitoringLimits` also set `pca_min_complete_rows=100` and
`pca_bootstrap_samples=200`. These defaults are operational screening values, not universal
statistical laws. Set stricter or looser values from validated historical variation and the
business consequences of a changed score meaning.

```python
from proxyscore import MonitoringLimits, create_monitoring_baseline

baseline = create_monitoring_baseline(
    reference[indicator_columns],
    score_id="customer-health-pca",
    score_version="2026.1",
    score_constructor=approved_pca,
    monitoring_limits=MonitoringLimits(
        pca_cosine_warning_below=0.97,
        pca_cosine_failure_below=0.92,
        pca_loading_delta_warning=0.12,
        pca_loading_delta_failure=0.22,
        pca_explained_variance_drop_warning=0.06,
        pca_explained_variance_drop_failure=0.12,
        pca_bootstrap_samples=500,
    ),
)
```

Loading drift asks whether the data's dominant direction has changed. It complements rather than
replaces score PSI, per-indicator PSI, missingness, volume, and matured-outcome performance. A
stable score distribution can coexist with a changed loading structure, and loading drift alone
does not establish whether the new structure is substantively better or worse.
