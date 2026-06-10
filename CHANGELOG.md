# Changelog

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
