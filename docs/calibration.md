# Probability calibration

Ranking and calibration answer different questions. A score can rank high-risk records first
while being wrong about the level of risk. Use calibration when a score will be presented or
consumed as a probability, expected event rate, or input to an expected-value decision.

`proxyscore` never treats an arbitrary proxy score as a probability. There are two supported
paths:

1. Fit a documented mapping from the arbitrary score to probability.
2. Explicitly declare that the supplied values are already probabilities in `[0, 1]`.

## Fit and evaluate on separate samples

The convenience workflow creates a reproducible, stratified holdout. The mapping is fitted on
the development partition and every reported metric is calculated on the disjoint evaluation
partition:

```python
from proxyscore import fit_and_assess_calibration

calibration = fit_and_assess_calibration(
    df["health_score"],
    df["churned_next_quarter"],
    method="logistic",       # or "isotonic"
    evaluation_fraction=0.25,
    random_state=42,
)

print(calibration.metrics)
print(calibration.curve)
print(calibration.to_markdown())
```

For a time-based or otherwise externally defined split, fit and evaluate explicitly. This is
usually preferable for production validation:

```python
from proxyscore import assess_calibration, fit_calibrator

mapping = fit_calibrator(
    development["health_score"],
    development["churned_next_quarter"],
    method="isotonic",
)
assessment = assess_calibration(
    validation["health_score"],
    validation["churned_next_quarter"],
    model=mapping,
)

mapping_json = mapping.to_json()
```

`CalibrationModel` stores only the fitted aggregate state, method, fit sample size, positive
count, and artifact version. It does not store training rows. `to_dict()`, `to_json()`,
`from_dict()`, and `from_json()` support deployment and reproducibility.

## Already-probabilistic inputs

When a score is genuinely a probability, opt in explicitly:

```python
from proxyscore import assess_calibration

assessment = assess_calibration(
    validation["predicted_churn_probability"],
    validation["churned"],
    assume_probabilities=True,
)
```

Without either `model=` or `assume_probabilities=True`, assessment raises an error. The explicit
opt-in prevents convenient-looking values such as a 0-to-100 health score from being interpreted
as probabilities.

## Methods

- **Logistic calibration** fits `sigmoid(intercept + slope * score)`. It is smooth,
  parsimonious, and generally the safer default for modest samples.
- **Isotonic calibration** uses weighted pool-adjacent-violators regression. It learns a
  monotonic step function without assuming a logistic shape, but needs more data and can
  overfit small development samples.

Both are implemented locally with NumPy and SciPy so calibration does not add a large mandatory
machine-learning dependency. Predictions outside the isotonic development range are clipped to
the nearest fitted step.

## Metrics and binning

The result reports:

- `brier_score`: mean squared error between probability and binary outcome; lower is better.
- `calibration_in_the_large` (also available as `calibration_intercept`): the intercept-only
  recalibration offset with `logit(prediction)` fixed at slope `1`. Its ideal value is `0`;
  positive or negative values indicate systematic underprediction or overprediction.
- `calibration_model_intercept`: the intercept estimated jointly with the calibration slope. Its
  ideal value is `0`, but it describes the fitted recalibration line at logit prediction `0` and
  is not calibration-in-the-large.
- `calibration_slope`: ideal value `1`; a value below `1` commonly indicates predictions that
  are too extreme. The slope and joint model intercept are not identifiable for constant
  predictions.
- `expected_calibration_error`: sample-weighted mean absolute difference between each bin's mean
  prediction and observed rate; lower is better.
- `evaluation_sample_size`, positive count/rate, requested/effective bin counts, and fit sample
  size where a mapping was fitted.

Curve data uses quantile boundaries on probability values. Equal predictions always remain in
the same bin, making the curve and ECE invariant to row order. When ties cross a requested
boundary, the effective number of bins is lower and bin sizes can be uneven. Each row includes
the bin size, positive count, mean prediction, observed rate, absolute gap, and a Wilson
confidence interval for the observed rate. The Brier score receives a nonparametric bootstrap
confidence interval by default.

`min_bin_size` defaults to 30. Every smaller bin is marked `sparse=True` and produces a warning.
Severe class imbalance and constant predictions also produce explicit warnings. These warnings
matter: calibration curves can look precise while being driven by very few positive outcomes.

Calibration evidence is scope-specific. A mapping validated for one population, outcome window,
or period should not be silently reused after those conditions change; retain the fitted artifact
and monitor both discrimination and calibration on newly matured outcomes.
