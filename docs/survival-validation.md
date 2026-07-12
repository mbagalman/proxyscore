# Survival-style validation

Fixed-window binary outcomes discard useful timing information and can misclassify customers whose
outcome has not had enough time to occur. `assess_survival_validation` evaluates a score against a
right-censored time-to-event outcome while retaining the follow-up duration for every row.

This is a standalone assessment. It does not run inside `ProxyAudit`, because censored outcomes
have a different data contract and cannot safely be treated as an ordinary binary label.

## Install

Survival estimators are deliberately supplied by the maintained `scikit-survival` package rather
than reimplemented in `proxyscore`:

```bash
pip install "proxyscore[survival]"
```

The base package remains limited to NumPy, pandas, and SciPy. Importing `proxyscore` does not load
the optional dependency; calling the survival assessment without it raises an installation error.

## Inputs

The evaluation sample requires one aligned value per entity:

- `risk_score`: an arbitrary numeric score. State whether higher or lower values imply an earlier
  event with `risk_direction`.
- `duration`: positive follow-up time from the score's as-of date to either the event or the last
  date the entity was known to be event-free. Use one consistent unit.
- `event_observed`: `True`/`1` when the event occurred at `duration`; `False`/`0` when follow-up
  ended without the event. This censoring indicator is mandatory.
- `horizons`: the positive business times at which performance should be evaluated, such as 30,
  60, and 90 days. Horizons must be supported by both samples' follow-up.

The assessment also requires `reference_duration` and `reference_event_observed`. This reference
follow-up sample is used by `scikit-survival` to estimate the censoring distribution for inverse
probability of censoring weighting (IPCW). Use the model-development or another representative
historical sample that precedes the evaluation cohort. Do not use a reference population with a
materially different follow-up process without investigating the effect.

To assess probability predictions, optionally supply `survival_probabilities`: an evaluation-row
by horizon matrix containing `P(T > horizon)`. A DataFrame must use the score index and have
columns exactly equal to the declared horizons. Probabilities must be finite, lie in `[0, 1]`, and
must not increase at later horizons.

```python
from proxyscore import assess_survival_validation

survival = assess_survival_validation(
    evaluation["account_risk"],
    evaluation["days_to_event_or_last_followup"],
    evaluation["event_observed"],
    horizons=[30, 60, 90],
    reference_duration=development["days_to_event_or_last_followup"],
    reference_event_observed=development["event_observed"],
    risk_direction="higher",
    survival_probabilities=predicted_survival[[30, 60, 90]],
)

print(survival.ranking_summary)
print(survival.ranking_by_horizon)
print(survival.calibration_by_horizon)
```

Series inputs use strict index and row-order alignment. Plain arrays must have the expected length.
Rows missing a score, duration, or censoring flag are dropped and reported. Probability matrices
must be complete for the retained evaluation sample. Duplicate indexes, nonpositive durations,
unsupported horizons, a constant score, and insufficient samples or events fail before estimation.

## Ranking results

`ranking_summary` reports:

- **IPCW concordance through the latest horizon:** the probability that, among comparable rows,
  the entity with the earlier event has higher oriented risk. The table also reports concordant,
  discordant, tied-risk, and tied-time pair counts.
- **Mean cumulative/dynamic AUC:** the survival-function-weighted average of the declared-horizon
  AUC values.

`ranking_by_horizon` reports cumulative/dynamic AUC at each declared horizon. Cases are entities
with an observed event by that horizon; controls are entities known to remain event-free beyond
it. Rows censored before the horizon are handled through IPCW rather than silently treated as
non-events. Case and control counts are included for interpretation.

Both estimators use the censoring distribution from the explicit reference sample. A value near
0.5 indicates chance ranking; larger values indicate better ordering after score polarity is
applied. Business acceptance thresholds should be declared before evaluation and justified for
the use case rather than borrowed mechanically from fixed-window AUC rules.

## Probability results

`calibration_by_horizon` is populated only when survival probabilities are supplied. It reports:

- **IPCW Brier score:** squared probability error adjusted for right censoring; lower is better.
- **Mean predicted event probability:** the cohort-level average of `1 - P(T > horizon)`.

The Brier score is a proper probability score, but it reflects both discrimination and
calibration. It is therefore kept in a distinct table and is not described as pure
calibration-in-the-large. Inspect calibration plots or use specialized survival-model tooling when
the decision requires detailed calibration shape, recalibration, competing risks, or uncertainty
intervals.

If probabilities are omitted, ranking still runs and the report explicitly marks probability
evaluation as not assessed. A good concordance or AUC result must never be interpreted as evidence
that raw score values are probabilities.

## Scope and limits

- Censoring must be conditionally independent of event risk given the available information. IPCW
  cannot repair informative loss to follow-up that is not represented in the analysis.
- The event definition, time origin, follow-up unit, administrative cutoff, reference population,
  and horizons must be fixed before reading results.
- This assessment supports one right-censored event process. It does not implement competing
  risks, recurrent events, left truncation, interval censoring, time-varying covariates, causal
  treatment effects, or model fitting.
- Small samples and sparse events are rejected by configurable minimums. Passing minimums does not
  guarantee precise estimates; use bootstrap or model-specific inference for uncertainty.
- Validate the censoring and event-extraction pipeline independently. Incorrectly labeling rows
  that have not matured as censored still produces plausible-looking but invalid output.

The returned `SurvivalValidationAssessment` includes sample counts, warnings, notes, `tables()`,
and `to_markdown()` for review artifacts. Ranking and probability evidence remain separate all the
way through the report.
