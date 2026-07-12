from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from proxyscore import (
    OutcomeSpec,
    Status,
    Verdict,
    align_delayed_outcomes,
    compare_outcomes,
    validate_outcomes,
)


def evidence(n: int = 600, seed: int = 91) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    indicators = pd.DataFrame(
        {
            "usage": latent + rng.normal(scale=1.5, size=n),
            "service": 0.4 * latent + rng.normal(scale=1.7, size=n),
        }
    )
    score = pd.Series(latent + rng.normal(scale=0.45, size=n), name="score")
    binary = pd.Series((latent + rng.normal(scale=0.8, size=n) > 0).astype(int))
    continuous = pd.Series(latent + rng.normal(scale=1.0, size=n))
    outcomes = pd.concat(
        [binary.rename("binary"), continuous.rename("continuous")], axis=1
    )
    return indicators, score, outcomes


def test_mixed_outcomes_keep_separate_samples_and_report_details():
    indicators, score, values = evidence()
    values.loc[:99, "binary"] = np.nan
    values.loc[500:, "continuous"] = np.nan
    report = validate_outcomes(
        score,
        indicators,
        {
            "churn": OutcomeSpec(
                values["binary"], "binary", polarity="positive", window="90d"
            ),
            "expansion": OutcomeSpec(
                values["continuous"],
                "continuous",
                polarity="positive",
                window="180d",
                importance="supporting",
            ),
        },
    )

    assert report["churn"].evaluation_rows == 500
    assert report["expansion"].evaluation_rows == 500
    assert report["churn"].downstream.metrics["outcome_type"] == "binary"
    assert report["expansion"].downstream.metrics["outcome_type"] == "continuous"
    assert {"churn.downstream", "churn.leakage", "expansion.downstream"}.issubset(
        report.tables()
    )
    assert "# Multi-outcome validation" in report.to_markdown(max_rows=5)


def test_maturity_is_per_outcome_and_immature_required_outcome_cannot_pass():
    indicators, score, values = evidence(n=300)
    mature = pd.Series(False, index=score.index)
    report = validate_outcomes(
        score,
        indicators,
        {
            "mature": OutcomeSpec(values["binary"], "binary", window="30d"),
            "immature": OutcomeSpec(
                values["continuous"],
                "continuous",
                window="365d",
                mature=mature,
            ),
        },
    )

    assert report["mature"].evaluation_rows == 300
    assert report["immature"].mature_rows == 0
    assert report["immature"].immature_rows == 300
    assert report.verdict is Verdict.NOT_VALIDATED


def test_expected_polarity_conflict_is_named_and_caps_supporting_evidence():
    indicators, score, values = evidence()
    report = validate_outcomes(
        score,
        indicators,
        {
            "required": OutcomeSpec(values["binary"], "binary", window="90d"),
            "contradiction": OutcomeSpec(
                values["continuous"],
                "continuous",
                polarity="negative",
                window="180d",
                importance="supporting",
            ),
        },
    )

    assert report["contradiction"].detected_polarity == 1
    assert report["contradiction"].polarity_contradiction
    assert report.verdict is Verdict.DIRECTIONAL
    assert "contradiction" in report.verdict_reason


def test_failed_required_outcome_controls_verdict_without_averaging():
    indicators, score, values = evidence()
    rng = np.random.default_rng(17)
    random_outcome = pd.Series(rng.normal(size=len(score)))
    report = validate_outcomes(
        score,
        indicators,
        {
            "strong": OutcomeSpec(values["binary"], "binary", window="90d"),
            "required_failure": OutcomeSpec(
                random_outcome, "continuous", window="180d"
            ),
        },
    )

    assert report["strong"].downstream.status is Status.PASS
    assert report["required_failure"].downstream.status is Status.FAIL
    assert report.verdict is Verdict.NOT_VALIDATED
    assert "required_failure" in report.verdict_reason


def test_comparison_runs_each_outcome_on_its_own_mature_sample():
    indicators, _, values = evidence(n=500)
    rng = np.random.default_rng(45)
    latent = values["continuous"]
    baseline = latent + rng.normal(scale=2.0, size=len(latent))
    candidate = latent + rng.normal(scale=0.3, size=len(latent))
    churn_mature = pd.Series([True] * 400 + [False] * 100)
    expansion_mature = pd.Series([False] * 100 + [True] * 400)

    result = compare_outcomes(
        baseline,
        candidate,
        {
            "churn": OutcomeSpec(
                values["binary"], "binary", window="90d", mature=churn_mature
            ),
            "expansion": OutcomeSpec(
                values["continuous"],
                "continuous",
                window="180d",
                mature=expansion_mature,
                importance="supporting",
            ),
        },
        n_bootstrap=40,
        random_state=7,
    )

    assert set(result.comparisons) == {"churn", "expansion"}
    assert (result.outcome_summary["evaluation_rows"] == 400).all()
    assert set(result.outcome_summary["assessment"]) == {"improved"}
    assert "churn.performance" in result.tables()
    assert "# Multi-outcome score comparison" in result.to_markdown(max_rows=5)


def test_alignment_result_preserves_censoring_as_maturity():
    observations = pd.DataFrame(
        {
            "account": ["a", "b", "c"],
            "scored_at": pd.to_datetime(["2026-01-01", "2026-01-10", "2026-01-20"]),
        }
    )
    events = pd.DataFrame(
        {
            "account": ["a"],
            "event_at": pd.to_datetime(["2026-01-05"]),
            "churned": [1],
        }
    )
    aligned = align_delayed_outcomes(
        observations,
        events,
        entity="account",
        score_time="scored_at",
        outcome="churned",
        outcome_time="event_at",
        horizon="10d",
        as_of="2026-01-15",
    )
    spec = OutcomeSpec.from_alignment(aligned, outcome_type="binary", window="10d")

    assert spec.mature.tolist() == [True, False, False]


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (OutcomeSpec([0, 1, 2], "binary", window="30d"), "declared binary"),
        (OutcomeSpec(["a", "b", "c"], "continuous", window="30d"), "must be numeric"),
        (OutcomeSpec([0, 1, 0], "binary", window=""), "window"),
        (OutcomeSpec([0, 1, 0], "binary", window="30d", mature=[1, 0, 1]), "boolean"),
    ],
)
def test_outcome_configuration_validation(spec: OutcomeSpec, message: str):
    with pytest.raises((TypeError, ValueError), match=message):
        validate_outcomes(
            pd.Series([1.0, 2.0, 3.0]),
            pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
            {"outcome": spec},
        )


def test_at_least_one_required_outcome_is_enforced():
    indicators, score, values = evidence(n=100)
    with pytest.raises(ValueError, match="at least one"):
        validate_outcomes(
            score,
            indicators,
            {
                "support": OutcomeSpec(
                    values["binary"], "binary", window="30d", importance="supporting"
                )
            },
        )
