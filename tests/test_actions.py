from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from proxyscore import ActionRecommendation, Thresholds, analyze_actions


def test_binary_cutoff_reports_confusion_and_rate_metrics():
    result = analyze_actions(
        pd.Series([0.1, 0.4, 0.8, 0.9]),
        pd.Series([0, 1, 0, 1]),
        cutoffs=0.5,
        polarity=1,
    )
    row = result.table.iloc[0]

    assert result.outcome_type == "binary"
    assert (row[["tp", "fp", "tn", "fn"]] == 1).all()
    assert row["selected_n"] == 2
    assert row["selected_rate"] == 0.5
    assert row["precision"] == 0.5
    assert row["recall"] == 0.5
    assert row["specificity"] == 0.5
    assert row["false_positive_rate"] == 0.5
    assert row["false_negative_rate"] == 0.5


def test_auto_polarity_selects_low_health_scores_for_churn():
    result = analyze_actions(
        pd.Series([90.0, 80.0, 20.0, 10.0]),
        pd.Series([0, 0, 1, 1]),
        cutoffs=30,
    )
    row = result.table.iloc[0]

    assert result.polarity == -1
    assert row["tp"] == 2
    assert row["fp"] == 0
    assert row["recall"] == 1
    assert row["precision"] == 1


def test_explicit_percentile_selects_oriented_population_share():
    result = analyze_actions(
        pd.Series(np.arange(1.0, 101.0)),
        pd.Series([0] * 50 + [1] * 50),
        percentiles=10,
        polarity=1,
    )
    row = result.table.iloc[0]
    assert row["strategy"] == "percentile"
    assert row["parameter"] == 10
    assert row["selected_n"] == 10
    assert row["score_cutoff"] == pytest.approx(90.1)


def test_top_n_is_capacity_exact_and_breaks_ties_by_source_order():
    result = analyze_actions(
        pd.Series([3.0, 2.0, 2.0, 1.0]),
        pd.Series([0, 1, 0, 0]),
        top_n=2,
        polarity=1,
    )
    row = result.table.iloc[0]
    assert row["selected_n"] == 2
    assert row["tp"] == 1
    assert row["fp"] == 1
    assert any("original row order" in note for note in result.notes)


@pytest.mark.parametrize("count", [0, 4])
def test_top_n_boundaries(count):
    result = analyze_actions(
        pd.Series([4.0, 3.0, 2.0, 1.0]),
        pd.Series([1, 1, 0, 0]),
        top_n=count,
        polarity=1,
    )
    assert result.table.iloc[0]["selected_n"] == count


def test_default_and_explicit_candidate_grids():
    score = pd.Series(np.linspace(0, 1, 100))
    outcome = pd.Series([0, 1] * 50)
    default = analyze_actions(score, outcome, polarity=1)
    explicit = analyze_actions(score, outcome, grid_size=5, polarity=1)

    assert len(default.table) == 20
    assert set(default.table["strategy"]) == {"grid"}
    assert len(explicit.table) == 5


def test_multiple_policy_modes_can_be_evaluated_together():
    result = analyze_actions(
        pd.Series(np.arange(20.0)),
        pd.Series([0] * 10 + [1] * 10),
        cutoffs=[5, 10],
        percentiles=[10, 25],
        top_n=[1, 3],
        grid_size=3,
        polarity=1,
    )
    assert set(result.table["strategy"]) == {"cutoff", "percentile", "top_n", "grid"}
    assert result.table["policy_id"].is_unique
    assert list(result.assignments.columns) == result.table["policy_id"].tolist()
    assert result.assignments.dtypes.eq(bool).all()


def test_business_value_and_break_even_metrics():
    result = analyze_actions(
        pd.Series([0.1, 0.4, 0.8, 0.9]),
        pd.Series([0, 1, 0, 1]),
        cutoffs=0.5,
        polarity=1,
        true_positive_benefit=100,
        false_positive_cost=20,
        false_negative_cost=50,
        action_cost=5,
    )
    row = result.table.iloc[0]

    assert row["expected_value"] == 20
    assert row["expected_value_per_record"] == 5
    assert row["net_value_vs_no_action"] == 120
    assert row["break_even_tp_benefit"] == 80
    assert result.assumptions == {
        "true_positive_benefit": 100.0,
        "false_positive_cost": 20.0,
        "false_negative_cost": 50.0,
        "action_cost": 5.0,
    }


def test_unspecified_economic_inputs_are_recorded_as_zero():
    result = analyze_actions(
        pd.Series([1.0, 2.0, 3.0, 4.0]),
        pd.Series([0, 0, 1, 1]),
        cutoffs=3,
        true_positive_benefit=10,
    )
    assert result.assumptions["false_positive_cost"] == 0
    assert result.assumptions["false_negative_cost"] == 0
    assert result.assumptions["action_cost"] == 0


def test_recommendation_is_explicit_and_records_objective_constraints_and_assumptions():
    result = analyze_actions(
        pd.Series(np.arange(1.0, 11.0)),
        pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        top_n=[2, 5, 8],
        polarity=1,
        true_positive_benefit=20,
        false_positive_cost=10,
        action_cost=1,
    )
    recommendation = result.recommend("expected_value", max_actions=5, min_recall=0.4)

    assert isinstance(recommendation, ActionRecommendation)
    assert recommendation.selected_n == 5
    assert recommendation.objective == "expected_value"
    assert recommendation.constraints == {"max_actions": 5, "min_recall": 0.4}
    assert recommendation.sample_size == 10
    assert "Validate prospectively" in recommendation.statement
    assert "true_positive_benefit=20" in recommendation.statement


def test_recommendation_rejects_missing_economics_and_impossible_constraints():
    result = analyze_actions(
        pd.Series([1.0, 2.0, 3.0, 4.0]),
        pd.Series([0, 0, 1, 1]),
        top_n=[1, 2],
    )
    with pytest.raises(ValueError, match="explicit business-value assumptions"):
        result.recommend("expected_value")
    with pytest.raises(ValueError, match="no evaluated policy"):
        result.recommend("recall", max_actions=0, min_recall=1)


def test_continuous_outcome_reports_selected_and_unselected_means():
    result = analyze_actions(
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        pd.Series([10.0, 20.0, 30.0, 40.0, 50.0]),
        top_n=2,
    )
    row = result.table.iloc[0]

    assert result.outcome_type == "continuous"
    assert result.polarity == 1
    assert row["selected_outcome_mean"] == 45
    assert row["unselected_outcome_mean"] == 20
    assert row["outcome_mean_difference"] == 25
    with pytest.raises(ValueError, match="not available for continuous"):
        result.recommend("f1")


def test_continuous_negative_polarity_selects_low_scores():
    result = analyze_actions(
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        pd.Series([50.0, 40.0, 30.0, 20.0, 10.0]),
        top_n=2,
    )
    assert result.polarity == -1
    assert result.table.iloc[0]["selected_outcome_mean"] == 45


def test_segment_metrics_and_sparse_segment_safeguards():
    score = pd.Series(np.arange(85.0))
    outcome = pd.Series(([0, 1] * 42) + [1])
    segments = pd.Series(["a"] * 40 + ["b"] * 40 + ["small"] * 5)
    result = analyze_actions(
        score,
        outcome,
        cutoffs=42,
        polarity=1,
        segments=segments,
        thresholds=Thresholds(min_segment_size=30, min_class_count=10),
    )
    segment_table = result.segment_table
    assert segment_table is not None
    assessed = segment_table.set_index("segment")["assessed"]
    assert bool(assessed["a"])
    assert bool(assessed["b"])
    assert not bool(assessed["small"])
    reason = segment_table.set_index("segment").loc["small", "unassessed_reason"]
    assert reason == "n < 30"


def test_highly_imbalanced_binary_outcome_is_calculated():
    score = pd.Series(np.arange(1000.0))
    outcome = pd.Series([0] * 990 + [1] * 10)
    result = analyze_actions(score, outcome, top_n=10)
    row = result.table.iloc[0]
    assert row["tp"] == 10
    assert row["precision"] == 1
    assert row["recall"] == 1


@pytest.mark.parametrize(
    ("score", "outcome", "message"),
    [
        ([1, 1, 1], [0, 1, 0], "score must contain"),
        ([1, 2, 3], [1, 1, 1], "outcome must contain"),
    ],
)
def test_degenerate_inputs_are_rejected(score, outcome, message):
    with pytest.raises(ValueError, match=message):
        analyze_actions(pd.Series(score), pd.Series(outcome), cutoffs=2)


def test_invalid_policy_and_economic_inputs_are_rejected():
    score = pd.Series([1.0, 2.0, 3.0, 4.0])
    outcome = pd.Series([0, 0, 1, 1])
    with pytest.raises(ValueError, match="percentiles"):
        analyze_actions(score, outcome, percentiles=0)
    with pytest.raises(ValueError, match="top_n cannot exceed"):
        analyze_actions(score, outcome, top_n=5)
    with pytest.raises(ValueError, match="non-negative"):
        analyze_actions(score, outcome, cutoffs=2, false_positive_cost=-1)
    with pytest.raises(ValueError, match="supported for binary"):
        analyze_actions(
            score,
            pd.Series([1.0, 2.0, 4.0, 8.0]),
            cutoffs=2,
            true_positive_benefit=1,
        )


def test_index_mismatch_is_rejected():
    score = pd.Series([1.0, 2.0, 3.0], index=[10, 11, 12])
    outcome = pd.Series([0, 0, 1], index=[0, 1, 2])
    with pytest.raises(ValueError, match="index does not match"):
        analyze_actions(score, outcome, cutoffs=2)


def test_two_valued_string_outcome_is_supported():
    result = analyze_actions(
        pd.Series([1.0, 2.0, 3.0, 4.0]),
        pd.Series(["no", "no", "yes", "yes"]),
        cutoffs=3,
    )
    assert result.table.iloc[0]["tp"] == 2
    assert result.table.iloc[0]["precision"] == 1
