from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from proxyscore import ScoreComparison, compare_scores


def binary_versions(n=1200, seed=41):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    outcome = (latent + rng.normal(scale=0.8, size=n) > 0).astype(int)
    baseline = 0.25 * latent + rng.normal(scale=1.4, size=n)
    candidate = latent + rng.normal(scale=0.35, size=n)
    return pd.Series(baseline), pd.Series(candidate), pd.Series(outcome)


def test_candidate_improvement_has_positive_paired_interval():
    baseline, candidate, outcome = binary_versions()
    result = compare_scores(
        baseline,
        candidate,
        outcome,
        n_bootstrap=200,
        random_state=7,
    )
    row = result.performance.iloc[0]

    assert isinstance(result, ScoreComparison)
    assert result.outcome_type == "binary"
    assert row["candidate"] > row["baseline"]
    assert row["ci_lower"] > 0
    assert row["assessment"] == "improved"
    assert row["valid_bootstrap_samples"] == 200
    assert result.dimensions.iloc[0]["assessment"] == "improved"


def test_candidate_regression_has_negative_paired_interval():
    baseline, candidate, outcome = binary_versions()
    result = compare_scores(
        candidate,
        baseline,
        outcome,
        n_bootstrap=200,
        random_state=7,
    )
    row = result.performance.iloc[0]
    assert row["ci_upper"] < 0
    assert row["assessment"] == "regressed"


def test_identical_scores_are_inconclusive_with_zero_delta():
    _, score, outcome = binary_versions(n=500)
    result = compare_scores(score, score.copy(), outcome, n_bootstrap=80)
    row = result.performance.iloc[0]

    assert row["candidate_minus_baseline"] == pytest.approx(0)
    assert row["ci_lower"] == pytest.approx(0)
    assert row["ci_upper"] == pytest.approx(0)
    assert row["assessment"] == "inconclusive"
    assert result.metrics["oriented_spearman"] == pytest.approx(1)
    assert result.rank_movements["absolute_rank_change"].max() == pytest.approx(0)


def test_reversed_scale_is_oriented_before_rank_and_performance_comparison():
    _, score, outcome = binary_versions(n=500)
    result = compare_scores(score, -score, outcome, n_bootstrap=80)

    assert result.metrics["baseline_polarity"] == 1
    assert result.metrics["candidate_polarity"] == -1
    assert result.metrics["raw_spearman"] == pytest.approx(-1)
    assert result.metrics["oriented_spearman"] == pytest.approx(1)
    assert result.performance.iloc[0]["candidate_minus_baseline"] == pytest.approx(0)
    assert (result.migration["baseline_band"] == result.migration["candidate_band"]).all()


def test_series_index_coverage_and_missing_rows_are_reported():
    baseline = pd.Series(np.arange(100.0), index=np.arange(0, 100))
    candidate = pd.Series(np.arange(100.0), index=np.arange(10, 110))
    outcome = pd.Series(([0, 1] * 55), index=np.arange(0, 110), dtype=float)
    baseline.loc[11] = np.nan
    candidate.loc[12] = np.nan
    outcome.loc[20] = np.nan

    result = compare_scores(baseline, candidate, outcome, n_bootstrap=0)
    coverage = result.coverage

    assert coverage.baseline_rows == 100
    assert coverage.candidate_rows == 100
    assert coverage.overlap_rows == 90
    assert coverage.baseline_only_rows == 10
    assert coverage.candidate_only_rows == 10
    assert coverage.baseline_missing_in_overlap == 1
    assert coverage.candidate_missing_in_overlap == 1
    assert coverage.outcome_missing_in_overlap == 1
    assert coverage.evaluation_rows == 87
    assert result.metrics["n"] == 87


def test_array_inputs_require_equal_lengths():
    with pytest.raises(ValueError, match="candidate_score has length"):
        compare_scores([1, 2, 3], [1, 2], [0, 1, 0])


def test_indexed_and_unindexed_scores_cannot_be_mixed():
    with pytest.raises(TypeError, match="must both be Series"):
        compare_scores(pd.Series([1, 2, 3]), [1, 2, 3], pd.Series([0, 1, 0]))


def test_indexed_scores_require_indexed_optional_inputs():
    baseline, candidate, outcome = binary_versions(n=50)
    with pytest.raises(TypeError, match="segments must be a Series"):
        compare_scores(baseline, candidate, outcome, segments=["a"] * 50)
    with pytest.raises(TypeError, match="period must be a Series"):
        compare_scores(baseline, candidate, outcome, period=["m1"] * 50)


def test_duplicate_entity_indexes_are_rejected():
    baseline = pd.Series([1.0, 2.0, 3.0], index=[1, 1, 2])
    candidate = pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3])
    outcome = pd.Series([0, 1, 0], index=[1, 2, 3])
    with pytest.raises(ValueError, match="duplicate labels"):
        compare_scores(baseline, candidate, outcome)


def test_lift_distributions_and_named_versions_use_same_sample():
    baseline, candidate, outcome = binary_versions(n=300)
    baseline.iloc[0] = np.nan
    result = compare_scores(
        baseline,
        candidate,
        outcome,
        baseline_name="v1",
        candidate_name="v2",
        n_bands=5,
        n_bootstrap=20,
    )

    assert set(result.distributions["version"]) == {"v1", "v2"}
    assert set(result.lift["version"]) == {"v1", "v2"}
    assert len(result.lift) == 10
    assert (result.distributions["n"] == 299).all()
    assert {"v1", "v2"}.issubset(result.performance.columns)


def test_stability_and_segment_tables_are_compared():
    baseline, candidate, outcome = binary_versions(n=600)
    period = pd.Series(np.repeat(["m1", "m2", "m3"], 200))
    segments = pd.Series(np.repeat(["enterprise", "smb"], 300))
    candidate.loc[period == "m3"] += 1.5

    result = compare_scores(
        baseline,
        candidate,
        outcome,
        period=period,
        segments=segments,
        n_bootstrap=30,
    )

    assert result.stability is not None
    assert set(result.stability["version"]) == {"baseline", "candidate"}
    assert "max_period_psi" in set(result.dimensions["dimension"])
    assert result.segments is not None
    assert set(result.segments["version"]) == {"baseline", "candidate"}
    assert set(result.segments["segment"]) == {"enterprise", "smb"}
    assert "minimum_segment_validity" in set(result.dimensions["dimension"])


def test_rank_and_band_migration_capture_changed_ordering():
    outcome = pd.Series([0, 0, 0, 1, 1, 1, 1, 1])
    baseline = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    candidate = pd.Series([8.0, 7.0, 3.0, 4.0, 5.0, 6.0, 2.0, 1.0])
    result = compare_scores(
        baseline,
        candidate,
        outcome,
        n_bands=4,
        n_bootstrap=20,
    )

    assert result.rank_movements.iloc[0]["absolute_rank_change"] > 0.3
    assert (result.migration["baseline_band"] != result.migration["candidate_band"]).any()
    assert result.migration["n"].sum() == 8
    assert result.migration["rate"].sum() == pytest.approx(1)


def test_action_capacity_comparison_reports_changed_assignments():
    outcome = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    baseline = pd.Series(np.arange(10.0))
    candidate = pd.Series([0.0, 1.0, 2.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0])
    result = compare_scores(
        baseline,
        candidate,
        outcome,
        action_top_n=[2, 4],
        n_bootstrap=20,
    )

    assert result.actions is not None
    assert set(result.actions["strategy"]) == {"top_n"}
    assert (result.actions["baseline_selected_n"] == result.actions["candidate_selected_n"]).all()
    assert (result.actions["changed_n"] > 0).any()
    assert (result.actions["changed_rate"] <= 1).all()


def test_action_cutoffs_report_version_specific_selection_counts():
    outcome = pd.Series([0, 0, 1, 1, 1, 1])
    baseline = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    candidate = baseline * 10
    result = compare_scores(
        baseline,
        candidate,
        outcome,
        action_cutoffs=3,
        n_bootstrap=20,
    )

    assert result.actions is not None
    row = result.actions.iloc[0]
    assert row["baseline_selected_n"] == 3
    assert row["candidate_selected_n"] == 5
    assert row["changed_n"] == 2


def test_continuous_outcome_uses_oriented_spearman():
    rng = np.random.default_rng(10)
    outcome = pd.Series(np.linspace(0, 10, 400))
    baseline = pd.Series(rng.normal(size=400))
    candidate = outcome + rng.normal(scale=0.5, size=400)
    result = compare_scores(
        baseline,
        candidate,
        outcome,
        n_bootstrap=100,
    )

    assert result.outcome_type == "continuous"
    assert result.performance.iloc[0]["metric"] == "oriented_spearman"
    assert result.performance.iloc[0]["assessment"] == "improved"
    assert "outcome_rate" in result.lift.columns


def test_no_bootstrap_is_explicitly_descriptive_and_inconclusive():
    baseline, candidate, outcome = binary_versions(n=200)
    result = compare_scores(baseline, candidate, outcome, n_bootstrap=0)
    row = result.performance.iloc[0]
    assert row["method"] == "descriptive_only"
    assert row["valid_bootstrap_samples"] == 0
    assert pd.isna(row["ci_lower"])
    assert row["assessment"] == "inconclusive"
    assert "descriptive only" in result.dimensions.iloc[0]["basis"]


def test_markdown_and_table_mapping_include_optional_sections():
    baseline, candidate, outcome = binary_versions(n=120)
    segments = pd.Series(["a"] * 60 + ["b"] * 60)
    result = compare_scores(
        baseline,
        candidate,
        outcome,
        segments=segments,
        action_top_n=20,
        n_bootstrap=20,
    )
    tables = result.tables()
    markdown = result.to_markdown(max_rows=5)

    assert {"coverage", "performance", "segments", "actions"}.issubset(tables)
    assert "# Score version comparison" in markdown
    assert "## Performance" in markdown
    assert "Showing first 5" in markdown


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_bootstrap": -1}, "n_bootstrap"),
        ({"confidence_level": 1}, "confidence_level"),
        ({"n_bands": 1}, "n_bands"),
        ({"baseline_name": "same", "candidate_name": "same"}, "must differ"),
    ],
)
def test_configuration_validation(kwargs, message):
    baseline, candidate, outcome = binary_versions(n=40)
    with pytest.raises(ValueError, match=message):
        compare_scores(baseline, candidate, outcome, **kwargs)


def test_degenerate_and_tiny_inputs_are_rejected():
    with pytest.raises(ValueError, match="at least 3"):
        compare_scores([1, 2], [1, 2], [0, 1])
    with pytest.raises(ValueError, match="score versions"):
        compare_scores([1, 1, 1], [1, 2, 3], [0, 1, 0])
    with pytest.raises(ValueError, match="outcome must contain"):
        compare_scores([1, 2, 3], [1, 2, 4], [0, 0, 0])
