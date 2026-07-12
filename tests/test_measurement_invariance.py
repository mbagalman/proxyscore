from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from proxyscore import (
    MeasurementInvarianceAssessment,
    assess_measurement_invariance,
)

CONSTRUCTS = {
    "trust": ["trust_1", "trust_2", "trust_3"],
    "value": ["value_1", "value_2", "value_3"],
}


def _multigroup_sample(
    *,
    n: int = 400,
    second_loadings: np.ndarray | None = None,
    second_intercepts: np.ndarray | None = None,
    second_residuals: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(812)
    loadings = np.array([1.0, 0.8, 0.7, 1.0, 0.9, 0.75])
    intercepts = np.array([2.0, 1.0, 3.0, 0.5, 2.5, 1.5])
    residuals = np.array([0.45, 0.40, 0.50, 0.35, 0.45, 0.40])
    factor_for = np.array([0, 0, 0, 1, 1, 1])
    frames = []
    labels = []
    for group_index, label in enumerate(("enterprise", "small_business")):
        latent = rng.multivariate_normal(
            mean=[0.0, 0.0] if group_index == 0 else [0.35, -0.25],
            cov=[[1.0, 0.30], [0.30, 1.15]],
            size=n,
        )
        group_loadings = (
            loadings if group_index == 0 or second_loadings is None else second_loadings
        )
        group_intercepts = (
            intercepts if group_index == 0 or second_intercepts is None else second_intercepts
        )
        group_residuals = (
            residuals if group_index == 0 or second_residuals is None else second_residuals
        )
        observed = group_intercepts + latent[:, factor_for] * group_loadings
        observed += rng.normal(scale=np.sqrt(group_residuals), size=(n, 6))
        columns = [item for values in CONSTRUCTS.values() for item in values]
        frames.append(pd.DataFrame(observed, columns=columns))
        labels.extend([label] * n)
    return pd.concat(frames, ignore_index=True), pd.Series(labels, name="segment")


def test_full_invariance_ladder_is_reported_and_supported():
    data, segments = _multigroup_sample()

    result = assess_measurement_invariance(data, segments, CONSTRUCTS)

    assert isinstance(result, MeasurementInvarianceAssessment)
    assert result.levels["level"].tolist() == ["configural", "metric", "scalar", "strict"]
    assert result.levels["converged"].all()
    assert result.levels["supported"].all()
    assert result.highest_supported_level == "strict"
    assert len(result.parameters) == 4 * 2 * 6
    assert result.group_sizes["assessed"].all()


def _exact_moment_sample(mean: np.ndarray, covariance: np.ndarray, n: int) -> np.ndarray:
    rng = np.random.default_rng(n)
    raw = rng.normal(size=(n, len(mean)))
    centered = raw - raw.mean(axis=0)
    q, _ = np.linalg.qr(centered)
    return mean + q @ np.linalg.cholesky(covariance).T * np.sqrt(n)


def _numbers(text: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    values = np.fromstring(text, sep=" ")
    return values.reshape(shape) if shape else values


def test_published_holzinger_swineford_lavaan_reference():
    """Reproduce the published two-school lavaan invariance example from group moments."""
    means = [
        _numbers(
            "4.941239 5.983974 2.487179 2.822650 3.995192 "
            "1.922161 4.429208 5.563141 5.416132"
        ),
        _numbers(
            "4.929885 6.200000 1.995690 3.317241 4.712069 "
            "2.468966 3.920840 5.488276 5.327203"
        ),
    ]
    covariances = [
        _numbers(
            """
            1.395230 0.402103 0.620107 0.567926 0.471940 0.497563 0.045473 0.165195 0.349713
            0.402103 1.503750 0.476758 0.088504 0.142551 0.194036 -0.203615 0.038512 0.218652
            0.620107 0.476758 1.345789 0.164660 0.066445 0.222102 -0.024389 0.151611 0.331457
            0.567926 0.088504 0.164660 1.319686 1.079810 0.755121 0.325563 0.146989 0.155413
            0.471940 0.142551 0.066445 1.079810 1.707910 0.933921 0.175280 0.151505 0.207619
            0.497563 0.194036 0.222102 0.755121 0.933921 0.974449 0.220979 0.205647 0.184915
            0.045473 -0.203615 -0.024389 0.325563 0.175280 0.220979 1.164673 0.423694 0.288559
            0.165195 0.038512 0.151611 0.146989 0.151505 0.205647 0.423694 0.951991 0.353088
            0.349713 0.218652 0.331457 0.155413 0.207619 0.184915 0.288559 0.353088 0.982050
            """,
            (9, 9),
        ),
        _numbers(
            """
            1.318647 0.414310 0.533750 0.439868 0.411134 0.412028 0.123285 0.369523 0.573133
            0.414310 1.226379 0.478448 0.283103 0.204569 0.243892 0.075727 0.194586 0.281398
            0.533750 0.478448 1.073365 0.380965 0.344233 0.407071 0.079681 0.258570 0.395735
            0.439868 0.283103 0.380965 1.257212 0.933298 0.906397 0.241305 0.121765 0.361459
            0.411134 0.204569 0.344233 0.933298 1.341665 0.898084 0.302994 0.239728 0.422277
            0.412028 0.243892 0.407071 0.906397 0.898084 1.280142 0.208295 0.143380 0.315245
            0.123285 0.075727 0.079681 0.241305 0.302994 0.208295 1.061800 0.632835 0.441902
            0.369523 0.194586 0.258570 0.121765 0.239728 0.143380 0.632835 1.094380 0.566652
            0.573133 0.281398 0.395735 0.361459 0.422277 0.315245 0.441902 0.566652 1.051048
            """,
            (9, 9),
        ),
    ]
    arrays = [
        _exact_moment_sample(means[0], covariances[0], 156),
        _exact_moment_sample(means[1], covariances[1], 145),
    ]
    columns = [f"x{index}" for index in range(1, 10)]
    data = pd.DataFrame(np.vstack(arrays), columns=columns)
    segments = pd.Series(["Pasteur"] * 156 + ["Grant-White"] * 145)
    constructs = {
        "visual": ["x1", "x2", "x3"],
        "textual": ["x4", "x5", "x6"],
        "speed": ["x7", "x8", "x9"],
    }

    result = assess_measurement_invariance(
        data, segments, constructs, max_rmsea=0.10
    )
    levels = result.levels.set_index("level")

    # Rosseel's published lavaan example reports 115.851, 124.044, and 164.103.
    assert levels.loc["configural", "chi_square"] == pytest.approx(115.851, abs=0.2)
    assert levels.loc["metric", "chi_square"] == pytest.approx(124.044, abs=0.2)
    assert levels.loc["scalar", "chi_square"] == pytest.approx(164.103, abs=0.2)
    assert levels.loc["configural", "df"] == 48
    assert levels.loc["metric", "df"] == 54
    assert levels.loc["scalar", "df"] == 60
    assert levels.loc["metric", "supported"]
    assert not levels.loc["scalar", "supported"]


def test_scalar_failure_blocks_strict_comparability_claim():
    intercepts = np.array([2.0, 2.3, 3.0, 0.5, 2.5, 1.5])
    data, segments = _multigroup_sample(second_intercepts=intercepts)

    result = assess_measurement_invariance(data, segments, CONSTRUCTS)
    levels = result.levels.set_index("level")

    assert levels.loc["metric", "supported"]
    assert not levels.loc["scalar", "supported"]
    assert not levels.loc["strict", "prerequisite_met"]
    assert not levels.loc["strict", "supported"]
    assert "prerequisite scalar level failed" in levels.loc["strict", "interpretation"]
    assert result.highest_supported_level == "metric"


def test_sparse_group_returns_all_levels_as_unassessed():
    data, segments = _multigroup_sample(n=120)
    segments.iloc[-100:] = "large_replacement"

    result = assess_measurement_invariance(data, segments, CONSTRUCTS, min_group_size=100)

    assert len(result.levels) == 4
    assert not result.levels["supported"].any()
    assert result.parameters.empty
    assert result.highest_supported_level is None
    assert any("n < 100" in warning for warning in result.warnings)


def test_complete_case_markdown_and_tables_are_explicit():
    data, segments = _multigroup_sample(n=120)
    data.loc[0, "trust_1"] = np.nan
    segments.iloc[1] = None

    result = assess_measurement_invariance(data, segments, CONSTRUCTS, min_group_size=100)

    assert result.input_rows == 240
    assert result.complete_rows == 238
    assert result.dropped_rows == 2
    assert set(result.tables()) == {"group_sizes", "levels", "parameters"}
    assert "Highest consecutively supported level" in result.to_markdown()
    assert "complete-case" in result.to_markdown()


def test_indicator_named_segment_does_not_collide_with_grouping_column():
    data, segments = _multigroup_sample(n=120)
    data = data.rename(columns={"trust_1": "segment"})
    constructs = {**CONSTRUCTS, "trust": ["segment", "trust_2", "trust_3"]}

    result = assess_measurement_invariance(
        data, segments, constructs, min_group_size=100
    )

    assert result.levels["converged"].all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_group_size": 2}, "min_group_size"),
        ({"min_cfi": 1.1}, "min_cfi"),
        ({"max_delta_cfi": -0.1}, "max_delta_cfi"),
    ],
)
def test_configuration_safeguards(kwargs, message):
    data, segments = _multigroup_sample(n=20)
    with pytest.raises(ValueError, match=message):
        assess_measurement_invariance(data, segments, CONSTRUCTS, **kwargs)


def test_requires_multiple_nonconstant_nonsingular_groups():
    data, segments = _multigroup_sample(n=120)
    with pytest.raises(ValueError, match="at least two observed segments"):
        assess_measurement_invariance(data, pd.Series("one", index=data.index), CONSTRUCTS)
    data.loc[segments == "enterprise", "trust_1"] = 1.0
    with pytest.raises(ValueError, match="must vary within every segment"):
        assess_measurement_invariance(data, segments, CONSTRUCTS)
