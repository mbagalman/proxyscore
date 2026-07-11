import numpy as np
import pandas as pd
import pytest

from proxyscore import CompositeScore, PCAScore


@pytest.fixture
def X():
    rng = np.random.default_rng(42)
    latent = rng.normal(0, 1, 500)
    return pd.DataFrame(
        {
            "a": latent + rng.normal(0, 0.5, 500),
            "b": 2 * latent + rng.normal(0, 0.5, 500),
            "c": -latent + rng.normal(0, 0.5, 500),
        }
    )


def test_composite_equal_weights(X):
    score = CompositeScore().fit_transform(X)
    assert len(score) == len(X)
    assert score.notna().all()


def test_composite_negative_weight_orients_indicator(X):
    score = CompositeScore(weights={"c": -1.0}).fit_transform(X)
    # with c flipped, the composite should track the latent strongly via all three
    assert score.corr(X["b"]) > 0.8


def test_composite_transform_uses_fit_scaling(X):
    cs = CompositeScore().fit(X.iloc[:250])
    s1 = cs.transform(X.iloc[250:])
    cs2 = CompositeScore().fit(X.iloc[250:])
    s2 = cs2.transform(X.iloc[250:])
    # same rows scored with different baselines differ (proves fit state is used)
    assert not np.allclose(s1.to_numpy(), s2.to_numpy())


def test_composite_scaling_modes(X):
    for scaling in ("zscore", "minmax", "rank"):
        s = CompositeScore(scaling=scaling).fit_transform(X)
        assert s.notna().all(), scaling


def test_composite_rejects_bad_scaling():
    with pytest.raises(ValueError):
        CompositeScore(scaling="bogus")


def test_composite_transform_before_fit_raises(X):
    with pytest.raises(RuntimeError):
        CompositeScore().transform(X)


def test_pca_score_tracks_latent(X):
    ps = PCAScore()
    score = ps.fit_transform(X)
    # first PC should capture the shared latent dimension
    assert abs(score.corr(X["b"])) > 0.85
    assert ps.explained_variance_ratio_ > 0.5
    assert set(ps.loadings_.index) == {"a", "b", "c"}


def test_pca_sign_alignment(X):
    # higher score should mean "more of the average indicator"
    score = PCAScore().fit_transform(X)
    z = (X - X.mean()) / X.std(ddof=0)
    assert score.corr(z.mean(axis=1)) > 0


def test_non_numeric_rejected():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    with pytest.raises(TypeError):
        CompositeScore().fit(df)


def test_composite_get_set_params():
    cs = CompositeScore(weights={"a": 2.0}, min_coverage=0.8)
    params = cs.get_params()
    assert params["weights"] == {"a": 2.0}
    assert params["min_coverage"] == 0.8

    cs2 = CompositeScore().set_params(**params)
    assert cs2.weights == {"a": 2.0}
    assert cs2.min_coverage == 0.8


def test_composite_save_load(tmp_path, X):
    cs = CompositeScore().fit(X)
    path = tmp_path / "model.pkl"
    cs.save(path)

    cs2 = CompositeScore.load(path)
    assert cs2.columns_ == cs.columns_
    assert (cs2.center_ == cs.center_).all()
    assert (cs2.scale_ == cs.scale_).all()
    assert np.allclose(cs2.transform(X), cs.transform(X))


def test_pca_get_set_params():
    ps = PCAScore()
    assert ps.get_params() == {}
    ps.set_params()


def test_pca_save_load(tmp_path, X):
    ps = PCAScore().fit(X)
    path = tmp_path / "model.pkl"
    ps.save(path)

    ps2 = PCAScore.load(path)
    assert ps2.columns_ == ps.columns_
    assert (ps2.mean_ == ps.mean_).all()
    assert (ps2.std_ == ps.std_).all()
    assert (ps2.loadings_ == ps.loadings_).all()
    assert ps2.explained_variance_ratio_ == ps.explained_variance_ratio_
    assert np.allclose(ps2.transform(X), ps.transform(X))
