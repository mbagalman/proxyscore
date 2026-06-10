"""Build composite proxy scores from indicator data.

Two pragmatic constructors are provided:

- :class:`CompositeScore` - the industry-scorecard approach: normalize each
  indicator, apply weights, sum. This treats the construct as *formative*
  (the indicators define the score).
- :class:`PCAScore` - the first principal component of the standardized
  indicators. Useful as a data-driven weighting when indicators are
  expected to share one dominant dimension.

Both follow a minimal fit/transform API so weights learned on a development
sample can be applied to later periods without re-fitting (which matters
for honest stability monitoring).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._utils import as_indicator_frame


class CompositeScore:
    """Weighted, normalized composite of indicators.

    Parameters
    ----------
    weights:
        Mapping of column name to weight. Missing columns get weight 1.0.
        Use negative weights for indicators that move against the
        construct (e.g. support tickets in a health score). If omitted,
        all indicators get equal weight 1.0.
    scaling:
        ``"zscore"`` (default), ``"minmax"``, or ``"rank"`` (percentile
        ranks in [0, 1]).

    Notes
    -----
    Scaling parameters (means, stds, mins, maxes) are learned in
    :meth:`fit` and reused in :meth:`transform`, so the score is
    comparable across batches scored with the same fitted instance.
    """

    def __init__(self, weights: dict[str, float] | None = None, scaling: str = "zscore"):
        if scaling not in ("zscore", "minmax", "rank"):
            raise ValueError("scaling must be 'zscore', 'minmax', or 'rank'")
        self.weights = dict(weights) if weights else None
        self.scaling = scaling
        self.columns_: list[str] | None = None
        self.center_: pd.Series | None = None
        self.scale_: pd.Series | None = None
        self.fit_values_: pd.DataFrame | None = None

    def fit(self, indicators: pd.DataFrame) -> CompositeScore:
        X = as_indicator_frame(indicators)
        self.columns_ = list(X.columns)
        if self.scaling == "zscore":
            self.center_ = X.mean()
            self.scale_ = X.std(ddof=0).replace(0, 1.0)
        elif self.scaling == "minmax":
            self.center_ = X.min()
            self.scale_ = (X.max() - X.min()).replace(0, 1.0)
        else:  # rank: keep the fit sample as the reference distribution
            self.fit_values_ = X.copy()
        return self

    def transform(self, indicators: pd.DataFrame) -> pd.Series:
        if self.columns_ is None:
            raise RuntimeError("call fit() before transform()")
        X = as_indicator_frame(indicators)[self.columns_]
        if self.scaling == "rank":
            scaled = pd.DataFrame(index=X.index)
            for c in self.columns_:
                ref = self.fit_values_[c].dropna().sort_values().to_numpy()
                if len(ref) == 0:
                    scaled[c] = 0.5
                else:
                    pos = np.searchsorted(ref, X[c].to_numpy(), side="right")
                    scaled[c] = pos / len(ref)
        else:
            scaled = (X - self.center_) / self.scale_
        w = pd.Series(
            {c: (self.weights or {}).get(c, 1.0) for c in self.columns_}, dtype=float
        )
        score = (scaled * w).sum(axis=1, skipna=True) / w.abs().sum()
        return score.rename("proxy_score")

    def fit_transform(self, indicators: pd.DataFrame) -> pd.Series:
        return self.fit(indicators).transform(indicators)


class PCAScore:
    """First principal component of standardized indicators as a score.

    The component sign is aligned so that the score correlates positively
    with the mean of the standardized indicators, making "higher = more of
    the construct" hold when most indicators are positively oriented.
    Re-orient negatively-oriented indicators (e.g. multiply by -1) before
    fitting for best results.
    """

    def __init__(self) -> None:
        self.columns_: list[str] | None = None
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None
        self.loadings_: pd.Series | None = None
        self.explained_variance_ratio_: float | None = None

    def fit(self, indicators: pd.DataFrame) -> PCAScore:
        X = as_indicator_frame(indicators).dropna()
        if len(X) < 3:
            raise ValueError("need at least 3 complete rows to fit PCAScore")
        self.columns_ = list(X.columns)
        self.mean_ = X.mean()
        self.std_ = X.std(ddof=0).replace(0, 1.0)
        Z = ((X - self.mean_) / self.std_).to_numpy()
        _, s, vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
        loadings = vt[0]
        # align sign with the average indicator so higher score = more construct
        if (Z @ loadings).std() > 0 and np.corrcoef(Z @ loadings, Z.mean(axis=1))[0, 1] < 0:
            loadings = -loadings
        self.loadings_ = pd.Series(loadings, index=self.columns_, name="loading")
        var = s**2
        self.explained_variance_ratio_ = float(var[0] / var.sum()) if var.sum() > 0 else 0.0
        return self

    def transform(self, indicators: pd.DataFrame) -> pd.Series:
        if self.loadings_ is None:
            raise RuntimeError("call fit() before transform()")
        X = as_indicator_frame(indicators)[self.columns_]
        Z = (X - self.mean_) / self.std_
        return (Z * self.loadings_).sum(axis=1, skipna=True).rename("proxy_score")

    def fit_transform(self, indicators: pd.DataFrame) -> pd.Series:
        return self.fit(indicators).transform(indicators)
