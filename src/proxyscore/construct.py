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

Missing-data policy: a missing indicator never silently becomes a number.
:class:`CompositeScore` renormalizes partially observed rows over the
weights actually present, and returns NaN when observed weight falls below
``min_coverage`` (or when everything is missing). :class:`PCAScore` returns
NaN for any incomplete row, because a partial projection is not on the same
scale as a complete one.
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
        Mapping of column name to weight. Columns not mentioned get weight
        1.0. Use negative weights for indicators that move against the
        construct (e.g. support tickets in a health score). If omitted,
        all indicators get equal weight 1.0. Keys must match fitted
        columns and values must be finite; total absolute weight must be
        nonzero (validated in :meth:`fit`).
    scaling:
        ``"zscore"`` (default), ``"minmax"``, or ``"rank"`` (percentile
        ranks in [0, 1]).
    min_coverage:
        Minimum share of total absolute weight that must come from
        non-missing indicators for a row to receive a score; rows below
        this get NaN. Default 0.5. Set to 1.0 to require complete rows.
        Partially observed rows above the floor are renormalized over the
        observed weights.

    Notes
    -----
    Scaling parameters (means, stds, mins, maxes) are learned in
    :meth:`fit` and reused in :meth:`transform`, so the score is
    comparable across batches scored with the same fitted instance.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        scaling: str = "zscore",
        min_coverage: float = 0.5,
    ):
        if scaling not in ("zscore", "minmax", "rank"):
            raise ValueError("scaling must be 'zscore', 'minmax', or 'rank'")
        if not 0 < min_coverage <= 1:
            raise ValueError(f"min_coverage must be in (0, 1], got {min_coverage}")
        self.weights = dict(weights) if weights else None
        self.scaling = scaling
        self.min_coverage = min_coverage
        self.columns_: list[str] | None = None
        self.center_: pd.Series | None = None
        self.scale_: pd.Series | None = None
        self.fit_values_: pd.DataFrame | None = None

    def fit(self, indicators: pd.DataFrame) -> CompositeScore:
        X = as_indicator_frame(indicators)
        self.columns_ = list(X.columns)
        if self.weights is not None:
            unknown = sorted(set(self.weights) - set(self.columns_))
            if unknown:
                raise ValueError(
                    f"weights refer to columns not in the indicators: {unknown}"
                )
            non_finite = {k: v for k, v in self.weights.items() if not np.isfinite(v)}
            if non_finite:
                raise ValueError(f"weights must be finite, got: {non_finite}")
        total = sum(abs((self.weights or {}).get(c, 1.0)) for c in self.columns_)
        if total == 0:
            raise ValueError("total absolute weight is zero; the composite is undefined")
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
                vals = X[c].to_numpy()
                if len(ref) == 0:
                    # no reference distribution was learned for this column;
                    # fabricating a neutral rank would be a silent number
                    pct = np.full(len(vals), np.nan)
                else:
                    pct = np.searchsorted(ref, vals, side="right") / len(ref)
                scaled[c] = np.where(np.isnan(vals), np.nan, pct)
        else:
            scaled = (X - self.center_) / self.scale_
        w = pd.Series(
            {c: (self.weights or {}).get(c, 1.0) for c in self.columns_}, dtype=float
        )
        total = float(w.abs().sum())
        observed_weight = scaled.notna().mul(w.abs(), axis=1).sum(axis=1)
        score = (scaled * w).sum(axis=1, min_count=1) / observed_weight.replace(0, np.nan)
        score = score.where(observed_weight / total >= self.min_coverage)
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

    Rows with any missing indicator transform to NaN: a projection using
    only some loadings is not on the same scale as a complete one.
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
        if float((s**2).sum()) == 0.0:
            raise ValueError(
                "PCAScore cannot be fitted: no indicator varies across the fitted rows, "
                "so there is no principal direction to learn"
            )
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
        return (Z * self.loadings_).sum(axis=1, skipna=False).rename("proxy_score")

    def fit_transform(self, indicators: pd.DataFrame) -> pd.Series:
        return self.fit(indicators).transform(indicators)
