"""Internal helpers: input coercion, alignment, basic statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def as_indicator_frame(indicators: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a numeric indicator DataFrame."""
    if not isinstance(indicators, pd.DataFrame):
        raise TypeError("indicators must be a pandas DataFrame")
    if indicators.shape[1] < 1:
        raise ValueError("indicators must have at least one column")
    non_numeric = [
        c for c in indicators.columns if not pd.api.types.is_numeric_dtype(indicators[c])
    ]
    if non_numeric:
        raise TypeError(
            f"indicator columns must be numeric; non-numeric columns: {non_numeric}. "
            "Encode categorical indicators before passing them in."
        )
    return indicators.astype(float)


def as_series(values, name: str, index=None) -> pd.Series:
    """Coerce array-like input to a Series, optionally adopting an index."""
    if isinstance(values, pd.Series):
        return values.rename(name)
    values = np.asarray(values)
    if index is not None and len(values) == len(index):
        return pd.Series(values, index=index, name=name)
    return pd.Series(values, name=name)


def is_binary(outcome: pd.Series) -> bool:
    """True when the outcome has exactly two distinct non-null values."""
    return outcome.dropna().nunique() == 2


def to_binary(outcome: pd.Series) -> pd.Series:
    """Map a two-valued outcome to {0, 1} (larger / later value becomes 1)."""
    vals = sorted(outcome.dropna().unique())
    if len(vals) != 2:
        raise ValueError("outcome is not binary")
    return (outcome == vals[1]).astype(float).where(outcome.notna())


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Column-wise z-score; zero-variance columns become 0."""
    std = df.std(ddof=0).replace(0, np.nan)
    out = (df - df.mean()) / std
    return out.fillna(0.0)


def auc_score(score: np.ndarray, outcome: np.ndarray) -> float:
    """ROC AUC via the rank (Mann-Whitney) formulation. Handles ties.

    ``outcome`` must contain only 0s and 1s. Returns NaN when only one
    class is present.
    """
    score = np.asarray(score, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    mask = ~(np.isnan(score) | np.isnan(outcome))
    score, outcome = score[mask], outcome[mask]
    n_pos = int(outcome.sum())
    n_neg = int(len(outcome) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(score)
    auc = (ranks[outcome == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman correlation on pairwise-complete observations."""
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 3 or df.iloc[:, 0].nunique() < 2 or df.iloc[:, 1].nunique() < 2:
        return float("nan")
    rho, _ = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return float(rho)


def fmt(x: float, digits: int = 3) -> str:
    """Compact number formatting for report text."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"
