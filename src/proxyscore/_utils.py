"""Internal helpers: input coercion, alignment, basic statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def as_indicator_frame(indicators: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a numeric, finite indicator DataFrame."""
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
    complex_cols = [
        c for c in indicators.columns if pd.api.types.is_complex_dtype(indicators[c])
    ]
    if complex_cols:
        raise TypeError(
            f"indicator columns must be real-valued; complex columns: {complex_cols}"
        )
    X = indicators.astype(float)
    inf_counts = np.isinf(X).sum()
    inf_cols = inf_counts[inf_counts > 0]
    if len(inf_cols) > 0:
        raise ValueError(
            f"indicators contain infinite values: {inf_cols.to_dict()} "
            "(column: count). Replace them with NaN or a finite value first."
        )
    return X


def check_unique_index(index: pd.Index, what: str) -> None:
    """Raise when an index has duplicate labels (label alignment is ambiguous)."""
    if index.has_duplicates:
        dups = index[index.duplicated()].unique()[:5].tolist()
        raise ValueError(
            f"{what} index contains duplicate labels (e.g. {dups}). "
            "Row alignment would be ambiguous; make the index unique first."
        )


def aligned_series(values, name: str, index: pd.Index) -> pd.Series:
    """Coerce input to a Series aligned to ``index``, refusing silent mismatches.

    A Series must carry exactly ``index`` (same labels, same order). An
    array-like must have exactly ``len(index)`` elements and adopts the index.
    """
    if isinstance(values, pd.Series):
        if not values.index.equals(index):
            raise ValueError(
                f"{name} index does not match the indicator/score index (same labels in "
                f"the same order required). Reindex it explicitly before passing it in."
            )
        return values.rename(name)
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != len(index):
        raise ValueError(
            f"{name} has length {values.shape[0] if values.ndim else 0}, expected "
            f"{len(index)} to align with the other inputs."
        )
    return pd.Series(values, index=index, name=name)


def as_series(values, name: str) -> pd.Series:
    """Coerce array-like input to a Series (used for the reference input itself)."""
    if isinstance(values, pd.Series):
        return values.rename(name)
    return pd.Series(np.asarray(values), name=name)


def ensure_count(value, minimum: int, name: str) -> None:
    """Raise when a count-like parameter is not an integer >= ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def validate_score(s: pd.Series, what: str = "score") -> None:
    """Raise unless a score Series is real-valued numeric and finite."""
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_complex_dtype(s):
        raise TypeError(f"{what} must be real-valued numeric, got dtype {s.dtype}")
    ensure_finite(s, what)


def ensure_finite(s: pd.Series, what: str) -> None:
    """Raise when a numeric Series is complex or contains infinite values
    (NaN is allowed)."""
    if pd.api.types.is_complex_dtype(s):
        raise TypeError(f"{what} must be real-valued, got complex dtype")
    if not pd.api.types.is_numeric_dtype(s):
        return
    n_inf = int(np.isinf(s.to_numpy(dtype="float64", na_value=np.nan)).sum())
    if n_inf:
        raise ValueError(
            f"{what} contains {n_inf} infinite value(s); replace them with NaN or "
            f"finite values first."
        )


def check_outcome_type(outcome: pd.Series) -> None:
    """Reject outcomes no check can handle, with one consistent error.

    Accepted: numeric outcomes (continuous or binary) and two-valued
    outcomes of any type (strings, booleans, categories).
    """
    y = outcome.dropna()
    if pd.api.types.is_complex_dtype(y):
        raise TypeError("outcome must be real-valued; complex outcomes are not supported")
    if pd.api.types.is_numeric_dtype(y):
        return
    if y.nunique() == 2:
        vals = list(y.unique())
        try:
            sorted(vals)
        except TypeError as exc:
            raise TypeError(
                f"two-valued outcome labels must be mutually orderable to identify the "
                f"positive class; got {vals!r}. Encode them consistently (e.g. 0/1) first."
            ) from exc
        return
    raise TypeError(
        f"outcome must be numeric or two-valued; got a non-numeric outcome with "
        f"{y.nunique()} distinct values"
    )


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
    """Column-wise z-score. Missing values stay missing; zero-variance
    columns become 0 (where observed)."""
    std = df.std(ddof=0)
    out = (df - df.mean()) / std.replace(0, np.nan)
    zero_var = std == 0
    if zero_var.any():
        out.loc[:, zero_var] = 0.0
        out = out.where(df.notna())
    return out


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
