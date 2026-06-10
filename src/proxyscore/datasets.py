"""Synthetic example data for docs, demos, and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_customer_health(
    n: int = 2000,
    n_months: int = 6,
    seed: int = 0,
    include_leak: bool = False,
    drift: float = 0.0,
) -> pd.DataFrame:
    """Simulate a B2B customer-health dataset with a known latent construct.

    Each row is one account in one month. A latent "health" variable drives
    five observable indicators (two of them negatively oriented) and a
    binary churn outcome observed after the indicator window. Segments
    differ in baseline behavior so segment checks have something to find.

    Parameters
    ----------
    n:
        Number of account-month rows.
    n_months:
        Number of distinct months, labelled ``"2025-01"`` onward.
    seed:
        RNG seed.
    include_leak:
        If True, adds a ``renewal_meeting_declined`` indicator that is
        derived almost directly from the churn outcome - a planted leak
        for testing leakage detection.
    drift:
        Amount of mean drift added to the latent health in later months
        (0 = stable). Use ~1.0 to trigger PSI alarms.

    Returns
    -------
    DataFrame with indicator columns (``logins``, ``feature_depth``,
    ``support_tickets``, ``nps``, ``payment_delay_days``), a built-in
    ``health_score`` (equal-weight z-composite with oriented signs),
    ``churned`` outcome, ``segment``, ``month``, and the true ``latent_health``
    (which a real analyst would never see).
    """
    rng = np.random.default_rng(seed)
    months = [f"2025-{m + 1:02d}" for m in range(n_months)]
    month = rng.choice(months, size=n)
    month_idx = np.array([months.index(m) for m in month])

    segment = rng.choice(["smb", "mid_market", "enterprise"], size=n, p=[0.5, 0.3, 0.2])
    seg_shift = np.select(
        [segment == "smb", segment == "mid_market", segment == "enterprise"],
        [-0.2, 0.0, 0.3],
    )

    latent = rng.normal(0, 1, n) + seg_shift
    if drift:
        latent = latent + drift * month_idx / max(n_months - 1, 1)

    noise = lambda s: rng.normal(0, s, n)  # noqa: E731
    logins = np.clip(np.round(np.exp(1.5 + 0.6 * latent + noise(0.4))), 0, None)
    feature_depth = np.clip(2 + 1.5 * latent + noise(1.0), 0, 10)
    support_tickets = np.clip(np.round(2 - 1.2 * latent + noise(1.2)), 0, None)
    nps = np.clip(np.round(7 + 2.0 * latent + noise(2.0)), 0, 10)
    payment_delay_days = np.clip(np.round(5 - 3.0 * latent + noise(4.0)), 0, None)

    # churn observed after the indicator window, driven by latent health
    churn_logit = -1.2 - 1.8 * latent + rng.normal(0, 0.8, n)
    churned = (rng.uniform(size=n) < 1 / (1 + np.exp(-churn_logit))).astype(int)

    df = pd.DataFrame(
        {
            "logins": logins,
            "feature_depth": feature_depth,
            "support_tickets": support_tickets,
            "nps": nps,
            "payment_delay_days": payment_delay_days,
            "segment": segment,
            "month": month,
            "churned": churned,
            "latent_health": latent,
        }
    )

    z = lambda s: (s - s.mean()) / s.std(ddof=0)  # noqa: E731
    df["health_score"] = (
        z(df["logins"])
        + z(df["feature_depth"])
        - z(df["support_tickets"])
        + z(df["nps"])
        - z(df["payment_delay_days"])
    ) / 5

    if include_leak:
        df["renewal_meeting_declined"] = (
            (churned == 1) & (rng.uniform(size=n) < 0.95)
        ).astype(int) | ((churned == 0) & (rng.uniform(size=n) < 0.02)).astype(int)

    return df
