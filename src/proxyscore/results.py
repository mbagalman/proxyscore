"""Result objects shared by all checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Status(str, Enum):
    """Outcome of a single check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"

    @property
    def symbol(self) -> str:
        """ASCII-safe marker (legacy Windows consoles choke on emoji)."""
        return {"pass": "[PASS]", "warn": "[WARN]", "fail": "[FAIL]", "skip": "[SKIP]"}[self.value]


_STATUS_ORDER = {Status.PASS: 0, Status.SKIP: 0, Status.WARN: 1, Status.FAIL: 2}


def worst(statuses: list[Status]) -> Status:
    """Return the most severe status in a list (PASS if empty)."""
    if not statuses:
        return Status.PASS
    return max(statuses, key=lambda s: _STATUS_ORDER[s])


@dataclass
class CheckResult:
    """Outcome of one validation check.

    Attributes
    ----------
    name:
        Machine-friendly check identifier, e.g. ``"stability"``.
    status:
        Overall :class:`Status` of the check.
    summary:
        One-paragraph human-readable explanation of the outcome.
    metrics:
        Flat dict of the headline numbers computed by the check.
    details:
        Optional DataFrame with the full per-indicator / per-segment /
        per-period breakdown.
    notes:
        Caveats and interpretation hints (e.g. "item-rest correlation is
        only meaningful for reflective constructs").
    """

    name: str
    status: Status
    summary: str
    metrics: dict[str, float] = field(default_factory=dict)
    details: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"CheckResult(name={self.name!r}, status={self.status.value!r}, "
            f"summary={self.summary!r})"
        )
