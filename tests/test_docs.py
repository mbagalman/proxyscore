"""Guards that keep the documentation honest.

These run under the normal pytest suite (and therefore in CI), so the
numbers, statuses, and links shown in the docs cannot silently drift away
from the implementation without a test failing.

Covers:
- the canonical metrics/statuses quoted in docs/getting-started.md,
- that every ```python fence in the public docs is at least syntactically valid,
- that every relative Markdown link in the public docs resolves to a real file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from proxyscore import ProxyAudit, Status, Verdict, lift_table
from proxyscore.datasets import make_customer_health

REPO = Path(__file__).resolve().parent.parent
PUBLIC_DOCS = [
    REPO / "README.md",
    REPO / "docs" / "README.md",
    REPO / "docs" / "getting-started.md",
    REPO / "docs" / "proxy-metrics-guide.md",
]
IND = ["logins", "feature_depth", "support_tickets", "nps", "payment_delay_days"]


# --- the numbers the tutorial quotes -----------------------------------------


def test_healthy_audit_matches_docs():
    df = make_customer_health(n=3000, seed=7)
    report = ProxyAudit(
        indicators=df[IND],
        score=df["health_score"],
        outcome=df["churned"],
        segments=df["segment"],
        period=df["month"],
    ).run()

    assert report.verdict is Verdict.DIRECTIONAL
    assert report["indicators"].status is Status.WARN
    assert report["stability"].status is Status.PASS
    assert report["downstream"].status is Status.PASS
    assert report["leakage"].status is Status.PASS
    assert report["segments"].status is Status.PASS

    m = report["downstream"].metrics
    assert m["polarity"] == -1
    assert m["n_pos"] == 989
    assert m["n_neg"] == 2011
    assert round(m["auc_oriented"], 3) == 0.784
    assert round(m["spearman"], 3) == -0.462
    assert round(m["base_rate"], 3) == 0.330
    assert round(report["stability"].metrics["max_psi"], 3) == 0.023


def test_lift_table_band1_matches_docs():
    df = make_customer_health(n=3000, seed=7)
    lt = lift_table(df["health_score"], df["churned"], n_bands=10, ascending=True)
    assert round(lt.iloc[0]["outcome_rate"], 2) == 0.77
    assert round(lt.iloc[0]["lift"], 1) == 2.3


def test_leakage_failure_matches_docs():
    leak = make_customer_health(n=3000, seed=8, include_leak=True)
    report = ProxyAudit(
        indicators=leak[IND + ["renewal_meeting_declined"]],
        score=leak["health_score"],
        outcome=leak["churned"],
    ).run()
    assert report.verdict is Verdict.NOT_VALIDATED
    assert report["leakage"].status is Status.FAIL
    table = report["leakage"].details.set_index("indicator")
    assert round(table.loc["renewal_meeting_declined", "association"], 2) == 0.97


def test_drift_failure_matches_docs():
    drift = make_customer_health(n=4000, seed=9, drift=2.0)
    report = ProxyAudit(
        indicators=drift[IND],
        score=drift["health_score"],
        outcome=drift["churned"],
        period=drift["month"],
    ).run()
    assert report.verdict is Verdict.NOT_VALIDATED
    assert report["stability"].status is Status.FAIL
    assert round(report["stability"].metrics["max_psi"], 3) == 2.665


def test_no_outcome_skips_match_docs():
    df = make_customer_health(n=3000, seed=7)
    report = ProxyAudit(indicators=df[IND], score=df["health_score"]).run()
    assert report.verdict is Verdict.NOT_VALIDATED
    assert report["downstream"].status is Status.SKIP
    assert report["stability"].status is Status.SKIP
    assert report["segments"].status is Status.SKIP
    assert report["indicators"].status is Status.WARN  # indicator check still runs


# --- structural guards on the docs themselves --------------------------------


def _python_fences(text: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)


@pytest.mark.parametrize("doc", PUBLIC_DOCS, ids=lambda p: p.name)
def test_python_fences_are_syntactically_valid(doc):
    """Every ```python block must parse (templates with placeholders are fine —
    compile() checks syntax, not undefined names)."""
    for i, code in enumerate(_python_fences(doc.read_text(encoding="utf-8"))):
        compile(code, f"{doc.name}#py{i}", "exec")


_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


@pytest.mark.parametrize("doc", PUBLIC_DOCS, ids=lambda p: p.name)
def test_relative_links_resolve(doc):
    text = doc.read_text(encoding="utf-8")
    for target in _LINK.findall(text):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path_part = target.split("#", 1)[0]
        if not path_part:
            continue
        resolved = (doc.parent / path_part).resolve()
        assert resolved.exists(), f"{doc.name}: broken link -> {target}"
