import pytest

from proxyscore import ProxyAudit, Status, Verdict
from proxyscore.datasets import make_customer_health

INDICATORS = ["logins", "feature_depth", "support_tickets", "nps", "payment_delay_days"]


@pytest.fixture(scope="module")
def df():
    return make_customer_health(n=3000, seed=7)


def run_audit(df, **overrides):
    kwargs = dict(
        indicators=df[INDICATORS],
        score=df["health_score"],
        outcome=df["churned"],
        segments=df["segment"],
        period=df["month"],
    )
    kwargs.update(overrides)
    return ProxyAudit(**kwargs).run()


def test_full_audit_on_healthy_data(df):
    report = run_audit(df)
    assert report.verdict in (Verdict.DECISION_GRADE, Verdict.DIRECTIONAL)
    assert report["downstream"].status is Status.PASS
    assert report["leakage"].status is Status.PASS
    assert report["stability"].status is Status.PASS
    summary = report.summary()
    assert set(summary["check"]) == {"indicators", "stability", "downstream", "leakage", "segments"}


def test_audit_without_outcome_is_not_validated(df):
    report = run_audit(df, outcome=None)
    assert report.verdict is Verdict.NOT_VALIDATED
    assert report["downstream"].status is Status.SKIP


def test_audit_builds_score_when_missing(df):
    report = run_audit(df, score=None)
    # equal-weight composite includes negatively-oriented indicators unflipped,
    # so it should still run end to end and report something coherent
    assert isinstance(report.verdict, Verdict)
    assert any("equal-weight" in n for n in report["indicators"].notes)


def test_audit_detects_planted_leak():
    df = make_customer_health(n=3000, seed=8, include_leak=True)
    cols = INDICATORS + ["renewal_meeting_declined"]
    report = ProxyAudit(
        indicators=df[cols], score=df["health_score"], outcome=df["churned"]
    ).run()
    assert report["leakage"].status is Status.FAIL
    assert report.verdict is Verdict.NOT_VALIDATED


def test_audit_detects_planted_drift():
    df = make_customer_health(n=4000, seed=9, drift=2.0)
    report = ProxyAudit(
        indicators=df[INDICATORS],
        score=df["health_score"],
        outcome=df["churned"],
        period=df["month"],
    ).run()
    assert report["stability"].status in (Status.WARN, Status.FAIL)


def test_markdown_report(df):
    report = run_audit(df)
    md = report.to_markdown()
    assert "# Proxy score audit" in md
    assert report.verdict.value in md
    for check in ("indicators", "stability", "downstream", "leakage", "segments"):
        assert f"## {check}" in md


def test_report_getitem_unknown_raises(df):
    report = run_audit(df)
    with pytest.raises(KeyError):
        report["nope"]


def test_index_alignment_with_nondefault_index(df):
    shifted = df.copy()
    shifted.index = range(1000, 1000 + len(df))
    report = run_audit(shifted)
    assert report["downstream"].status is Status.PASS


def test_health_score_actually_predicts_churn(df):
    # sanity-check the synthetic data itself: oriented AUC should be strong
    report = run_audit(df)
    assert report["downstream"].metrics["auc_oriented"] > 0.7
    assert report["downstream"].metrics["polarity"] == -1  # higher health -> less churn


def test_audit_rejects_non_dataframe():
    with pytest.raises(TypeError):
        ProxyAudit(indicators=[[1, 2], [3, 4]])


def test_audit_warns_small_sample():
    df = make_customer_health(n=50, seed=10)
    report = ProxyAudit(
        indicators=df[INDICATORS], score=df["health_score"], outcome=df["churned"]
    ).run()
    assert report["sample_size"].status is Status.WARN
    assert report.verdict in (Verdict.DIRECTIONAL, Verdict.NOT_VALIDATED)


def test_audit_downsamples_large_sample():
    from proxyscore.config import Thresholds
    df = make_customer_health(n=1000, seed=11)

    # Test setting max_audit_rows lower than n
    thresholds = Thresholds(max_audit_rows=500)
    report = ProxyAudit(
        indicators=df[INDICATORS],
        score=df["health_score"],
        outcome=df["churned"],
        thresholds=thresholds
    ).run()

    assert report["sample_size"].status is Status.PASS
    assert "downsampled 1000 rows to 500" in report["sample_size"].summary
    assert report["sample_size"].metrics["audit_rows"] == 500

    # Check that score and outcome were also correctly aligned to the subset
    # which is implicit since check_downstream would fail if indices didn't match.
    assert report["downstream"].status is Status.PASS
