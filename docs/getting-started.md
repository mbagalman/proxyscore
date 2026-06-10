# Getting started with proxyscore

This is a hands-on walkthrough: by the end you will have run a full audit, read the
verdict, interpreted every check, seen what failures look like, and pointed the tool at
your own data. Every snippet here is runnable, and every block of output is real — copied
from an actual run against the synthetic dataset that ships with the package.

If you just want the API surface at a glance, the [README](../README.md) has terse
reference snippets. This guide is the slower, narrative version that teaches *interpretation* —
the part that actually trips people up.

> **Prerequisites:** `pip install proxyscore` (Python ≥ 3.10; depends only on numpy, pandas,
> scipy). No data of your own is needed — we use the built-in synthetic dataset throughout.

---

## 1. Your first audit

The package ships with `make_customer_health()`, a synthetic dataset built around a *known*
latent construct ("customer health") so you can see the tool work before trusting it on your
own numbers. Each row is one account in one month.

```python
from proxyscore.datasets import make_customer_health

df = make_customer_health(n=3000, seed=7)
df.columns.tolist()
# ['logins', 'feature_depth', 'support_tickets', 'nps', 'payment_delay_days',
#  'segment', 'month', 'churned', 'latent_health', 'health_score']
```

The columns fall into four groups:

| Group | Columns | Role in the audit |
| --- | --- | --- |
| **Indicators** | `logins`, `feature_depth`, `support_tickets`, `nps`, `payment_delay_days` | The observable inputs the score is built from. |
| **The score** | `health_score` | The proxy we want to validate (a weighted composite). |
| **The outcome** | `churned` | A *delayed hard outcome* — did the account churn? This is what we validate the score against. |
| **Context** | `segment`, `month` | Used for segment-bias and over-time stability checks. |
| **(hidden)** | `latent_health` | The true construct. A real analyst never gets to see this; it is here only so the dataset has a known answer. |

Now run the full audit. You hand `ProxyAudit` the indicators, the score, and any of the
optional context (outcome, segments, period), then call `.run()`:

```python
from proxyscore import ProxyAudit

indicator_cols = ["logins", "feature_depth", "support_tickets", "nps", "payment_delay_days"]

report = ProxyAudit(
    indicators=df[indicator_cols],
    score=df["health_score"],
    outcome=df["churned"],
    segments=df["segment"],
    period=df["month"],
).run()

print(report.summary())
```

```text
     check status                                                                                                                   summary
indicators   warn 1 indicator(s) with |item-rest corr| < 0.1: ['nps'] (fine if formative); Cronbach's alpha -0.46 < 0.7 (fine if formative)
 stability   pass                                                        Score distribution stable across 6 graded periods (max PSI 0.023).
downstream   pass                                           Strong downstream signal: AUC 0.784 (989 positives / 2011 negatives) on n=3000.
   leakage   pass                                    No leakage signals: all 5 indicators assessed, none suspiciously close to the outcome.
  segments   pass                                                               Score levels and validity are consistent across 3 segments.
```

Five checks, each with a status (`pass` / `warn` / `fail` / `skip`) and a plain-language
summary. We'll unpack each one — but first, the bottom line.

---

## 2. Reading the verdict

The whole audit rolls up into a single verdict:

```python
print(report.verdict)
# Verdict.DIRECTIONAL

print(report.verdict_reason)
# strong downstream signal, but with warnings (indicators). Suitable for dashboards
# and prioritization; resolve these before automating decisions on it.
```

There are exactly three possible verdicts, and the difference between them is *what they
license you to do*:

| Verdict | What it means | What you can do with the score |
| --- | --- | --- |
| **`decision_grade`** | Strong validated signal, no failures, no unresolved warnings. | Drive per-record decisions: prioritization queues, automated alerts, account actions — within the validated population and time horizon. |
| **`directional`** | Real but moderate signal, *or* a strong signal with unresolved warnings. | Read trends and dashboards; rank and triage. **Not** safe for automated per-record action. |
| **`not_validated`** | A check failed, or there was no outcome to validate against. | Treat the score as an untested hypothesis, not a measurement. |

### Why is this audit `directional` and not `decision_grade`?

The downstream check passed strongly, nothing failed — but the **indicators** check raised a
warning, and any unresolved warning caps the verdict at `directional`. That is deliberate
conservatism: the tool will not hand you a decision-grade stamp while something is unexplained.

In this case the warning is benign (we'll see exactly why in the next section), but *you* have
to make that call and either accept it or address it. `decision_grade` requires the downstream
check to pass **and** zero warnings **and** no supplied check left unassessable. The bar is
high on purpose — decision-grade is a license to act on individual records automatically.

### Downstream validation is the gate

Notice that `not_validated` is the verdict whenever there's no outcome to check against. That's
the core philosophy: a proxy for something you can't observe can only be validated against its
*observable consequences*. A score that has never been confronted with a real outcome is, in
the README's phrase, "an opinion with units." Downstream validation is the gate to every
verdict above `not_validated`.

---

## 3. Reading each check

Every check is accessible by name on the report, and each carries machine-readable `metrics`,
a `details` DataFrame, and interpretation `notes`:

```python
report["downstream"].metrics       # the headline numbers
report["segments"].details         # the per-segment table
report["indicators"].notes         # caveats and how to read the result
```

### indicators — are the inputs healthy?

```python
print(report["indicators"].details)
```

```text
         indicator  missing_rate  n_unique      std  item_rest_corr      vif  score_corr_abs
            logins           0.0        41 4.919672        0.171639 1.791162        0.774376
     feature_depth           0.0      2541 1.602763        0.132552 2.184540        0.835311
   support_tickets           0.0         9 1.567846       -0.480745 1.561642        0.750993
               nps           0.0        11 2.516982        0.088982 1.633696        0.765210
payment_delay_days           0.0        22 4.320566       -0.417766 1.399701        0.706454
```

What each column tells you:

- **`missing_rate`** — share of rows where the indicator is absent. All clean here.
- **`std`** — a zero here would mean a dead, constant column (an automatic failure).
- **`item_rest_corr`** — how each indicator correlates with the average of the others. The
  warning fired because `nps` is near zero and the overall Cronbach's alpha is *negative*.
- **`vif`** — variance inflation factor; high values (default threshold 10) flag an indicator
  that's redundant with the others. All low here.
- **`score_corr_abs`** — how strongly the final score tracks each indicator; a value near 1.0
  would mean the "composite" is really just one indicator wearing a trenchcoat.

**This is the most important interpretation lesson in the tutorial.** Item-rest correlation and
Cronbach's alpha are *reflective* measures: they assume the construct **causes** the indicators,
so the indicators should move together. But `health_score` is a **formative** composite — the
indicators *define* health, and `support_tickets` and `payment_delay_days` are deliberately
reverse-oriented (more tickets = less healthy). They are *not* supposed to correlate positively
with logins and NPS. So a low or negative alpha here is expected and fine, which is exactly why
the check appends "(fine if formative)" and the notes spell it out:

```python
print(report["indicators"].notes[0])
# Item-rest correlation and Cronbach's alpha assume a reflective construct (indicators
# caused by the latent variable, expected to covary). For formative composites (indicators
# define the construct), low values are not a defect.
```

Knowing your construct is formative, you can read this warning as informational and move on —
the verdict stays `directional`, but you understand *why*.

### stability — does the score hold still over time?

```text
stability  pass  Score distribution stable across 6 graded periods (max PSI 0.023).
```

This uses the **Population Stability Index (PSI)**, comparing each period's score distribution
to a baseline period. The standard bands are baked in: **< 0.10** stable, **0.10–0.25** a
moderate shift worth reviewing, **≥ 0.25** a significant shift that means scores from different
periods are no longer comparable. Here the max PSI across six months is 0.023 — rock stable.

### downstream — does the score predict the outcome?

```python
print(report["downstream"].metrics)
# {'n': 3000, 'spearman': -0.462, 'polarity': -1, 'outcome_type': 'binary',
#  'auc': 0.216, 'auc_oriented': 0.784, 'base_rate': 0.330, 'n_pos': 989, 'n_neg': 2011}
```

The headline is `auc_oriented = 0.784`. Two things to understand:

- **Polarity is detected automatically.** A *health* score should predict *less* churn, so the
  raw relationship is negative (`polarity: -1`, raw `auc: 0.216`). The tool orients it for you
  so that `auc_oriented` is always "how much signal is there," regardless of direction. You
  never have to remember which way your score points.
- **Class counts are reported** (`n_pos`, `n_neg`). Binary validation needs enough of *both*
  classes; with 989 churners and 2011 non-churners, this AUC rests on plenty of events. (A
  near-perfect AUC computed on three events would be SKIPped, not trusted.)

You can also pull a decile **lift table** to see the signal concretely. With `ascending=True`,
band 1 holds the *lowest*-health accounts — and they churn at 77%, more than double the 33%
base rate (`lift` ≈ 2.3):

```python
from proxyscore import lift_table
lift_table(df["health_score"], df["churned"], n_bands=10, ascending=True)
```

```text
 band   n  score_min  score_max  outcome_rate     lift  cum_capture
    1 300  -2.238169  -0.967940      0.770000 2.335693     0.233569
    2 300  -0.966715  -0.676567      0.550000 1.668352     0.400404
    3 300  -0.676556  -0.423261      0.506667 1.536906     0.554095
    ...
```

`cum_capture` is the punchline for a triage queue: working just the worst two deciles (20% of
accounts) reaches 40% of all churners.

### leakage — is the score secretly built from the outcome?

```text
leakage  pass  No leakage signals: all 5 indicators assessed, none suspiciously close to the outcome.
```

Leakage is the single most common way a proxy score lies: an indicator that is really a
downstream echo of the outcome (a field only populated *after* the customer decided to churn).
Such a score validates beautifully and predicts nothing going forward. The check flags any
indicator whose standalone association with the outcome is implausibly strong, plus any
indicator whose *name* looks outcome-derived. Here all five indicators were assessed and none
is suspicious — we'll see a failure in the next section.

### segments — does the score mean the same thing for everyone?

```python
print(report["segments"].details)
```

```text
   segment    n  score_mean  score_std  smd_vs_rest  n_outcome  n_pos  n_neg  outcome_rate  validity
enterprise  612    0.243025   0.784145     0.403428        612    137    475      0.223856  0.779662
mid_market  837    0.053023   0.750698     0.096004        837    271    566      0.323775  0.779618
       smb 1551   -0.124508   0.741864    -0.341060       1551    581    970      0.374597  0.776299
```

Two distinct questions live here:

- **Score level** (`smd_vs_rest`, a standardized mean difference): enterprise accounts score
  higher (+0.40) and SMB lower (−0.34). That is *not* automatically bias — enterprise accounts
  may genuinely be healthier, and the `outcome_rate` column confirms it (enterprise churns at
  22%, SMB at 37%). The score level tracks reality.
- **Score validity** (`validity`, the oriented predictive strength *within* each segment): all
  three segments sit at ~0.78. The score works equally well everywhere. A score that predicted
  well for SMB but was noise for enterprise would quietly misallocate attention — that's the
  failure mode this column exists to catch.

---

## 4. When a check fails — and what to do

A passing audit is reassuring, but the tool earns its keep on the failures. Here are the two
most important ones.

### Leakage failure

The synthetic dataset can plant a leak: an indicator (`renewal_meeting_declined`) that is
almost a direct copy of the churn outcome.

```python
leak = make_customer_health(n=3000, seed=8, include_leak=True)

report = ProxyAudit(
    indicators=leak[indicator_cols + ["renewal_meeting_declined"]],
    score=leak["health_score"],
    outcome=leak["churned"],
).run()

print(report.verdict)
# Verdict.NOT_VALIDATED
print(report["leakage"].summary)
# indicator(s) with implausibly strong standalone association with the outcome:
# renewal_meeting_declined (0.97) - likely leakage (indicator measured after, or
# defined by, the outcome)
```

A single failed check drops the whole verdict to `not_validated` — leakage isn't a warning you
weigh, it's a disqualifier. **What to do:** trace the offending indicator back to its source
and confirm its timing. If `renewal_meeting_declined` is only set *after* an account signals
it's leaving, it cannot be an input to a score meant to *predict* leaving. Remove it (or
re-derive it from a strictly pre-outcome snapshot) and re-audit. The tool's notes say it
plainly: the only hard guarantee against leakage is a pipeline that snapshots indicators
strictly before the outcome window opens.

### Stability failure (drift)

The dataset can also inject month-over-month drift:

```python
drift = make_customer_health(n=4000, seed=9, drift=2.0)

report = ProxyAudit(
    indicators=drift[indicator_cols],
    score=drift["health_score"],
    outcome=drift["churned"],
    period=drift["month"],
).run()

print(report["stability"].summary)
# Significant distribution shift: PSI reached 2.665 in period '2025-06' (threshold 0.25).
# Scores from different periods are not comparable; recalibrate before using thresholds or trends.
```

**What to do:** a PSI this high means a threshold you set in January (say, "flag anyone below
40") means something completely different by June — the same number now corresponds to a
different slice of the population. Either recalibrate your bands per period, or investigate
whether the shift is a real business change versus a measurement artifact (a new logging
pipeline, a definition change). Seasonal businesses can show benign PSI spikes; compare
like-for-like periods if so.

### No outcome at all

If you can't supply an outcome yet, the audit is honest about it rather than implying success:

```python
report = ProxyAudit(indicators=df[indicator_cols], score=df["health_score"]).run()

print(report.verdict)         # Verdict.NOT_VALIDATED
print(report["downstream"].status)  # Status.SKIP
```

The indicator, stability, and structural checks still run and are useful — but without a
delayed hard outcome, the score remains an untested hypothesis. Collect an outcome and re-audit.

---

## 5. Bringing your own data

Everything above works the same on your data. Three things to get right:

### Shape

One row per entity (or per entity-period if you're monitoring over time). Indicators go in a
DataFrame of numeric columns; the score, outcome, segment, and period are aligned columns or
arrays of the same length.

```python
report = ProxyAudit(
    indicators=mydf[["feature_a", "feature_b", "feature_c"]],
    score=mydf["my_score"],              # omit this to have an equal-weight one built for you
    outcome=mydf["renewed_within_90d"],  # binary or continuous; numeric, bool, or two-valued strings
    segments=mydf["plan_tier"],          # optional
    period=mydf["snapshot_month"],       # optional; any sortable labels
).run()
```

### The alignment contract

All inputs must line up row-for-row. If you pass pandas `Series`, they must carry the **same
index** as the indicators (same labels, same order); if you pass plain arrays, they must match
the indicators' length exactly. This is enforced — a mismatch raises a clear error rather than
silently comparing the wrong rows. The safest pattern is to pull every input as a column of one
DataFrame, as above.

### The temporal rule (read this twice)

**The outcome must be observed *after* the window the indicators were measured from.** This is
the single most common way to get a meaningless-but-impressive audit. If your "engagement
score" is computed from the same time window in which the outcome was realized, you have a
time machine, not a predictor — and the leakage check can only catch the blatant cases, not a
subtle whole-score timing error. Snapshot your indicators, *then* wait for the outcome.

### Don't have a score yet?

Build one. Two constructors are included, both with a `fit`/`transform` API so weights learned
on a development sample apply unchanged to later periods (which keeps stability monitoring
honest):

```python
from proxyscore import CompositeScore, PCAScore

# industry-scorecard style: normalize each indicator, weight, sum.
# negative weights for reverse-oriented indicators (more tickets = less healthy)
score = CompositeScore(
    weights={"logins": 1, "feature_depth": 1, "support_tickets": -1,
             "nps": 1, "payment_delay_days": -1},
    scaling="zscore",   # or "minmax", "rank"
).fit_transform(df[indicator_cols])

# data-driven alternative: first principal component of the standardized indicators
score = PCAScore().fit_transform(df[indicator_cols])
```

Both refuse to invent numbers from missing data: `CompositeScore` renormalizes a partial row
over the weights actually present and returns `NaN` below a coverage floor; `PCAScore` returns
`NaN` for any incomplete row. (If you omit `score` from `ProxyAudit` entirely, an equal-weight
z-score composite is built for you and the report says so.)

### Tightening the thresholds

Every cutoff is overridable through `Thresholds`. If your use case needs a sharper bar before
acting:

```python
from proxyscore import Thresholds

strict = Thresholds(
    min_auc_strong=0.75,   # demand a stronger downstream signal
    psi_unstable=0.20,     # alarm on smaller distribution shifts
    leak_auc=0.85,         # be more suspicious of outcome-like indicators
)
report = ProxyAudit(indicators=X, score=s, outcome=y, thresholds=strict).run()
```

---

## 6. Exporting the report

For a pull request, a wiki page, or a model card, render the whole thing to Markdown:

```python
print(report.to_markdown())
```

It produces a verdict line, a status table, and a per-check section with metrics, the details
table, and interpretation notes — the same content you've explored here, in a single
copy-pasteable document.

---

## Where to go next

- **[README](../README.md)** — quick reference, the "what this library deliberately is not"
  section (how it differs from SEM packages and ML-observability tools), and the validation
  philosophy in brief.
- **[CHANGELOG](../CHANGELOG.md)** — what's in each release.
- **Roadmap** (in the README) — convergent/discriminant validity (AVE, HTMT), measurement
  invariance testing, loading-drift monitoring, and more.

Found a rough edge or have a construct the tool handles awkwardly? Issues and contributions are
welcome.
