# A practitioner's guide to proxy metrics

*Constructing, validating, and using scores for things you cannot directly measure.*

Businesses constantly need a number for something nobody can observe directly: customer
health, lead quality, engagement, brand strength, account risk, demand, developer
productivity. Nobody can hand you a column called `customer_health`. So analysts build a
**proxy score** — a combination of observable signals that is *supposed* to stand in for the
hidden thing.

The trouble is that building the score is the easy part. The hard, usually-skipped part is
answering the question that decides whether the score is worth anything: **does this number
actually track the hidden construct, or have we just blended noise into something that looks
authoritative?** This guide is about answering that question rigorously.

It is deliberately broader than the `proxyscore` library. The goal is to give you a complete
mental model of the proxy-metric lifecycle — drawn from psychometrics, econometrics, and modern
ML monitoring — and then to show, honestly, **where the tool already helps, where you should
reach for a specialized package, and what is still on the roadmap.** Throughout, you'll see
callouts:

> 🔧 **In proxyscore:** a check the library performs today.
> 🧭 **Use a specialized tool:** something outside this library's scope.
> 🚧 **On the roadmap:** planned, not yet implemented.

For runnable, hands-on usage, see the [Getting started guide](getting-started.md). This
document is the *why* behind it.

---

## Contents

1. [The problem: latent constructs](#1-the-problem-latent-constructs)
2. [Foundations: reflective vs. formative](#2-foundations-reflective-vs-formative)
3. [Construction: turning indicators into a score](#3-construction-turning-indicators-into-a-score)
4. [Validation: is the score real?](#4-validation-is-the-score-real)
5. [Bias: does it mean the same thing for everyone?](#5-bias-does-it-mean-the-same-thing-for-everyone)
6. [Monitoring: does it stay real?](#6-monitoring-does-it-stay-real)
7. [Decision-grade thinking](#7-decision-grade-thinking)
8. [References & further reading](#8-references--further-reading)

---

## 1. The problem: latent constructs

A **latent construct** is a real, consequential thing that has no direct measurement. "Customer
health" drives renewals, but no instrument reads it off an account. "Brand strength" commands
price premiums, but it lives in millions of heads. "Productivity" determines output, but the
input that produced it is unobserved.

What you *can* observe are **manifest indicators** — the measurable traces a construct leaves
behind: logins, support tickets, NPS responses, payment delays, invoice sizes. A proxy score is
a rule for combining manifest indicators into a single number that you hope rises and falls with
the latent construct.

Three things make this genuinely hard, and a discipline rather than a spreadsheet exercise:

- **You can never directly check your work.** Because the construct is unobserved, you cannot
  correlate your score against "true health." You can only validate against *observable
  consequences* the construct should drive (renewal, churn, expansion). This indirection is the
  central fact of the whole field — and the reason validation gets skipped, because it requires
  patience and a hard outcome you don't have yet.
- **Plausibility is not validity.** A weighted blend of sensible-sounding metrics will always
  *look* reasonable on a dashboard. Looking reasonable and carrying signal are unrelated
  properties. Most "health scores" in production have never been tested against an outcome.
- **The number gets used.** Once a proxy score drives a playbook — an automated alert, a
  renewal-risk queue, an account-suspension threshold — its errors become decisions. A score
  good enough for a directional dashboard can be dangerous as an automation trigger.

The fields that have wrestled with this longest — psychometrics (measuring intelligence,
anxiety, satisfaction), econometrics (measuring firm productivity), and credit risk (scoring
unobservable default propensity) — converge on the same lifecycle: **declare what kind of
construct you have, construct a score appropriate to it, validate it against reality, watch it
for drift, and match its proven sharpness to the weight of the decisions you hang on it.** The
rest of this guide walks that lifecycle.

---

## 2. Foundations: reflective vs. formative

Before any modeling, you must answer one question, and getting it wrong contaminates everything
downstream: **does the construct *cause* its indicators, or do the indicators *constitute* the
construct?**

### Reflective: the construct causes the indicators

In a **reflective** model, the latent variable is the common cause of its indicators. Change the
construct and all its indicators move together. Classic example: *brand trust* causes a
customer's stated likelihood to recommend, their rating of authenticity, and their willingness
to forgive a misstep — all at once. Because one cause drives them, reflective indicators are
expected to **correlate strongly** and are largely **interchangeable** (drop one and the
construct's meaning survives). This is the world survey instruments live in.

### Formative: the indicators constitute the construct

In a **formative** (composite) model, the indicators are the *ingredients* — the construct is a
weighted combination of them. *Customer health* is **formed** by login frequency, support
burden, contract expansion, and sponsor engagement. These need not correlate at all: a spike in
support tickets has no mechanical reason to move login frequency. The indicators are **not
interchangeable** — drop "payment delays" and you've changed what the score *means*, not just
measured it more noisily.

| | Reflective (latent factor) | Formative (composite) |
| --- | --- | --- |
| **Causal direction** | construct → indicators | indicators → construct |
| **Indicators correlate?** | Yes, strongly (shared cause) | Not necessarily |
| **Interchangeable?** | Yes — drop one, meaning survives | No — each defines part of the scope |
| **Drop an indicator** | Lose a little precision | Change the construct |
| **Typical domain** | Survey/psychometric scales | Customer health, engagement, lead quality |

### Why this is the keystone decision

Most business proxy scores — health, engagement, loyalty, risk — are **formative**. And the most
common methodological error is to validate a formative composite with reflective tools. If you
run Cronbach's alpha or a factor analysis on a formative score and demand high inter-item
correlation, you will "discover" that your perfectly good health score is "unreliable" and
start deleting the very indicators that give it scope. Misclassifying a formative construct as
reflective introduces severe parameter bias and leads to flatly wrong conclusions.

> 🔧 **In proxyscore:** the indicator-quality check computes reflective-style diagnostics
> (Cronbach's alpha, item-rest correlation) because they're genuinely useful *if* your construct
> is reflective — but it labels every such finding "(fine if formative)" and its notes spell out
> that low values are not a defect for a composite. The tool reports the diagnostic and leaves
> the reflective/formative judgment to you, because only you know your construct. Make that call
> *before* you read the report.

---

## 3. Construction: turning indicators into a score

How you build the score should follow from the section above. There is a spectrum from
pragmatic to academic.

### The industry scorecard (formative, by hand)

The dominant commercial approach — popularized by customer-success platforms — is a weighted,
normalized scorecard. The recipe:

1. **Segment first.** Score enterprise and SMB accounts on different models; the same login rate
   means different things at different scales.
2. **Normalize** each indicator onto a common scale (z-score, min-max, or percentile rank) so a
   login count and a ticket age are comparable.
3. **Weight** by importance — a deep integration milestone counts more than a single login — and
   use **negative weights** for reverse-oriented indicators (more tickets = less health).
4. **Aggregate** into one index, then **band** it into color-coded risk tiers that trigger
   playbooks.
5. **Refine quarterly** against actual renewal/churn outcomes.

Customer-success frameworks typically organize indicators into five families — **behavioral**
(adoption, login frequency), **support** (ticket trends, resolution time), **relationship**
(sponsor alignment, QBR cadence), **financial** (payment timeliness, expansion), and
**feedback** (NPS, CSAT) — which is a useful checklist for whether your indicator set has blind
spots.

> 🔧 **In proxyscore:** `CompositeScore` implements exactly this — normalize (`scaling="zscore"`,
> `"minmax"`, or `"rank"`), apply weights (negative for reverse-oriented indicators), and
> aggregate. It uses a `fit`/`transform` API so the normalization learned on a development sample
> applies unchanged to later periods — which is what keeps over-time monitoring honest (you're
> comparing scores on one fixed ruler, not re-deriving the ruler each month). It refuses to
> invent numbers from missing data: partial rows are renormalized over the weights actually
> present and fall to `NaN` below a coverage floor.

### Data-driven weighting (PCA)

If you believe one dominant dimension runs through your standardized indicators, the first
principal component gives you data-driven weights instead of hand-set ones. This sits between
formative and reflective: it assumes shared variance worth extracting.

> 🔧 **In proxyscore:** `PCAScore` fits the first principal component, aligns its sign so "higher
> = more construct," and exposes `explained_variance_ratio_` and per-indicator `loadings_`. It
> refuses to fit when no indicator varies (there is no direction to learn) and returns `NaN` for
> any incomplete row rather than projecting a partial vector onto a different scale.

### The academic and econometric end

For completeness, two heavier families exist, and it's worth knowing when you've outgrown a
scorecard:

- **Structural Equation Modeling (SEM).** Factor-based (CB-SEM) for confirming theoretical
  structure; composite-based (PLS-SEM) for predictive composites that produce an explicit score
  per record. This is the right tool when you have a measurement *theory* to test, multiple
  interrelated constructs, and (often) survey data.
- **Econometric control functions.** When the unobserved construct creates *endogeneity* — e.g.,
  estimating firm productivity, which is correlated with the firm's own input choices — methods
  like Olley–Pakes (proxy productivity with investment), Levinsohn–Petrin (proxy with
  intermediate inputs), and Ackerberg–Caves–Frazer invert an observable behavior to back out the
  latent variable and remove "transmission bias." This is the right tool when you're estimating a
  *causal* effect, not building a monitoring score.

> 🧭 **Use a specialized tool:** for full SEM, reach for `semopy` (Python) or `lavaan` /
> SmartPLS. For production-function estimation, use an econometrics package. `proxyscore`
> deliberately targets the messy behavioral/transactional data of real business stacks, where
> SEM's distributional assumptions rarely hold — it is not, and does not try to be, an SEM
> package.

---

## 4. Validation: is the score real?

This is the heart of the discipline. "Validity" is not one property; it's a stack of distinct
questions. Establishing each one closes off a different way your score could be fooling you.

### 4a. Indicator quality and reliability

Before asking whether the *score* is good, check whether the *ingredients* are. Dead columns
(zero variance), heavy missingness, and near-duplicate indicators all corrupt a composite.

- **Cronbach's alpha** — internal-consistency reliability; the standard threshold is **α > 0.7**.
  *Reflective constructs only* (it expects indicators to covary).
- **Item-rest correlation** — how each indicator relates to the rest. Low values flag an
  indicator that isn't pulling with the others. *Reflective only.*
- **Variance Inflation Factor (VIF)** — collinearity; a common flag is **VIF > 10**. Two
  indicators that are near-duplicates inflate each other and add no independent signal.
- **Single-indicator dominance** — if the "composite" correlates ~1.0 with one of its inputs,
  it isn't a composite; it's that one metric in disguise.

> 🔧 **In proxyscore:** the indicator-quality check reports missingness, zero-variance failures,
> item-rest correlation, Cronbach's alpha, VIF, near-duplicate pairs, and single-indicator
> dominance in one table — with the reflective-vs-formative caveats attached so you don't
> "fix" a healthy formative score.

### 4b. Convergent and discriminant validity

For **reflective** constructs (and multi-construct models), two classic checks:

- **Convergent validity** — do indicators of the same construct agree? Measured by **Average
  Variance Extracted (AVE)**; the rule is **AVE > 0.5** (the construct explains more than half
  its indicators' variance).
- **Discriminant validity** — is this construct empirically *distinct* from your other
  constructs, or are your "health" and "engagement" scores secretly the same thing? The field
  has **retired** the older Fornell–Larcker criterion and cross-loadings (Monte Carlo studies
  show they routinely miss real collinearity) in favor of the **Heterotrait–Monotrait (HTMT)
  ratio**: flag a problem when **HTMT > 0.90** (distinct constructs) or **> 0.85** (conceptually
  similar ones), ideally with a bootstrap confidence interval — if it contains 1.0, the
  constructs aren't distinct.

> 🚧 **On the roadmap:** AVE and HTMT for multi-construct setups are planned but not yet in the
> library. Today, if you maintain several related scores (health, engagement, risk) and want to
> prove they measure different things, you'd compute HTMT with a dedicated SEM/PLS tool. This is
> high on the list precisely because "are my five scores actually five things?" is a question
> every mature scoring program eventually faces.

### 4c. Criterion validity — the keystone

Because the construct is unobservable, the validation that matters most is against its
**delayed hard outcomes**: real, observable consequences the construct is supposed to drive.

- **Predictive validity** — does the score forecast the outcome? A health score must predict
  renewal/churn; a lead score must predict conversion. Quantify with AUC (binary outcomes) or
  rank correlation (continuous), plus lift/capture by score band.
- **Nomological validity** — does the score relate to *other* constructs in theoretically
  expected directions? Brand strength should correlate *positively* with price premiums and
  *negatively* with price sensitivity. A score that points the wrong way against a known
  relationship is suspect even if its headline AUC looks fine.

The non-negotiable discipline: **the outcome must be observed *after* the window the indicators
were measured from.** Validate a score against an outcome from the same window and you've built
a time machine, not a predictor.

> 🔧 **In proxyscore:** downstream validation is the library's signature and the **gate** to any
> non-trivial verdict. It computes oriented AUC or rank correlation, detects polarity
> automatically (a health score predicting *less* churn is handled correctly), reports per-class
> counts so a near-perfect AUC on three events is skipped rather than trusted, and produces
> lift/capture tables. Nomological validity (relating to *other* constructs) isn't a built-in
> check — you'd assert those relationships yourself — but the predictive half is the core of the
> audit.

### 4d. Leakage — the failure that masquerades as success

The most dangerous validation failure is the one that produces *spectacular* numbers. **Leakage**
is when an indicator is secretly a downstream echo of the outcome — a field populated only
*after* the customer has effectively decided to churn, or a flag a salesperson sets *because*
they've judged a lead dead. Such a score validates beautifully on history and predicts nothing
going forward, because in production the leaky signal isn't available until it's too late.

> 🔧 **In proxyscore:** the leakage check flags indicators whose standalone association with the
> outcome is implausibly strong, and indicators whose *names* look outcome-derived ("renewal",
> "churn", "closed_won"). A single leakage failure drops the whole verdict to `not_validated` —
> it's a disqualifier, not a warning you weigh. But the tool is honest that this is a heuristic:
> the only hard guarantee is a pipeline that snapshots indicators strictly before the outcome
> window opens.

---

## 5. Bias: does it mean the same thing for everyone?

A score can be valid *on average* and still be broken *within a group*. Two distinct questions:

- **Does the score mean the same thing across groups?** This is **measurement invariance**,
  tested hierarchically: **configural** (same indicators load on the construct everywhere) →
  **metric/weak** (same loadings — same conceptual meaning and scale) → **scalar/strong** (same
  intercepts — group differences in the score reflect true differences, not response bias) →
  **strict** (same residual variances). You need at least scalar invariance before comparing
  scores across segments at face value.
- **Does the score work equally well in every group?** A score that predicts the outcome sharply
  for SMB but is noise for enterprise will quietly misallocate attention — every automated
  decision in the broken segment is arbitrary.

> 🔧 **In proxyscore:** the segment check audits two things directly. **Score level** —
> standardized mean differences across segments (a large gap isn't automatically bias; verify it
> matches the segments' real outcome rates). **Score validity** — per-segment predictive strength,
> so a score that works in one segment and fails in another is caught rather than averaged away.
> Segments too small to assess are flagged, not silently dropped.
>
> 🚧 **On the roadmap:** full measurement-invariance testing (configural→strict) is planned. The
> current segment check is a pragmatic stand-in that catches the failures that matter operationally
> (level gaps, validity divergence) without the full SEM apparatus.

---

## 6. Monitoring: does it stay real?

A validated score is not a finished artifact; it's a live asset that degrades. User behavior
shifts, the product changes, the market moves. Two kinds of decay matter.

### Distribution drift (PSI)

The **Population Stability Index** measures how much a score's *distribution* has moved between a
baseline period and now. Bin the baseline (deciles are typical), compute the proportion of each
population in each bin, and sum `(p_now − p_base) · ln(p_now / p_base)` across bins. The standard
governance bands:

| PSI | Interpretation | Action |
| --- | --- | --- |
| **< 0.10** | Insignificant change | None — stable |
| **0.10 – 0.25** | Moderate shift | Alert; review weights / recalibrate bands |
| **≥ 0.25** | Significant, unstable shift | High risk; retrain and re-validate |

Why it matters operationally: a threshold you set in January ("flag anyone below 40") means
something different by June if the distribution has drifted — the same cutoff now selects a
different slice of the population. **Two honest limitations:** PSI is sensitive to bin count
(too many bins manufacture spurious instability), and it cannot tell benign seasonality from
real degradation — compare like-for-like periods.

> 🔧 **In proxyscore:** the stability check computes PSI per period against a baseline with these
> exact bands, guards against undersized periods (where PSI is too noisy to trust), and reports
> the worst period. Because the score's normalization is fit once and reused, the PSI reflects
> real population movement rather than a moving ruler.

### Goodhart's Law and "green churn"

The subtler decay is human. **Goodhart's Law:** *"When a measure becomes a target, it ceases to
be a good measure."* Once a score is visible and consequential, people optimize the score
instead of the underlying health. The canonical failure is **green churn**: an account sits
comfortably in the green zone because junior staff log in daily out of habit — while the
executive sponsor who championed the purchase has quietly left. The score over-weights activity
volume and under-weights relationship decay, so it reports "healthy" right up until the
surprise non-renewal.

There's no statistic that fully defends against this; it's a design discipline:

- **Balance quantitative with structured qualitative signal** — e.g., a standardized CSM pulse,
  not unstructured opinion.
- **Detect drift, not just thresholds** — watch for slow declines (a power user disengaging)
  that absolute cutoffs miss.
- **Anchor to outcomes, not activity** — tie the score to value delivered, using frameworks like
  SPACE or DORA as templates for balancing raw activity against durable health.

> 🔧 **In proxyscore:** the tool gives you the *instruments* that make green churn detectable —
> per-segment validity (is the score still predictive, or has it decoupled from outcomes?),
> downstream re-validation against fresh outcomes, and leakage/dominance checks that flag a score
> leaning too hard on a single gameable activity metric. The *governance* — keeping the metric
> honest once it's a target — is organizational, and no library substitutes for it.

---

## 7. Decision-grade thinking

The final idea ties the whole lifecycle together: **match a score's proven sharpness to the
weight of the decision you hang on it.** The same score can be entirely appropriate for one use
and reckless for another.

- A score good enough to **sort a dashboard** or **rank a CSM's outreach list** needs only
  directional signal. If it's right on average, it saves time even when individual records are
  noisy.
- A score that **automatically suspends an account, denies a renewal discount, or pages a
  human** is making per-record decisions. There, an individual misclassification is a wrong
  action, so the score needs proven, sharp, validated signal — and you need to know it holds in
  every segment you'll apply it to.

Treating these the same is how a "directional dashboard metric" quietly becomes an automation
trigger and starts making confident mistakes. The discipline is to state, explicitly, what tier
of decision a score has *earned* the right to drive.

> 🔧 **In proxyscore:** this is exactly what the audit verdict encodes:
>
> - **`decision_grade`** — strong validated downstream signal, no failures, no unresolved
>   warnings: cleared for per-record decisions *within the validated population and time
>   horizon*.
> - **`directional`** — real but moderate signal, or a strong signal with unresolved warnings:
>   good for dashboards and triage, not for automated per-record action.
> - **`not_validated`** — a check failed, or there was no outcome to validate against: an
>   untested hypothesis, not a measurement.
>
> The verdict is intentionally conservative — *any* unresolved warning, or any supplied check
> that couldn't be assessed, caps it below decision-grade. The point isn't to hand out a passing
> grade; it's to make the score state plainly what it has earned.

---

## 8. References & further reading

The methods above draw on three literatures. These are the load-bearing sources behind the
concepts and thresholds in this guide.

**Latent constructs, reflective vs. formative, and SEM**

- m-clark, *Graphical & Latent Variable Modeling* — accessible treatment of latent variables and
  measurement invariance.
- *Structural Equation Modeling with Latent Variables and Composites* (arXiv:2508.06112) —
  reflective vs. formative specification and the bias from misclassifying them.

**Construct validity (convergent, discriminant, HTMT)**

- Fornell & Larcker (1981) — the AVE-based criterion (now considered insufficient on its own).
- Henseler, Ringle & Sarstedt (2015) — the **HTMT** ratio for discriminant validity; thresholds
  0.85 / 0.90 and the bootstrap procedure.
- Ab Hamid et al. (2017), *Discriminant Validity Assessment: Fornell-Larcker vs. HTMT* — the
  comparison and why HTMT is the modern standard.

**Econometric proxy estimation**

- Olley & Pakes (1996) — investment as a proxy for unobserved productivity; control-function
  inversion.
- Levinsohn & Petrin (2003) — intermediate inputs as a smoother proxy.
- Ackerberg, Caves & Frazer — collinearity corrections; production functions with fixed effects.

**Production monitoring and metric governance**

- Burke, *Population Stability Index* — PSI computation and the 0.10 / 0.25 governance bands.
- Coralogix / Encord, *A Practical Introduction to PSI* — binning procedure and limitations.
- *Goodhart's Law: The Hidden Risk in Software Engineering Metrics* (Axify) — Goodhart's Law,
  green churn, and the SPACE/DORA framing.

**Industry customer-health practice**

- Gainsight, Vitally, Realm — the five-category scorecard model (behavioral, support,
  relationship, financial, feedback) and the construction/refinement pipeline.

---

*This guide describes the full proxy-metric lifecycle; `proxyscore` implements a growing portion
of it. For what the library does today, see the [README](../README.md) and the
[Getting started guide](getting-started.md); for what's planned, see the roadmap in the README.
The ambition is for the tool to take on more of this terrain — convergent/discriminant validity,
measurement invariance, and loading-drift monitoring among them — over successive releases.*
