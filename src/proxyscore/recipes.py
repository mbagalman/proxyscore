"""Business recipes for preparing honest proxy-score audit inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pandas as pd

from .adapters import TabularData
from .alignment import AlignmentResult, MatchPolicy, align_delayed_outcomes


@dataclass(frozen=True)
class RecipeResult:
    """Prepared recipe output plus diagnostics needed for audit handoff."""

    recipe: BusinessRecipe
    alignment: AlignmentResult
    input_rows: Mapping[str, int]
    deduplicated_rows: Mapping[str, int]

    @property
    def data(self) -> pd.DataFrame:
        """Aligned snapshot rows, including censored rows."""
        return self.alignment.data

    def audit_inputs(self, *, include_censored: bool = False) -> dict[str, Any]:
        """Return keyword arguments for ``ProxyAudit``."""
        return self.alignment.audit_inputs(
            self.recipe.indicators,
            score_column=self.recipe.score,
            segment_column=self.recipe.segment,
            period_column=self.recipe.period,
            include_censored=include_censored,
        )

    def summary(self) -> pd.Series:
        """Return recipe preparation counts."""
        return pd.Series(
            {
                "recipe": self.recipe.name,
                "snapshot_input_rows": self.input_rows["snapshots"],
                "outcome_input_rows": self.input_rows["outcomes"],
                "snapshot_deduplicated_rows": self.deduplicated_rows["snapshots"],
                "outcome_deduplicated_rows": self.deduplicated_rows["outcomes"],
                "aligned_rows": self.alignment.diagnostics.aligned_rows,
                "matched_rows": self.alignment.diagnostics.matched_rows,
                "unmatched_rows": self.alignment.diagnostics.unmatched_rows,
                "censored_rows": self.alignment.diagnostics.censored_rows,
            },
            name=self.recipe.name,
        )


@dataclass(frozen=True)
class BusinessRecipe:
    """A reusable recipe for turning business tables into audit-ready rows."""

    name: str
    construct: str
    description: str
    indicators: tuple[str, ...]
    score: str
    entity: str
    snapshot_time: str
    outcome: str
    outcome_time: str
    default_horizon: str | timedelta | pd.Timedelta
    segment: str | None = None
    period: str | None = None
    snapshot_table: str = "snapshots"
    outcome_table: str = "outcomes"
    snapshot_updated_at: str | None = "updated_at"
    outcome_updated_at: str | None = "updated_at"
    no_outcome_value: Any = 0
    sql_example: str = ""

    def prepare(
        self,
        tables: Mapping[str, pd.DataFrame] | TabularData,
        *,
        as_of: Any,
        horizon: str | timedelta | pd.Timedelta | None = None,
        match: MatchPolicy = "first",
    ) -> RecipeResult:
        """Deduplicate inputs and align delayed outcomes for this recipe."""
        table_map = tables.tables if isinstance(tables, TabularData) else tables
        snapshots = table_map[self.snapshot_table]
        outcomes = table_map[self.outcome_table]

        snapshot_required = [
            self.entity,
            self.snapshot_time,
            self.score,
            *self.indicators,
            *([self.segment] if self.segment is not None else []),
            *([self.period] if self.period is not None else []),
        ]
        outcome_required = [self.entity, self.outcome_time, self.outcome]
        _require_columns(snapshots, snapshot_required, self.snapshot_table)
        _require_columns(outcomes, outcome_required, self.outcome_table)

        deduped_snapshots, snapshot_dropped = _deduplicate(
            snapshots,
            [self.entity, self.snapshot_time],
            self.snapshot_updated_at,
        )
        deduped_outcomes, outcome_dropped = _deduplicate(
            outcomes,
            [self.entity, self.outcome_time],
            self.outcome_updated_at,
        )
        alignment = align_delayed_outcomes(
            deduped_snapshots,
            deduped_outcomes,
            entity=self.entity,
            score_time=self.snapshot_time,
            outcome=self.outcome,
            outcome_time=self.outcome_time,
            horizon=self.default_horizon if horizon is None else horizon,
            as_of=as_of,
            match=match,
            no_outcome_value=self.no_outcome_value,
        )
        return RecipeResult(
            recipe=self,
            alignment=alignment,
            input_rows={"snapshots": len(snapshots), "outcomes": len(outcomes)},
            deduplicated_rows={"snapshots": snapshot_dropped, "outcomes": outcome_dropped},
        )


def list_business_recipes() -> tuple[BusinessRecipe, ...]:
    """Return the built-in business recipes."""
    return (
        customer_health_recipe(),
        lead_quality_recipe(),
        account_risk_recipe(),
    )


def get_business_recipe(name: str) -> BusinessRecipe:
    """Return a built-in recipe by name."""
    recipes = {recipe.name: recipe for recipe in list_business_recipes()}
    try:
        return recipes[name]
    except KeyError as exc:
        available = ", ".join(sorted(recipes))
        raise KeyError(f"unknown business recipe {name!r}; available recipes: {available}") from exc


def customer_health_recipe() -> BusinessRecipe:
    """Recipe for customer-health scores validated against delayed churn."""
    return BusinessRecipe(
        name="customer_health",
        construct="Customer health",
        description=(
            "Validate account-level health snapshots against churn events observed after "
            "the score window."
        ),
        indicators=(
            "logins_30d",
            "feature_depth",
            "support_tickets_30d",
            "nps",
            "payment_delay_days",
        ),
        score="health_score",
        entity="account_id",
        snapshot_time="snapshot_at",
        outcome="churned",
        outcome_time="churned_at",
        default_horizon="90d",
        segment="plan_tier",
        period="snapshot_month",
        sql_example=_CUSTOMER_HEALTH_SQL,
    )


def lead_quality_recipe() -> BusinessRecipe:
    """Recipe for lead-quality scores validated against delayed conversion."""
    return BusinessRecipe(
        name="lead_quality",
        construct="Lead quality",
        description=(
            "Validate lead scoring snapshots against downstream opportunity creation or "
            "conversion events."
        ),
        indicators=(
            "firmographic_fit",
            "engagement_score",
            "source_quality",
            "sales_activity_14d",
        ),
        score="lead_score",
        entity="lead_id",
        snapshot_time="scored_at",
        outcome="converted",
        outcome_time="converted_at",
        default_horizon="30d",
        segment="lead_source",
        period="snapshot_week",
        sql_example=_LEAD_QUALITY_SQL,
    )


def account_risk_recipe() -> BusinessRecipe:
    """Recipe for account-risk scores validated against delayed default."""
    return BusinessRecipe(
        name="account_risk",
        construct="Account risk",
        description=(
            "Validate portfolio risk snapshots against default or serious-delinquency "
            "events observed after the score window."
        ),
        indicators=(
            "utilization_rate",
            "days_past_due",
            "collateral_ratio",
            "covenant_breaches",
        ),
        score="risk_score",
        entity="account_id",
        snapshot_time="snapshot_at",
        outcome="defaulted",
        outcome_time="defaulted_at",
        default_horizon="180d",
        segment="portfolio",
        period="snapshot_month",
        sql_example=_ACCOUNT_RISK_SQL,
    )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], table_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{table_name} is missing required columns: {missing}")


def _deduplicate(
    frame: pd.DataFrame,
    keys: Sequence[str],
    updated_at: str | None,
) -> tuple[pd.DataFrame, int]:
    working = frame.copy()
    working["_recipe_source_order"] = range(len(working))
    sort_columns = [*keys, "_recipe_source_order"]
    temp_columns = ["_recipe_source_order"]
    if updated_at is not None and updated_at in working.columns:
        working["_recipe_updated_at"] = pd.to_datetime(working[updated_at], errors="raise")
        sort_columns = [*keys, "_recipe_updated_at", "_recipe_source_order"]
        temp_columns.append("_recipe_updated_at")

    deduped = (
        working.sort_values(sort_columns, kind="stable")
        .drop_duplicates(list(keys), keep="last")
        .drop(columns=temp_columns)
        .reset_index(drop=True)
    )
    return deduped, len(frame) - len(deduped)


_CUSTOMER_HEALTH_SQL = """
with snapshot_candidates as (
  select
    account_id,
    snapshot_at,
    date_trunc('month', snapshot_at) as snapshot_month,
    plan_tier,
    logins_30d,
    feature_depth,
    support_tickets_30d,
    nps,
    payment_delay_days,
    health_score,
    updated_at,
    row_number() over (
      partition by account_id, snapshot_at
      order by updated_at desc
    ) as rn
  from analytics.customer_health_snapshots
  where snapshot_at <= :as_of
),
snapshots as (
  select * from snapshot_candidates where rn = 1
),
outcome_candidates as (
  select
    account_id,
    churned_at,
    1 as churned,
    updated_at,
    row_number() over (
      partition by account_id, churned_at
      order by updated_at desc
    ) as rn
  from analytics.churn_events
  where churned_at <= :as_of
)
select * from snapshots;
-- Load outcome_candidates where rn = 1 as the outcomes table, then align with a 90d horizon.
""".strip()

_LEAD_QUALITY_SQL = """
with snapshot_candidates as (
  select
    lead_id,
    scored_at,
    date_trunc('week', scored_at) as snapshot_week,
    lead_source,
    firmographic_fit,
    engagement_score,
    source_quality,
    sales_activity_14d,
    lead_score,
    updated_at,
    row_number() over (
      partition by lead_id, scored_at
      order by updated_at desc
    ) as rn
  from marketing.lead_score_snapshots
  where scored_at <= :as_of
),
conversion_candidates as (
  select
    lead_id,
    converted_at,
    1 as converted,
    updated_at,
    row_number() over (
      partition by lead_id, converted_at
      order by updated_at desc
    ) as rn
  from sales.lead_conversions
  where converted_at <= :as_of
)
select * from snapshot_candidates where rn = 1;
-- Load conversion_candidates where rn = 1 as the outcomes table, then align with a 30d horizon.
""".strip()

_ACCOUNT_RISK_SQL = """
with snapshot_candidates as (
  select
    account_id,
    snapshot_at,
    date_trunc('month', snapshot_at) as snapshot_month,
    portfolio,
    utilization_rate,
    days_past_due,
    collateral_ratio,
    covenant_breaches,
    risk_score,
    updated_at,
    row_number() over (
      partition by account_id, snapshot_at
      order by updated_at desc
    ) as rn
  from risk.account_score_snapshots
  where snapshot_at <= :as_of
),
default_candidates as (
  select
    account_id,
    defaulted_at,
    1 as defaulted,
    updated_at,
    row_number() over (
      partition by account_id, defaulted_at
      order by updated_at desc
    ) as rn
  from risk.default_events
  where defaulted_at <= :as_of
)
select * from snapshot_candidates where rn = 1;
-- Load default_candidates where rn = 1 as the outcomes table, then align with a 180d horizon.
""".strip()
