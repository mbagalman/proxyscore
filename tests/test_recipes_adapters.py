from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import pandas.testing as pdt
import pytest

from proxyscore import (
    LocalCSVAdapter,
    LocalParquetAdapter,
    TabularAdapter,
    TabularData,
    account_risk_recipe,
    customer_health_recipe,
    get_business_recipe,
    lead_quality_recipe,
    list_business_recipes,
)


def customer_snapshots() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account_id": ["a", "a", "b", "c"],
            "snapshot_at": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03"]
            ),
            "updated_at": pd.to_datetime(
                [
                    "2026-01-01 01:00",
                    "2026-01-01 02:00",
                    "2026-01-02 00:00",
                    "2026-01-03 00:00",
                ]
            ),
            "snapshot_month": ["2026-01", "2026-01", "2026-01", "2026-01"],
            "plan_tier": ["pro", "pro", "enterprise", "pro"],
            "logins_30d": [3, 10, 20, 30],
            "feature_depth": [1, 3, 5, 7],
            "support_tickets_30d": [8, 5, 2, 1],
            "nps": [1, 4, 7, 9],
            "payment_delay_days": [20, 10, 2, 0],
            "health_score": [10.0, 40.0, 75.0, 92.0],
        }
    )


def customer_outcomes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account_id": ["a", "a", "b"],
            "churned_at": pd.to_datetime(["2026-01-10", "2026-01-10", "2026-02-15"]),
            "updated_at": pd.to_datetime(
                ["2026-01-10 01:00", "2026-01-10 02:00", "2026-02-15 00:00"]
            ),
            "churned": [0, 1, 1],
        }
    )


def test_csv_adapter_loads_named_tables_with_provenance(tmp_path: Path):
    snapshots = customer_snapshots()
    outcomes = customer_outcomes()
    snapshot_path = tmp_path / "snapshots.csv"
    outcome_path = tmp_path / "outcomes.csv"
    snapshots.to_csv(snapshot_path, index=False)
    outcomes.to_csv(outcome_path, index=False)

    adapter = LocalCSVAdapter({"snapshots": snapshot_path, "outcomes": outcome_path})

    assert isinstance(adapter, TabularAdapter)
    data = adapter.load()

    assert isinstance(data, TabularData)
    assert set(data.tables) == {"snapshots", "outcomes"}
    assert data.provenance.adapter == "LocalCSVAdapter"
    assert data.provenance.tables["snapshots"].rows == len(snapshots)
    assert data.provenance.tables["outcomes"].format == "csv"
    assert data.provenance.summary()["table"].tolist() == ["snapshots", "outcomes"]


def test_parquet_adapter_conforms_without_requiring_live_services(monkeypatch: pytest.MonkeyPatch):
    captured: list[Path] = []
    expected = pd.DataFrame({"id": [1, 2], "score": [0.1, 0.2]})

    def fake_read_parquet(path: Path, **kwargs: object) -> pd.DataFrame:
        captured.append(path)
        assert kwargs == {"columns": ["id", "score"]}
        return expected

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    adapter = LocalParquetAdapter(
        {"snapshots": Path("snapshots.parquet")},
        read_parquet_kwargs={"columns": ["id", "score"]},
    )

    loaded = adapter.load()

    assert isinstance(adapter, TabularAdapter)
    assert captured == [Path("snapshots.parquet")]
    pdt.assert_frame_equal(loaded.require("snapshots"), expected)
    assert loaded.provenance.tables["snapshots"].format == "parquet"


def test_customer_health_recipe_deduplicates_and_aligns_delayed_outcomes():
    recipe = customer_health_recipe()
    result = recipe.prepare(
        {"snapshots": customer_snapshots(), "outcomes": customer_outcomes()},
        as_of="2026-05-01",
    )

    assert result.deduplicated_rows == {"snapshots": 1, "outcomes": 1}
    assert result.data["health_score"].tolist() == [40.0, 75.0, 92.0]
    assert result.data["aligned_outcome"].tolist() == [1, 1, 0]
    assert result.data["outcome_status"].tolist() == ["matched", "matched", "unmatched"]
    assert result.alignment.diagnostics.matched_rows == 2
    kwargs = result.audit_inputs()
    assert kwargs["indicators"].shape == (3, 5)
    assert kwargs["outcome"].tolist() == [1, 1, 0]


def test_customer_health_recipe_keeps_immature_rows_out_of_audit_inputs_by_default():
    result = customer_health_recipe().prepare(
        {"snapshots": customer_snapshots(), "outcomes": customer_outcomes()},
        as_of="2026-01-20",
    )

    assert result.data["outcome_status"].tolist() == ["matched", "censored", "censored"]
    assert len(result.audit_inputs()["outcome"]) == 1
    assert len(result.audit_inputs(include_censored=True)["outcome"]) == 3
    assert result.summary()["censored_rows"] == 2


def test_built_in_recipe_catalog_and_sql_examples_cover_business_patterns():
    recipes = list_business_recipes()

    assert {recipe.name for recipe in recipes} == {
        "customer_health",
        "lead_quality",
        "account_risk",
    }
    assert get_business_recipe("lead_quality") == lead_quality_recipe()
    assert get_business_recipe("account_risk") == account_risk_recipe()
    for recipe in recipes:
        sql = recipe.sql_example.lower()
        assert "row_number() over" in sql
        assert "partition by" in sql
        assert ":as_of" in sql
        assert "updated_at desc" in sql


def test_recipe_accepts_tabular_data_from_adapter(tmp_path: Path):
    snapshot_path = tmp_path / "snapshots.csv"
    outcome_path = tmp_path / "outcomes.csv"
    customer_snapshots().to_csv(snapshot_path, index=False)
    customer_outcomes().to_csv(outcome_path, index=False)
    loaded = LocalCSVAdapter({"snapshots": snapshot_path, "outcomes": outcome_path}).load()

    result = customer_health_recipe().prepare(loaded, as_of="2026-03-01")

    assert result.alignment.diagnostics.aligned_rows == 3
    assert loaded.provenance.tables["snapshots"].source.endswith("snapshots.csv")


def test_recipe_reports_missing_required_columns():
    snapshots = customer_snapshots().drop(columns=["health_score"])
    with pytest.raises(KeyError, match="health_score"):
        customer_health_recipe().prepare(
            {"snapshots": snapshots, "outcomes": customer_outcomes()},
            as_of="2026-03-01",
        )


def test_unknown_recipe_names_available_options():
    with pytest.raises(KeyError, match="customer_health"):
        get_business_recipe("missing")


def test_tabular_data_require_lists_available_tables(tmp_path: Path):
    path = tmp_path / "only.csv"
    pd.DataFrame({"id": [1]}).to_csv(path, index=False)
    data = LocalCSVAdapter(path).load()

    assert list(data.tables) == ["only"]
    with pytest.raises(KeyError, match="available tables: only"):
        data.require("snapshots")


def test_protocol_type_accepts_local_adapter(tmp_path: Path):
    path = tmp_path / "table.csv"
    pd.DataFrame({"id": [1]}).to_csv(path, index=False)
    adapter = cast(TabularAdapter, LocalCSVAdapter(path))

    assert adapter.load().provenance.adapter == "LocalCSVAdapter"
