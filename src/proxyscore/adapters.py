"""Small tabular input adapters for local audit recipes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import pandas as pd

TableFormat = Literal["csv", "parquet", "database", "memory"]
PathLike = str | Path


@dataclass(frozen=True)
class TableProvenance:
    """Provenance for one loaded tabular input."""

    name: str
    source: str
    format: TableFormat
    rows: int
    columns: tuple[str, ...]
    loaded_at: str
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterProvenance:
    """Provenance for a full adapter load."""

    adapter: str
    loaded_at: str
    tables: Mapping[str, TableProvenance]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Return one provenance row per loaded table."""
        return pd.DataFrame(
            [
                {
                    "table": item.name,
                    "source": item.source,
                    "format": item.format,
                    "rows": item.rows,
                    "columns": len(item.columns),
                    "loaded_at": item.loaded_at,
                }
                for item in self.tables.values()
            ]
        )


@dataclass(frozen=True)
class TabularData:
    """Pandas tables plus adapter provenance."""

    tables: Mapping[str, pd.DataFrame]
    provenance: AdapterProvenance

    def require(self, name: str) -> pd.DataFrame:
        """Return a named table or raise a clear error."""
        try:
            return self.tables[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.tables)) or "(none)"
            raise KeyError(f"missing table {name!r}; available tables: {available}") from exc

    def copy_tables(self) -> dict[str, pd.DataFrame]:
        """Return shallow copies of all loaded tables."""
        return {name: frame.copy() for name, frame in self.tables.items()}


@runtime_checkable
class TabularAdapter(Protocol):
    """Protocol for adapters that return local pandas tables and provenance."""

    def load(self) -> TabularData:
        """Load all configured tables."""


class LocalCSVAdapter:
    """Load one or more local CSV files as named tables."""

    def __init__(
        self,
        paths: PathLike | Mapping[str, PathLike],
        *,
        read_csv_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.paths = _normalize_paths(paths)
        self.read_csv_kwargs = dict(read_csv_kwargs or {})

    def load(self) -> TabularData:
        """Read configured CSV files."""
        loaded_at = _utc_now()
        tables: dict[str, pd.DataFrame] = {}
        provenance: dict[str, TableProvenance] = {}
        for name, path in self.paths.items():
            frame = pd.read_csv(path, **self.read_csv_kwargs)
            tables[name] = frame
            provenance[name] = _table_provenance(
                name,
                path,
                "csv",
                frame,
                loaded_at,
            )
        return TabularData(
            tables=tables,
            provenance=AdapterProvenance(
                adapter=self.__class__.__name__,
                loaded_at=loaded_at,
                tables=provenance,
            ),
        )


class LocalParquetAdapter:
    """Load one or more local Parquet files as named tables."""

    def __init__(
        self,
        paths: PathLike | Mapping[str, PathLike],
        *,
        read_parquet_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.paths = _normalize_paths(paths)
        self.read_parquet_kwargs = dict(read_parquet_kwargs or {})

    def load(self) -> TabularData:
        """Read configured Parquet files."""
        loaded_at = _utc_now()
        tables: dict[str, pd.DataFrame] = {}
        provenance: dict[str, TableProvenance] = {}
        for name, path in self.paths.items():
            frame = pd.read_parquet(path, **self.read_parquet_kwargs)
            tables[name] = frame
            provenance[name] = _table_provenance(
                name,
                path,
                "parquet",
                frame,
                loaded_at,
            )
        return TabularData(
            tables=tables,
            provenance=AdapterProvenance(
                adapter=self.__class__.__name__,
                loaded_at=loaded_at,
                tables=provenance,
            ),
        )


def _normalize_paths(paths: PathLike | Mapping[str, PathLike]) -> dict[str, Path]:
    if isinstance(paths, Mapping):
        if not paths:
            raise ValueError("paths must contain at least one table")
        return {name: Path(path) for name, path in paths.items()}

    path = Path(paths)
    return {path.stem: path}


def _table_provenance(
    name: str,
    path: Path,
    table_format: TableFormat,
    frame: pd.DataFrame,
    loaded_at: str,
) -> TableProvenance:
    return TableProvenance(
        name=name,
        source=str(path.resolve(strict=False)),
        format=table_format,
        rows=len(frame),
        columns=tuple(str(column) for column in frame.columns),
        loaded_at=loaded_at,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
