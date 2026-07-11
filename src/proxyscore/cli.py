"""Command-line workflows for audits, comparisons, baselines, and monitoring."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pandas as pd

from .audit import ProxyAudit, Verdict
from .comparison import ScoreComparison, compare_scores
from .config import Thresholds
from .monitoring import (
    ArtifactVersionError,
    MonitoringBaseline,
    MonitoringLimits,
    MonitorStatus,
    create_monitoring_baseline,
    monitor_batch,
)

CONFIG_VERSION = 1
EXIT_SUCCESS = 0
EXIT_WARNING = 1
EXIT_VALIDATION_FAILED = 2
EXIT_BAD_INPUT = 3
EXIT_INTERNAL_ERROR = 4


class CliInputError(ValueError):
    """Expected configuration, file, or column error."""


def _load_toml(path: str | Path) -> dict[str, Any]:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10 only
        import tomli as tomllib  # type: ignore[no-redef]

    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            values = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CliInputError(f"could not load TOML config {config_path}: {exc}") from exc
    if values.get("config_version") != CONFIG_VERSION:
        raise CliInputError(
            f"unsupported config_version {values.get('config_version')!r}; "
            f"expected {CONFIG_VERSION}"
        )
    return values


def _command_config(values: dict[str, Any], command: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for section in ("common", command, "output"):
        section_values = values.get(section, {})
        if not isinstance(section_values, dict):
            raise CliInputError(f"config section [{section}] must be a table")
        merged.update(section_values)
    return merged


def _option(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    name: str,
    default: Any = None,
) -> Any:
    value = getattr(args, name, None)
    return value if value is not None else config.get(name, default)


def _required(value: Any, name: str) -> Any:
    if value is None or value == "" or value == []:
        raise CliInputError(f"{name} is required (provide it on the command line or in config)")
    return value


def _read_table(path_value: Any) -> pd.DataFrame:
    path = Path(str(_required(path_value, "input")))
    if not path.exists():
        raise CliInputError(f"input file does not exist: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in (".parquet", ".pq"):
            return pd.read_parquet(path)
    except (OSError, ValueError, ImportError) as exc:
        if suffix in (".parquet", ".pq") and isinstance(exc, ImportError):
            raise CliInputError(
                "Parquet input requires an installed pandas Parquet engine; "
                "install proxyscore[parquet]"
            ) from exc
        raise CliInputError(f"could not read input file {path}: {exc}") from exc
    raise CliInputError("input must use .csv, .parquet, or .pq")


def _columns(frame: pd.DataFrame, names: Any, label: str) -> list[str]:
    if isinstance(names, str):
        resolved = [names]
    elif isinstance(names, Sequence) and not isinstance(names, (bytes, str)):
        resolved = [str(name) for name in names]
    else:
        raise CliInputError(f"{label} must be a column name or list of column names")
    if not resolved:
        raise CliInputError(f"{label} must contain at least one column")
    missing = [name for name in resolved if name not in frame.columns]
    if missing:
        raise CliInputError(f"input is missing {label}: {missing}")
    return resolved


def _column(frame: pd.DataFrame, name: Any, label: str, required: bool = False) -> pd.Series | None:
    if name is None:
        if required:
            raise CliInputError(f"{label} column is required")
        return None
    column = str(name)
    if column not in frame.columns:
        raise CliInputError(f"input is missing {label} column: {column}")
    return frame[column]


def _pairs(values: Sequence[str] | None, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise CliInputError(f"{label} must use KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key or not value:
            raise CliInputError(f"{label} must use non-empty KEY=VALUE")
        result[key] = value
    return result


def _coerce_dataclass_value(current: Any, value: Any, name: str) -> Any:
    if not isinstance(value, str):
        return value
    if current is None:
        if value.lower() == "none":
            return None
        try:
            return int(value)
        except ValueError as exc:
            raise CliInputError(f"{name} must be an integer or 'none'") from exc
    if isinstance(current, bool):
        lowered = value.lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        raise CliInputError(f"{name} must be true or false")
    if isinstance(current, int):
        try:
            return int(value)
        except ValueError as exc:
            raise CliInputError(f"{name} must be an integer") from exc
    if isinstance(current, float):
        try:
            return float(value)
        except ValueError as exc:
            raise CliInputError(f"{name} must be a number") from exc
    return value


def _thresholds(config_values: Any, overrides: Sequence[str] | None) -> Thresholds:
    if config_values is None:
        values: dict[str, Any] = {}
    elif isinstance(config_values, dict):
        values = dict(config_values)
    else:
        raise CliInputError("[thresholds] must be a TOML table")
    defaults = Thresholds()
    allowed = {item.name for item in fields(Thresholds)}
    for key, raw in _pairs(overrides, "--threshold").items():
        if key not in allowed:
            raise CliInputError(f"unknown threshold: {key}")
        values[key] = _coerce_dataclass_value(getattr(defaults, key), raw, key)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise CliInputError(f"unknown thresholds: {unknown}")
    try:
        return Thresholds(**values)
    except (TypeError, ValueError) as exc:
        raise CliInputError(f"invalid thresholds: {exc}") from exc


def _limits(config_values: Any, overrides: Sequence[str] | None) -> MonitoringLimits:
    if config_values is None:
        values: dict[str, Any] = {}
    elif isinstance(config_values, dict):
        values = dict(config_values)
    else:
        raise CliInputError("[monitoring_limits] must be a TOML table")
    defaults = MonitoringLimits()
    allowed = {item.name for item in fields(MonitoringLimits)}
    for key, raw in _pairs(overrides, "--limit").items():
        if key not in allowed:
            raise CliInputError(f"unknown monitoring limit: {key}")
        values[key] = _coerce_dataclass_value(getattr(defaults, key), raw, key)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise CliInputError(f"unknown monitoring limits: {unknown}")
    try:
        return MonitoringLimits(**values)
    except (TypeError, ValueError) as exc:
        raise CliInputError(f"invalid monitoring limits: {exc}") from exc


def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _json_document(value: Any) -> str:
    return json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False)


def _write(path_value: Any, document: str) -> None:
    if path_value is None:
        return
    path = Path(str(path_value))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document.rstrip() + "\n", encoding="utf-8")


def _emit(
    json_document: str,
    *,
    markdown: str | None,
    html_document: str | None,
    json_output: Any,
    markdown_output: Any,
    html_output: Any,
) -> None:
    _write(json_output, json_document)
    if markdown is not None:
        _write(markdown_output, markdown)
    if html_document is not None:
        _write(html_output, html_document)
    if json_output is None:
        print(json_document)


def _audit_dict(report: Any) -> dict[str, Any]:
    return {
        "verdict": report.verdict.value,
        "verdict_reason": report.verdict_reason,
        "scope": report.scope,
        "results": [
            {
                "name": result.name,
                "status": result.status.value,
                "summary": result.summary,
                "metrics": result.metrics,
                "details": (
                    result.details.to_dict(orient="records")
                    if result.details is not None
                    else None
                ),
                "notes": result.notes,
            }
            for result in report.results
        ],
    }


def _comparison_dict(result: ScoreComparison) -> dict[str, Any]:
    return {
        "baseline_name": result.baseline_name,
        "candidate_name": result.candidate_name,
        "outcome_type": result.outcome_type,
        "coverage": asdict(result.coverage),
        "metrics": result.metrics,
        "tables": {
            name: table.to_dict(orient="records") for name, table in result.tables().items()
        },
        "notes": result.notes,
    }


def _tables_html(title: str, summary: Mapping[str, Any], tables: Mapping[str, pd.DataFrame]) -> str:
    summary_items = "".join(
        f"<dt>{html.escape(str(key))}</dt><dd>{html.escape(str(value))}</dd>"
        for key, value in summary.items()
    )
    rendered = []
    for name, table in tables.items():
        rendered.append(
            f"<h2>{html.escape(name.replace('_', ' ').title())}</h2>"
            + table.head(100).to_html(index=False, escape=True, border=0)
            + (
                f"<p>Showing first 100 of {len(table)} rows.</p>" if len(table) > 100 else ""
            )
        )
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>body{{font-family:system-ui,sans-serif;"
        "color:#18212b;max-width:1100px;margin:32px auto;padding:0 16px;line-height:1.5}"
        "table{border-collapse:collapse;width:100%;margin-bottom:24px}th,td{border:1px "
        "solid #cbd3dc;padding:7px;text-align:left}th{background:#f3f6f8}"
        "dl{display:grid;grid-template-columns:220px 1fr;gap:6px 12px}dd{margin:0}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1><dl>{summary_items}</dl>"
        + "".join(rendered)
        + "</body></html>"
    )


def _audit_command(args: argparse.Namespace, config: dict[str, Any], raw: dict[str, Any]) -> int:
    frame = _read_table(_option(args, config, "input"))
    indicator_names = _columns(
        frame,
        _required(_option(args, config, "indicators"), "indicators"),
        "indicator columns",
    )
    report = ProxyAudit(
        indicators=frame[indicator_names],
        score=_column(frame, _option(args, config, "score"), "score"),
        outcome=_column(frame, _option(args, config, "outcome"), "outcome"),
        segments=_column(frame, _option(args, config, "segments"), "segments"),
        period=_column(frame, _option(args, config, "period"), "period"),
        thresholds=_thresholds(raw.get("thresholds"), args.threshold),
    ).run()
    title = str(_option(args, config, "title", "Proxy score audit"))
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise CliInputError("[metadata] must be a TOML table")
    _emit(
        _json_document(_audit_dict(report)),
        markdown=report.to_markdown(),
        html_document=report.to_html(title=title, metadata=metadata),
        json_output=_option(args, config, "json_output"),
        markdown_output=_option(args, config, "markdown_output"),
        html_output=_option(args, config, "html_output"),
    )
    return {
        Verdict.DECISION_GRADE: EXIT_SUCCESS,
        Verdict.DIRECTIONAL: EXIT_WARNING,
        Verdict.NOT_VALIDATED: EXIT_VALIDATION_FAILED,
    }[report.verdict]


def _compare_command(args: argparse.Namespace, config: dict[str, Any], raw: dict[str, Any]) -> int:
    frame = _read_table(_option(args, config, "input"))
    baseline_name = str(_option(args, config, "baseline_name", "baseline"))
    candidate_name = str(_option(args, config, "candidate_name", "candidate"))
    result = compare_scores(
        _column(
            frame,
            _required(_option(args, config, "baseline_score"), "baseline_score"),
            "baseline score",
            required=True,
        ),
        _column(
            frame,
            _required(_option(args, config, "candidate_score"), "candidate_score"),
            "candidate score",
            required=True,
        ),
        _column(
            frame,
            _required(_option(args, config, "outcome"), "outcome"),
            "outcome",
            required=True,
        ),
        segments=_column(frame, _option(args, config, "segments"), "segments"),
        period=_column(frame, _option(args, config, "period"), "period"),
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        n_bands=int(_option(args, config, "n_bands", 10)),
        n_bootstrap=int(_option(args, config, "n_bootstrap", 500)),
        random_state=int(_option(args, config, "random_state", 42)),
        action_cutoffs=_option(args, config, "action_cutoffs"),
        action_percentiles=_option(args, config, "action_percentiles"),
        action_top_n=_option(args, config, "action_top_n"),
        thresholds=_thresholds(raw.get("thresholds"), args.threshold),
    )
    _emit(
        _json_document(_comparison_dict(result)),
        markdown=result.to_markdown(),
        html_document=_tables_html(
            "Score version comparison",
            {
                "baseline": baseline_name,
                "candidate": candidate_name,
                "paired_rows": result.coverage.evaluation_rows,
            },
            result.tables(),
        ),
        json_output=_option(args, config, "json_output"),
        markdown_output=_option(args, config, "markdown_output"),
        html_output=_option(args, config, "html_output"),
    )
    assessments = set(result.dimensions["assessment"])
    if "regressed" in assessments:
        return EXIT_VALIDATION_FAILED
    if "inconclusive" in assessments:
        return EXIT_WARNING
    return EXIT_SUCCESS


def _baseline_command(args: argparse.Namespace, config: dict[str, Any], raw: dict[str, Any]) -> int:
    frame = _read_table(_option(args, config, "input"))
    indicator_names = _columns(
        frame,
        _required(_option(args, config, "indicators"), "indicators"),
        "indicator columns",
    )
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise CliInputError("[metadata] must be a TOML table")
    metadata.update(_pairs(args.metadata, "--metadata"))
    artifact = create_monitoring_baseline(
        frame[indicator_names],
        score_id=str(_required(_option(args, config, "score_id"), "score_id")),
        score_version=str(
            _required(_option(args, config, "score_version"), "score_version")
        ),
        score=_column(frame, _option(args, config, "score"), "score", required=True),
        outcome=_column(frame, _option(args, config, "outcome"), "outcome"),
        thresholds=_thresholds(raw.get("thresholds"), args.threshold),
        monitoring_limits=_limits(raw.get("monitoring_limits"), args.limit),
        bins=int(_option(args, config, "bins", 10)),
        metadata=metadata,
    )
    artifact_path = _required(_option(args, config, "artifact"), "artifact")
    artifact.save(str(artifact_path))
    summary = pd.DataFrame(
        [
            {"measure": "baseline_rows", "value": artifact.baseline_rows},
            {"measure": "indicator_count", "value": len(artifact.indicator_columns)},
            {"measure": "score_id", "value": artifact.score_id},
            {"measure": "score_version", "value": artifact.score_version},
        ]
    )
    markdown = (
        "# Monitoring baseline\n\n"
        f"- Score: `{artifact.score_id}`\n"
        f"- Version: `{artifact.score_version}`\n"
        f"- Rows: {artifact.baseline_rows}\n"
        f"- Artifact: `{artifact_path}`\n"
    )
    _emit(
        artifact.to_json(),
        markdown=markdown,
        html_document=_tables_html(
            "Monitoring baseline",
            {"score_id": artifact.score_id, "score_version": artifact.score_version},
            {"summary": summary},
        ),
        json_output=_option(args, config, "json_output", artifact_path),
        markdown_output=_option(args, config, "markdown_output"),
        html_output=_option(args, config, "html_output"),
    )
    return EXIT_SUCCESS


def _monitor_command(args: argparse.Namespace, config: dict[str, Any], raw: dict[str, Any]) -> int:
    del raw
    frame = _read_table(_option(args, config, "input"))
    artifact_path = _required(_option(args, config, "artifact"), "artifact")
    baseline = MonitoringBaseline.load(str(artifact_path))
    indicator_names = baseline.indicator_columns
    result = monitor_batch(
        baseline,
        frame[indicator_names] if all(name in frame for name in indicator_names) else frame,
        score=_column(frame, _option(args, config, "score"), "score"),
        outcome=_column(frame, _option(args, config, "outcome"), "outcome"),
        score_version=_option(args, config, "score_version"),
        batch_id=str(_option(args, config, "batch_id", "batch")),
    )
    _emit(
        result.to_json(),
        markdown=result.to_markdown(),
        html_document=result.to_html(),
        json_output=_option(args, config, "json_output"),
        markdown_output=_option(args, config, "markdown_output"),
        html_output=_option(args, config, "html_output"),
    )
    if result.alert_state is MonitorStatus.FAILURE:
        return EXIT_VALIDATION_FAILED
    if result.alert_state in (MonitorStatus.WARNING, MonitorStatus.NOT_ASSESSABLE):
        return EXIT_WARNING
    return EXIT_SUCCESS


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Versioned TOML configuration file")
    parser.add_argument("--input", help="Input CSV or Parquet file")
    parser.add_argument("--json-output", help="Write machine-readable JSON to this path")
    parser.add_argument("--markdown-output", help="Write Markdown report to this path")
    parser.add_argument("--html-output", help="Write standalone HTML report to this path")


def _add_thresholds(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--threshold",
        action="append",
        metavar="NAME=VALUE",
        help="Override a Thresholds field; may be repeated",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the public command parser."""
    parser = argparse.ArgumentParser(prog="proxyscore", description=__doc__)
    parser.add_argument("--version", action="version", version=f"proxyscore {_package_version()}")
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="Run a proxy-score audit")
    _add_common(audit)
    _add_thresholds(audit)
    audit.add_argument("--indicators", nargs="+", help="Numeric indicator columns")
    audit.add_argument("--score", help="Score column")
    audit.add_argument("--outcome", help="Delayed outcome column")
    audit.add_argument("--segments", help="Segment column")
    audit.add_argument("--period", help="Period column")
    audit.add_argument("--title", help="HTML report title")

    compare = commands.add_parser("compare", help="Compare baseline and candidate scores")
    _add_common(compare)
    _add_thresholds(compare)
    compare.add_argument("--baseline-score", help="Baseline score column")
    compare.add_argument("--candidate-score", help="Candidate score column")
    compare.add_argument("--outcome", help="Delayed outcome column")
    compare.add_argument("--segments", help="Segment column")
    compare.add_argument("--period", help="Period column")
    compare.add_argument("--baseline-name")
    compare.add_argument("--candidate-name")
    compare.add_argument("--n-bands", type=int)
    compare.add_argument("--n-bootstrap", type=int)
    compare.add_argument("--random-state", type=int)
    compare.add_argument("--action-cutoffs", nargs="+", type=float)
    compare.add_argument("--action-percentiles", nargs="+", type=float)
    compare.add_argument("--action-top-n", nargs="+", type=int)

    baseline = commands.add_parser("baseline", help="Create a monitoring baseline artifact")
    _add_common(baseline)
    _add_thresholds(baseline)
    baseline.add_argument("--limit", action="append", metavar="NAME=VALUE")
    baseline.add_argument("--indicators", nargs="+", help="Numeric indicator columns")
    baseline.add_argument("--score", help="Externally generated score column")
    baseline.add_argument("--outcome", help="Delayed outcome column")
    baseline.add_argument("--score-id")
    baseline.add_argument("--score-version")
    baseline.add_argument("--artifact", help="Output baseline JSON artifact")
    baseline.add_argument("--bins", type=int)
    baseline.add_argument("--metadata", action="append", metavar="KEY=VALUE")

    monitor = commands.add_parser("monitor", help="Evaluate a batch against a baseline")
    _add_common(monitor)
    monitor.add_argument("--artifact", help="Baseline JSON artifact")
    monitor.add_argument("--score", help="Externally generated score column")
    monitor.add_argument("--outcome", help="Matured delayed outcome column")
    monitor.add_argument("--score-version")
    monitor.add_argument("--batch-id")
    return parser


def _package_version() -> str:
    from importlib import metadata

    try:
        return metadata.version("proxyscore")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def _run(args: argparse.Namespace) -> int:
    raw = _load_toml(args.config) if args.config else {"config_version": CONFIG_VERSION}
    config = _command_config(raw, args.command)
    handlers = {
        "audit": _audit_command,
        "compare": _compare_command,
        "baseline": _baseline_command,
        "monitor": _monitor_command,
    }
    return handlers[args.command](args, config, raw)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a stable process exit status."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        return _run(args)
    except (CliInputError, ArtifactVersionError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"proxyscore: input error: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
    except Exception as exc:  # pragma: no cover - last-resort process boundary
        print(f"proxyscore: internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INTERNAL_ERROR


def cli() -> NoReturn:
    """Console-script wrapper."""
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover
    cli()
