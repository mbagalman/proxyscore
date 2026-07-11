"""Self-contained HTML rendering for audit reports."""

from __future__ import annotations

import html
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .audit import AuditReport


_CSS = """
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; color: #18212b; background: #ffffff; line-height: 1.5; }
main { width: min(1120px, calc(100% - 32px)); margin: 0 auto; padding: 40px 0 64px; }
header { border-bottom: 3px solid #18212b; padding-bottom: 20px; margin-bottom: 28px; }
h1 { margin: 0 0 8px; font-size: 2rem; letter-spacing: 0; }
h2 { margin: 0 0 14px; font-size: 1.35rem; letter-spacing: 0; }
h3 { margin: 0 0 10px; font-size: 1.05rem; letter-spacing: 0; }
p { margin: 8px 0; }
section { padding: 24px 0; border-bottom: 1px solid #d8dee5; }
.verdict { font-size: 1.05rem; font-weight: 700; }
.limitations { color: #3f4d5a; max-width: 82ch; }
.status { display: inline-block; padding: 2px 7px; border: 1px solid currentColor;
  border-radius: 4px; font-size: .78rem; font-weight: 800; }
.status-pass { color: #17653a; background: #edf8f1; }
.status-warn { color: #765400; background: #fff8df; }
.status-fail { color: #9b2525; background: #fff0f0; }
.status-skip { color: #475569; background: #f2f5f8; }
.table-wrap { width: 100%; overflow-x: auto; margin-top: 12px; }
table { width: 100%; border-collapse: collapse; font-size: .9rem; }
caption { text-align: left; font-weight: 700; margin-bottom: 8px; }
th, td { border: 1px solid #cbd3dc; padding: 7px 9px; text-align: left; vertical-align: top; }
th { background: #f3f6f8; }
tbody tr:nth-child(even) { background: #fafbfc; }
dl { display: grid; grid-template-columns: minmax(150px, 240px) 1fr; gap: 6px 14px; }
dt { font-weight: 700; }
dd { margin: 0; overflow-wrap: anywhere; }
.check { padding: 20px 0; }
.check + .check { border-top: 1px dashed #cbd3dc; }
.check-heading { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.note { border-left: 4px solid #60758a; padding-left: 12px; color: #354554; }
.truncation { color: #5d6875; font-size: .85rem; }
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
  overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
@media (max-width: 640px) {
  main { width: min(100% - 20px, 1120px); padding-top: 24px; }
  h1 { font-size: 1.55rem; }
  dl { grid-template-columns: 1fr; gap: 2px; }
  dd { margin-bottom: 8px; }
}
"""


def _text(value: Any) -> str:
    if value is None or value is pd.NA:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        if math.isinf(value):
            return "infinity" if value > 0 else "-infinity"
        return f"{value:.4g}"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _definition_list(values: Mapping[str, Any]) -> str:
    if not values:
        return '<p class="limitations">None supplied.</p>'
    items = []
    for key, value in values.items():
        items.append(
            f"<dt>{html.escape(str(key))}</dt><dd>{html.escape(_text(value))}</dd>"
        )
    return "<dl>" + "".join(items) + "</dl>"


def _table(
    frame: pd.DataFrame,
    caption: str,
    max_rows: int | None,
) -> str:
    original_rows = len(frame)
    shown = frame if max_rows is None else frame.head(max_rows)
    table = shown.to_html(index=False, escape=True, border=0, classes="data-table")
    table = table.replace(
        ">",
        f"><caption>{html.escape(caption)}</caption>",
        1,
    )
    truncation = ""
    if len(shown) < original_rows:
        truncation = (
            '<p class="truncation">'
            f"Showing first {len(shown)} of {original_rows} rows."
            "</p>"
        )
    return f'<div class="table-wrap">{table}</div>{truncation}'


def _package_version() -> str:
    try:
        return importlib_metadata.version("proxyscore")
    except importlib_metadata.PackageNotFoundError:
        return "0.1.0"


def _generated_at(value: datetime | None) -> str:
    stamp = value or datetime.now(timezone.utc)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def render_audit_html(
    report: AuditReport,
    *,
    title: str = "Proxy score audit",
    metadata: Mapping[str, Any] | None = None,
    max_detail_rows: int | None = 100,
    generated_at: datetime | None = None,
) -> str:
    """Render an audit as one portable HTML document with inline styling."""
    if max_detail_rows is not None and (
        isinstance(max_detail_rows, bool)
        or not isinstance(max_detail_rows, int)
        or max_detail_rows < 1
    ):
        raise ValueError("max_detail_rows must be an integer >= 1 or None")

    generated = _generated_at(generated_at)
    safe_title = html.escape(str(title))
    summary = report.summary().copy()
    summary["status"] = summary["status"].str.upper()

    report_meta: dict[str, Any] = {
        "generated_at_utc": generated,
        "proxyscore_version": _package_version(),
    }
    if metadata:
        report_meta.update(metadata)

    check_sections: list[str] = []
    for result in report.results:
        status = result.status.value
        metric_html = ""
        if result.metrics:
            metric_html = "<h3>Metrics</h3>" + _definition_list(result.metrics)
        detail_html = ""
        if result.details is not None and len(result.details):
            detail_html = _table(
                result.details,
                f"{result.name} details",
                max_detail_rows,
            )
        notes_html = ""
        if result.notes:
            notes = "".join(f"<li>{html.escape(str(note))}</li>" for note in result.notes)
            notes_html = f"<h3>Notes</h3><ul class=\"note\">{notes}</ul>"
        check_sections.append(
            '<article class="check">'
            '<div class="check-heading">'
            f"<h3>{html.escape(result.name)}</h3>"
            f'<span class="status status-{status}">{html.escape(status.upper())}</span>'
            "</div>"
            f"<p>{html.escape(result.summary)}</p>"
            f"{metric_html}{detail_html}{notes_html}"
            "</article>"
        )

    action_html = ""
    action = report.action_analysis
    if action is not None:
        assumptions = ""
        if action.assumptions:
            assumptions = "<h3>Business-value assumptions</h3>" + _definition_list(
                action.assumptions
            )
        action_notes = ""
        if action.notes:
            notes = "".join(f"<li>{html.escape(str(note))}</li>" for note in action.notes)
            action_notes = f"<h3>Notes</h3><ul class=\"note\">{notes}</ul>"
        segments = ""
        if action.segment_table is not None and len(action.segment_table):
            segments = _table(
                action.segment_table,
                "Action analysis by segment",
                max_detail_rows,
            )
        action_html = (
            '<section aria-labelledby="action-analysis-heading">'
            '<h2 id="action-analysis-heading">Action analysis</h2>'
            f"<p>Outcome type: <strong>{html.escape(action.outcome_type)}</strong>; "
            f"polarity: <strong>{action.polarity:+d}</strong>.</p>"
            f"{assumptions}"
            f"{_table(action.table, 'Evaluated action policies', max_detail_rows)}"
            f"{segments}{action_notes}"
            "</section>"
        )

    return "".join(
        [
            "<!doctype html><html lang=\"en\"><head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{safe_title}</title><style>{_CSS}</style></head><body><main>",
            f"<header><h1>{safe_title}</h1>",
            f'<p class="verdict">Verdict: {html.escape(report.verdict.value)}</p>',
            f"<p>{html.escape(report.verdict_reason)}</p>",
            '<p class="limitations">A favorable verdict summarizes the checks run within '
            "the supplied validation scope. It does not by itself establish calibration, "
            "prospective performance, intervention effects, or governance clearance.</p>",
            "</header>",
            '<section aria-labelledby="scope-heading"><h2 id="scope-heading">Input scope</h2>',
            _definition_list(report.scope),
            "</section>",
            '<section aria-labelledby="metadata-heading">'
            '<h2 id="metadata-heading">Report metadata</h2>',
            _definition_list(report_meta),
            "</section>",
            '<section aria-labelledby="summary-heading"><h2 id="summary-heading">Summary</h2>',
            _table(summary, "Audit check summary", None),
            "</section>",
            '<section aria-labelledby="checks-heading"><h2 id="checks-heading">Checks</h2>',
            "".join(check_sections),
            "</section>",
            action_html,
            "</main></body></html>",
        ]
    )


def write_audit_html(
    report: AuditReport,
    path: str | Path,
    *,
    title: str = "Proxy score audit",
    metadata: Mapping[str, Any] | None = None,
    max_detail_rows: int | None = 100,
    generated_at: datetime | None = None,
) -> Path:
    """Write a self-contained audit report and return its resolved path."""
    destination = Path(path)
    document = render_audit_html(
        report,
        title=title,
        metadata=metadata,
        max_detail_rows=max_detail_rows,
        generated_at=generated_at,
    )
    destination.write_text(document, encoding="utf-8")
    return destination.resolve()
