from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from proxyscore.cli import (
    EXIT_BAD_INPUT,
    EXIT_INTERNAL_ERROR,
    EXIT_SUCCESS,
    EXIT_VALIDATION_FAILED,
    EXIT_WARNING,
    _read_table,
    build_parser,
    main,
)


def cli_data(n=400, seed=61):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    x1 = latent + rng.normal(scale=0.8, size=n)
    x2 = latent + rng.normal(scale=1.0, size=n)
    baseline = 0.4 * latent + rng.normal(scale=1.2, size=n)
    candidate = latent + rng.normal(scale=0.3, size=n)
    probability = 1 / (1 + np.exp(-1.5 * latent))
    outcome = (rng.uniform(size=n) < probability).astype(int)
    return pd.DataFrame(
        {
            "entity_id": [f"account-{i}" for i in range(n)],
            "x1": x1,
            "x2": x2,
            "baseline_score": baseline,
            "candidate_score": candidate,
            "outcome": outcome,
            "segment": np.where(np.arange(n) % 2, "smb", "enterprise"),
            "period": np.where(np.arange(n) < n / 2, "m1", "m2"),
        }
    )


def write_csv(tmp_path, frame=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "data.csv"
    (frame if frame is not None else cli_data()).to_csv(path, index=False)
    return path


def test_audit_command_writes_json_markdown_and_html(tmp_path):
    data = write_csv(tmp_path)
    json_path = tmp_path / "out" / "audit.json"
    markdown_path = tmp_path / "out" / "audit.md"
    html_path = tmp_path / "out" / "audit.html"
    code = main(
        [
            "audit",
            "--input",
            str(data),
            "--indicators",
            "x1",
            "x2",
            "--score",
            "candidate_score",
            "--outcome",
            "outcome",
            "--segments",
            "segment",
            "--period",
            "period",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
            "--html-output",
            str(html_path),
        ]
    )

    assert code in (EXIT_SUCCESS, EXIT_WARNING, EXIT_VALIDATION_FAILED)
    result = json.loads(json_path.read_text(encoding="utf-8"))
    assert result["verdict"] in {"decision_grade", "directional", "not_validated"}
    assert {item["name"] for item in result["results"]} >= {"downstream", "stability"}
    assert "# Proxy score audit" in markdown_path.read_text(encoding="utf-8")
    assert "<!doctype html>" in html_path.read_text(encoding="utf-8")


def test_audit_without_outcome_returns_validation_failure(tmp_path):
    data = write_csv(tmp_path)
    output = tmp_path / "audit.json"
    code = main(
        [
            "audit",
            "--input",
            str(data),
            "--indicators",
            "x1",
            "x2",
            "--score",
            "candidate_score",
            "--json-output",
            str(output),
        ]
    )
    assert code == EXIT_VALIDATION_FAILED
    assert json.loads(output.read_text())["verdict"] == "not_validated"


def test_json_is_printed_when_no_json_output_is_given(tmp_path, capsys):
    data = write_csv(tmp_path)
    code = main(
        [
            "audit",
            "--input",
            str(data),
            "--indicators",
            "x1",
            "x2",
            "--score",
            "candidate_score",
        ]
    )
    captured = capsys.readouterr()
    assert code == EXIT_VALIDATION_FAILED
    assert json.loads(captured.out)["verdict"] == "not_validated"
    assert captured.err == ""


def test_compare_command_writes_all_formats(tmp_path):
    data = write_csv(tmp_path)
    json_path = tmp_path / "compare.json"
    markdown_path = tmp_path / "compare.md"
    html_path = tmp_path / "compare.html"
    code = main(
        [
            "compare",
            "--input",
            str(data),
            "--baseline-score",
            "baseline_score",
            "--candidate-score",
            "candidate_score",
            "--outcome",
            "outcome",
            "--segments",
            "segment",
            "--period",
            "period",
            "--n-bootstrap",
            "30",
            "--action-top-n",
            "20",
            "50",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(markdown_path),
            "--html-output",
            str(html_path),
        ]
    )

    assert code in (EXIT_SUCCESS, EXIT_WARNING, EXIT_VALIDATION_FAILED)
    result = json.loads(json_path.read_text())
    assert "performance" in result["tables"]
    assert "actions" in result["tables"]
    assert "# Score version comparison" in markdown_path.read_text()
    assert "<!doctype html>" in html_path.read_text()


def test_baseline_and_monitor_commands_run_end_to_end(tmp_path):
    data = write_csv(tmp_path)
    artifact = tmp_path / "baseline.json"
    baseline_html = tmp_path / "baseline.html"
    baseline_code = main(
        [
            "baseline",
            "--input",
            str(data),
            "--indicators",
            "x1",
            "x2",
            "--score",
            "candidate_score",
            "--outcome",
            "outcome",
            "--score-id",
            "health",
            "--score-version",
            "v1",
            "--artifact",
            str(artifact),
            "--html-output",
            str(baseline_html),
        ]
    )
    assert baseline_code == EXIT_SUCCESS
    assert json.loads(artifact.read_text())["score_id"] == "health"
    assert baseline_html.exists()

    run_json = tmp_path / "monitor.json"
    run_md = tmp_path / "monitor.md"
    run_html = tmp_path / "monitor.html"
    monitor_code = main(
        [
            "monitor",
            "--input",
            str(data),
            "--artifact",
            str(artifact),
            "--score",
            "candidate_score",
            "--outcome",
            "outcome",
            "--score-version",
            "v1",
            "--batch-id",
            "2026-07",
            "--json-output",
            str(run_json),
            "--markdown-output",
            str(run_md),
            "--html-output",
            str(run_html),
        ]
    )
    assert monitor_code == EXIT_SUCCESS
    result = json.loads(run_json.read_text())
    assert result["alert_state"] == "informational"
    assert result["batch_id"] == "2026-07"
    assert run_md.exists() and run_html.exists()


def test_monitor_without_matured_outcome_maps_to_warning_exit(tmp_path):
    data = write_csv(tmp_path)
    artifact = tmp_path / "baseline.json"
    assert (
        main(
            [
                "baseline",
                "--input",
                str(data),
                "--indicators",
                "x1",
                "x2",
                "--score",
                "candidate_score",
                "--score-id",
                "health",
                "--score-version",
                "v1",
                "--artifact",
                str(artifact),
            ]
        )
        == EXIT_SUCCESS
    )
    code = main(
        [
            "monitor",
            "--input",
            str(data),
            "--artifact",
            str(artifact),
            "--score",
            "candidate_score",
            "--json-output",
            str(tmp_path / "run.json"),
        ]
    )
    assert code == EXIT_WARNING


def test_monitor_schema_failure_maps_to_validation_failure(tmp_path):
    data = write_csv(tmp_path)
    artifact = tmp_path / "baseline.json"
    main(
        [
            "baseline",
            "--input",
            str(data),
            "--indicators",
            "x1",
            "x2",
            "--score",
            "candidate_score",
            "--score-id",
            "health",
            "--score-version",
            "v1",
            "--artifact",
            str(artifact),
        ]
    )
    broken = cli_data().drop(columns="x1")
    broken_path = write_csv(tmp_path / "broken", broken)
    code = main(
        [
            "monitor",
            "--input",
            str(broken_path),
            "--artifact",
            str(artifact),
            "--score",
            "candidate_score",
            "--json-output",
            str(tmp_path / "broken.json"),
        ]
    )
    assert code == EXIT_VALIDATION_FAILED


def test_versioned_toml_config_and_cli_override(tmp_path):
    data = write_csv(tmp_path)
    output = tmp_path / "configured.json"
    config = tmp_path / "audit.toml"
    config.write_text(
        f'''config_version = 1

[audit]
input = "{data.as_posix()}"
indicators = ["x1", "x2"]
score = "does_not_exist"
outcome = "outcome"
json_output = "{output.as_posix()}"

[thresholds]
min_auc_strong = 0.60

[metadata]
owner = "analytics"
''',
        encoding="utf-8",
    )
    code = main(["audit", "--config", str(config), "--score", "candidate_score"])

    assert code in (EXIT_SUCCESS, EXIT_WARNING, EXIT_VALIDATION_FAILED)
    assert output.exists()
    assert json.loads(output.read_text())["scope"]["score_supplied"] is True


def test_threshold_and_limit_command_line_overrides(tmp_path):
    data = write_csv(tmp_path)
    artifact = tmp_path / "baseline.json"
    code = main(
        [
            "baseline",
            "--input",
            str(data),
            "--indicators",
            "x1",
            "x2",
            "--score",
            "candidate_score",
            "--score-id",
            "health",
            "--score-version",
            "v1",
            "--artifact",
            str(artifact),
            "--threshold",
            "psi_unstable=0.2",
            "--limit",
            "performance_failure_drop=0.08",
            "--metadata",
            "owner=analytics",
        ]
    )
    values = json.loads(artifact.read_text())
    assert code == EXIT_SUCCESS
    assert values["thresholds"]["psi_unstable"] == 0.2
    assert values["monitoring_limits"]["performance_failure_drop"] == 0.08
    assert values["metadata"]["owner"] == "analytics"


@pytest.mark.parametrize("version", [0, 2, "1"])
def test_unsupported_config_version_is_bad_input(tmp_path, capsys, version):
    config = tmp_path / "bad.toml"
    encoded = f'"{version}"' if isinstance(version, str) else str(version)
    config.write_text(f"config_version = {encoded}\n", encoding="utf-8")
    code = main(["audit", "--config", str(config)])
    captured = capsys.readouterr()
    assert code == EXIT_BAD_INPUT
    assert "unsupported config_version" in captured.err


def test_bad_column_error_does_not_log_row_values(tmp_path, capsys):
    frame = cli_data()
    frame.loc[0, "entity_id"] = "SECRET-ROW-VALUE"
    data = write_csv(tmp_path, frame)
    code = main(
        [
            "audit",
            "--input",
            str(data),
            "--indicators",
            "missing_indicator",
        ]
    )
    captured = capsys.readouterr()
    assert code == EXIT_BAD_INPUT
    assert "missing_indicator" in captured.err
    assert "SECRET-ROW-VALUE" not in captured.err
    assert captured.out == ""


def test_missing_file_and_unsupported_extension_are_bad_input(tmp_path, capsys):
    missing = main(["audit", "--input", str(tmp_path / "missing.csv"), "--indicators", "x"])
    assert missing == EXIT_BAD_INPUT
    unsupported_path = tmp_path / "data.xlsx"
    unsupported_path.write_text("not excel")
    unsupported = main(
        ["audit", "--input", str(unsupported_path), "--indicators", "x"]
    )
    assert unsupported == EXIT_BAD_INPUT
    assert "input must use" in capsys.readouterr().err


def test_parquet_suffix_dispatches_to_pandas(monkeypatch, tmp_path):
    path = tmp_path / "data.parquet"
    path.write_bytes(b"placeholder")
    expected = pd.DataFrame({"x": [1]})
    called = {}

    def fake_read_parquet(value):
        called["path"] = value
        return expected

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    result = _read_table(path)
    pd.testing.assert_frame_equal(result, expected)
    assert called["path"] == path


def test_parquet_missing_engine_has_actionable_error(monkeypatch, tmp_path):
    path = tmp_path / "data.parquet"
    path.write_bytes(b"placeholder")

    def fail(_):
        raise ImportError("no engine")

    monkeypatch.setattr(pd, "read_parquet", fail)
    with pytest.raises(ValueError, match=r"proxyscore\[parquet\]"):
        _read_table(path)


def test_parser_exposes_all_commands():
    parser = build_parser()
    help_text = parser.format_help()
    for command in ("audit", "compare", "baseline", "monitor"):
        assert command in help_text


def test_console_script_is_declared():
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = project.read_text(encoding="utf-8")
    assert 'proxyscore = "proxyscore.cli:cli"' in text


def test_unexpected_exception_maps_to_internal_error(monkeypatch, capsys):
    import proxyscore.cli as cli_module

    def fail(_):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(cli_module, "_run", fail)
    code = main(["audit", "--input", "unused", "--indicators", "x"])
    captured = capsys.readouterr()
    assert code == EXIT_INTERNAL_ERROR
    assert "internal error: RuntimeError" in captured.err
