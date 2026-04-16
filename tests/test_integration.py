"""End-to-end integration tests using the CLI with a mock adapter."""

from __future__ import annotations

import json
from unittest.mock import patch

from click.testing import CliRunner

from ragharness.cli import main
from tests.conftest import DummyRAGSystem

runner = CliRunner()


def _write_dataset(tmp_path, n: int = 3):
    ds = tmp_path / "dataset.jsonl"
    ds.write_text(
        "\n".join(json.dumps({"question": f"Q{i}", "expected_answer": "42"}) for i in range(n))
    )
    return ds


def _write_config(tmp_path, ds_path, sweep=None):
    sweep_block = ""
    if sweep:
        lines = ["sweep:"]
        for k, v in sweep.items():
            lines.append(f"  {k}: {v}")
        sweep_block = "\n".join(lines) + "\n"

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"dataset:\n  path: {ds_path}\n"
        f"system:\n  adapter: raw\n"
        f"  adapter_config:\n    llm_provider: openai\n"
        f"{sweep_block}"
        f"metrics:\n  - exact_match\n  - latency_p50\n  - latency_p95\n"
        f"output:\n  csv: {tmp_path / 'out' / 'results.csv'}\n"
        f"  charts: {tmp_path / 'out' / 'charts'}\n"
    )
    return cfg


def _mock_create_adapter(*args, **kwargs):
    return DummyRAGSystem(answer="42", docs=["doc_a"])


# ── run command ──────────────────────────────────────────


@patch("ragharness.orchestrator.create_adapter", _mock_create_adapter)
def test_run_end_to_end(tmp_path):
    ds = _write_dataset(tmp_path)
    cfg = _write_config(tmp_path, ds)

    result = runner.invoke(
        main, ["run", str(cfg), "--no-confirm", "--output-dir", str(tmp_path / "out")]
    )

    assert result.exit_code == 0, result.output
    assert "Summary" in result.output

    # CSV files created
    assert (tmp_path / "out" / "results_detail.csv").exists()
    assert (tmp_path / "out" / "results_summary.csv").exists()

    # Chart files created
    charts = list((tmp_path / "out" / "charts").glob("*.png"))
    assert len(charts) >= 1


@patch("ragharness.orchestrator.create_adapter", _mock_create_adapter)
def test_run_dry_run(tmp_path):
    ds = _write_dataset(tmp_path)
    cfg = _write_config(tmp_path, ds)

    result = runner.invoke(main, ["run", str(cfg), "--dry-run"])

    assert result.exit_code == 0
    assert "DRY RUN" in result.output
    # No output files
    assert not (tmp_path / "out").exists()


@patch("ragharness.orchestrator.create_adapter", _mock_create_adapter)
def test_run_with_sweep(tmp_path):
    ds = _write_dataset(tmp_path, n=2)
    cfg = _write_config(tmp_path, ds, sweep={"top_k": "[3, 5]"})

    result = runner.invoke(
        main, ["run", str(cfg), "--no-confirm", "--output-dir", str(tmp_path / "out")]
    )

    assert result.exit_code == 0, result.output
    # Should see 2 configs in the summary
    assert "Config 1/2" in result.output
    assert "Config 2/2" in result.output


@patch("ragharness.orchestrator.create_adapter", _mock_create_adapter)
def test_run_with_filter(tmp_path):
    ds = _write_dataset(tmp_path, n=2)
    cfg = _write_config(tmp_path, ds, sweep={"top_k": "[3, 5, 10]"})

    result = runner.invoke(
        main,
        [
            "run",
            str(cfg),
            "--no-confirm",
            "--output-dir",
            str(tmp_path / "out"),
            "--filter",
            "top_k=5",
        ],
    )

    assert result.exit_code == 0, result.output
    # Filter should reduce to 1 config
    assert "Config 1/1" in result.output


# ── report command ───────────────────────────────────────


@patch("ragharness.orchestrator.create_adapter", _mock_create_adapter)
def test_report_regenerates_charts(tmp_path):
    # First, run to produce a summary CSV
    ds = _write_dataset(tmp_path)
    cfg = _write_config(tmp_path, ds)

    runner.invoke(main, ["run", str(cfg), "--no-confirm", "--output-dir", str(tmp_path / "out")])

    summary_csv = tmp_path / "out" / "results_summary.csv"
    assert summary_csv.exists()

    # Now regenerate charts from the CSV
    regen_dir = tmp_path / "regen"
    result = runner.invoke(main, ["report", str(summary_csv), "--output-dir", str(regen_dir)])

    assert result.exit_code == 0, result.output
    assert "regenerated" in result.output.lower()
    pngs = list(regen_dir.glob("*.png"))
    assert len(pngs) >= 1
