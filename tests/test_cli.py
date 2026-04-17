from click.testing import CliRunner

from ragharness.cli import main

runner = CliRunner()


def test_version():
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_help():
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "validate" in result.output
    assert "report" in result.output
    assert "compare" in result.output


def test_validate_valid_config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("dataset:\n  path: data.jsonl\nsystem:\n  adapter: raw\n")
    result = runner.invoke(main, ["validate", str(cfg)])
    assert result.exit_code == 0
    assert "Config is valid" in result.output
    assert "raw" in result.output


def test_validate_invalid_config(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("dataset:\n  source: jsonl\n")
    result = runner.invoke(main, ["validate", str(cfg)])
    assert result.exit_code == 1
    assert "validation failed" in result.output.lower() or result.exit_code != 0


def test_run_help_mentions_seed():
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--seed" in result.output


def test_compare_help():
    result = runner.invoke(main, ["compare", "--help"])
    assert result.exit_code == 0
    assert "CSV_A" in result.output
    assert "CSV_B" in result.output
    assert "--threshold" in result.output
    assert "--html" in result.output


def test_compare_basic(tmp_path):
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    csv_a.write_text("adapter,mean_exact_match\nraw,0.8000\n")
    csv_b.write_text("adapter,mean_exact_match\nraw,0.9000\n")

    result = runner.invoke(main, ["compare", str(csv_a), str(csv_b)])
    assert result.exit_code == 0
    assert "adapter=raw" in result.output


def test_compare_with_csv_output(tmp_path):
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    csv_a.write_text("adapter,mean_exact_match\nraw,0.8000\n")
    csv_b.write_text("adapter,mean_exact_match\nraw,0.9000\n")
    out = tmp_path / "cmp.csv"

    result = runner.invoke(main, ["compare", str(csv_a), str(csv_b), "-o", str(out)])
    assert result.exit_code == 0
    assert out.exists()


def test_report_html_flag(tmp_path):
    csv_file = tmp_path / "summary.csv"
    csv_file.write_text("top_k,mean_exact_match,latency_p50\n3,0.8000,100.0000\n")
    html_out = tmp_path / "report.html"

    result = runner.invoke(main, ["report", str(csv_file), "--html", str(html_out)])
    assert result.exit_code == 0
    assert html_out.exists()
    content = html_out.read_text(encoding="utf-8")
    assert "ragharness" in content
