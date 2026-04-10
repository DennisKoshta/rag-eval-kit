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


def test_validate_valid_config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "dataset:\n  path: data.jsonl\nsystem:\n  adapter: raw\n"
    )
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
