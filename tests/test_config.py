import pytest
from pydantic import ValidationError

from ragharness.config import RagHarnessConfig, load_config


VALID_YAML = """\
dataset:
  source: jsonl
  path: ./data/test.jsonl

system:
  adapter: raw
  adapter_config:
    llm_provider: anthropic
    llm_model: claude-sonnet-4-20250514

sweep:
  top_k: [3, 5, 10]
  chunk_size: [256, 512]
  temperature: [0.0, 0.3]

metrics:
  - exact_match
  - llm_judge
  - precision_at_k:
      k: 5
  - latency_p50
  - latency_p95
  - token_cost:
      pricing:
        claude-sonnet-4-20250514:
          input_per_1k: 0.003
          output_per_1k: 0.015

output:
  csv: ./results/run.csv
  charts: ./results/charts/
"""

MINIMAL_YAML = """\
dataset:
  path: data.jsonl
system:
  adapter: raw
"""


# ── Valid config loading ─────────────────────────────────


def test_load_valid_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(VALID_YAML)

    cfg = load_config(path)
    assert cfg.dataset.source == "jsonl"
    assert cfg.dataset.path == "./data/test.jsonl"
    assert cfg.system.adapter == "raw"
    assert cfg.system.adapter_config["llm_provider"] == "anthropic"
    assert cfg.sweep["top_k"] == [3, 5, 10]
    assert cfg.sweep["chunk_size"] == [256, 512]
    assert cfg.output.csv == "./results/run.csv"


def test_load_minimal_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(MINIMAL_YAML)

    cfg = load_config(path)
    assert cfg.dataset.source == "jsonl"
    assert cfg.system.adapter == "raw"
    assert cfg.sweep == {}
    assert len(cfg.metrics) == 3  # defaults
    assert cfg.output.csv is not None


def test_parameterised_metrics(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(VALID_YAML)

    cfg = load_config(path)
    # precision_at_k entry is a dict
    prek = [m for m in cfg.metrics if isinstance(m, dict) and "precision_at_k" in m]
    assert len(prek) == 1
    assert prek[0]["precision_at_k"]["k"] == 5


# ── Defaults ─────────────────────────────────────────────


def test_defaults():
    cfg = RagHarnessConfig(
        dataset={"path": "data.jsonl"},
        system={"adapter": "raw"},
    )
    assert cfg.dataset.source == "jsonl"
    assert cfg.sweep == {}
    assert "exact_match" in cfg.metrics
    assert cfg.output.csv is not None
    assert cfg.output.charts is not None


# ── Validation errors ────────────────────────────────────


def test_missing_dataset_raises():
    with pytest.raises(ValidationError):
        RagHarnessConfig(system={"adapter": "raw"})


def test_missing_system_raises():
    with pytest.raises(ValidationError):
        RagHarnessConfig(dataset={"path": "data.jsonl"})


def test_unknown_adapter_raises():
    with pytest.raises(ValidationError, match="Unknown adapter"):
        RagHarnessConfig(
            dataset={"path": "data.jsonl"},
            system={"adapter": "nonexistent"},
        )


def test_unknown_metric_raises():
    with pytest.raises(ValidationError, match="Unknown metric"):
        RagHarnessConfig(
            dataset={"path": "data.jsonl"},
            system={"adapter": "raw"},
            metrics=["exact_match", "bogus_metric"],
        )


def test_jsonl_source_without_path_raises():
    with pytest.raises(ValidationError, match="requires 'path'"):
        RagHarnessConfig(
            dataset={"source": "jsonl"},
            system={"adapter": "raw"},
        )


def test_huggingface_source_without_name_raises():
    with pytest.raises(ValidationError, match="requires 'name'"):
        RagHarnessConfig(
            dataset={"source": "huggingface"},
            system={"adapter": "raw"},
        )


# ── File-level errors ───────────────────────────────────


def test_load_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_config("/no/such/file.yaml")


def test_load_non_mapping(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(path)
