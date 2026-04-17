from __future__ import annotations

import json

import pytest

from ragharness.config import RagHarnessConfig
from ragharness.orchestrator import (
    SweepResult,
    estimate_cost,
    expand_sweep,
    run_sweep,
)

# ── expand_sweep ─────────────────────────────────────────


def test_expand_sweep_empty():
    assert expand_sweep({}) == [{}]


def test_expand_sweep_single_param():
    result = expand_sweep({"top_k": [3, 5]})
    assert result == [{"top_k": 3}, {"top_k": 5}]


def test_expand_sweep_two_params():
    result = expand_sweep({"a": [1, 2], "b": ["x", "y"]})
    assert len(result) == 4
    assert {"a": 1, "b": "x"} in result
    assert {"a": 2, "b": "y"} in result


def test_expand_sweep_three_params():
    result = expand_sweep({"a": [1, 2], "b": [3], "c": [4, 5]})
    assert len(result) == 4  # 2 * 1 * 2


# ── estimate_cost ────────────────────────────────────────


def test_estimate_cost_basic():
    cost = estimate_cost(
        100,
        2,
        avg_prompt_tokens=1000,
        avg_completion_tokens=500,
        input_per_1k=0.003,
        output_per_1k=0.015,
    )
    # 200 queries * (1000/1000 * 0.003 + 500/1000 * 0.015) = 200 * 0.0105 = 2.10
    assert abs(cost - 2.10) < 1e-9


def test_estimate_cost_zero():
    assert estimate_cost(0, 5) == 0.0
    assert estimate_cost(5, 0) == 0.0


# ── run_sweep ────────────────────────────────────────────


@pytest.fixture
def _patch_adapter(monkeypatch):
    """Patch create_adapter to return a DummyRAGSystem."""
    from tests.conftest import DummyRAGSystem

    def _factory(adapter_type, adapter_config, sweep_overrides=None):
        return DummyRAGSystem(answer="42", docs=["doc_a"])

    monkeypatch.setattr("ragharness.orchestrator.create_adapter", _factory)


def _make_config(dataset_path: str, sweep: dict | None = None, metrics: list | None = None):
    return RagHarnessConfig(
        dataset={"source": "jsonl", "path": dataset_path},
        system={"adapter": "raw", "adapter_config": {"llm_provider": "openai"}},
        sweep=sweep or {},
        metrics=metrics or ["exact_match", "latency_p50", "latency_p95"],
    )


def test_run_sweep_baseline(tmp_path, _patch_adapter):
    """Single baseline config (no sweep), 3 questions."""
    ds_path = tmp_path / "ds.jsonl"
    ds_path.write_text(
        "\n".join(json.dumps({"question": f"Q{i}", "expected_answer": "42"}) for i in range(3))
    )

    cfg = _make_config(str(ds_path))
    result = run_sweep(cfg, no_confirm=True)

    assert len(result.runs) == 1
    run = result.runs[0]
    assert run.config_params == {}
    assert len(run.per_question_scores) == 3
    assert len(run.raw_results) == 3
    # All answers are "42" == expected "42", so exact_match should be 1.0
    assert run.aggregate_scores["mean_exact_match"] == 1.0


def test_run_sweep_with_sweep_params(tmp_path, _patch_adapter):
    """2x2 sweep = 4 configs."""
    ds_path = tmp_path / "ds.jsonl"
    ds_path.write_text(
        "\n".join(json.dumps({"question": f"Q{i}", "expected_answer": "42"}) for i in range(2))
    )

    cfg = _make_config(
        str(ds_path),
        sweep={"top_k": [3, 5], "temperature": [0.0, 0.3]},
    )
    result = run_sweep(cfg, no_confirm=True)

    assert len(result.runs) == 4
    # Each run should have 2 per-question scores
    for run in result.runs:
        assert len(run.per_question_scores) == 2


def test_run_sweep_dry_run(tmp_path, _patch_adapter):
    """Dry run returns empty SweepResult and doesn't call any adapters."""
    ds_path = tmp_path / "ds.jsonl"
    ds_path.write_text(json.dumps({"question": "Q1", "expected_answer": "A1"}))

    cfg = _make_config(str(ds_path))
    result = run_sweep(cfg, dry_run=True)

    assert isinstance(result, SweepResult)
    assert len(result.runs) == 0


def test_run_sweep_with_limit(tmp_path, _patch_adapter):
    """Dataset limit truncates questions."""
    ds_path = tmp_path / "ds.jsonl"
    ds_path.write_text(
        "\n".join(json.dumps({"question": f"Q{i}", "expected_answer": "42"}) for i in range(10))
    )

    cfg = _make_config(str(ds_path))
    cfg.dataset.limit = 3
    result = run_sweep(cfg, no_confirm=True)

    assert len(result.runs[0].per_question_scores) == 3


def test_run_sweep_hf_source(monkeypatch, _patch_adapter):
    """HF source: no path required, dataset loaded via from_huggingface."""
    from ragharness.dataset import EvalDataset, EvalItem

    def _fake_hf_load(name, **kwargs):
        assert name == "fake/ds"
        return EvalDataset([EvalItem(question=f"Q{i}", expected_answer="42") for i in range(3)])

    monkeypatch.setattr(
        EvalDataset,
        "from_huggingface",
        classmethod(lambda cls, name, **kwargs: _fake_hf_load(name, **kwargs)),
    )

    cfg = RagHarnessConfig(
        dataset={"source": "huggingface", "name": "fake/ds"},
        system={"adapter": "raw", "adapter_config": {"llm_provider": "openai"}},
        metrics=["exact_match"],
    )
    result = run_sweep(cfg, no_confirm=True)
    assert len(result.runs) == 1
    assert len(result.runs[0].per_question_scores) == 3


def test_run_sweep_parameterised_non_llm_metric_is_wrapped(tmp_path, monkeypatch):
    """A parameterised per-question metric (e.g. recall_at_k with k=3) must be
    wrapped with functools.partial — not passed params as kwargs that its
    callable signature wouldn't accept."""
    from tests.conftest import DummyRAGSystem

    # Adapter that returns a fixed retrieved_docs list so recall_at_k has
    # something to measure.
    def _factory(adapter_type, adapter_config, sweep_overrides=None):
        return DummyRAGSystem(answer="42", docs=["a", "b", "c", "d"])

    monkeypatch.setattr("ragharness.orchestrator.create_adapter", _factory)

    ds_path = tmp_path / "ds.jsonl"
    ds_path.write_text(
        json.dumps(
            {
                "question": "Q1",
                "expected_answer": "42",
                "expected_docs": ["a", "b"],
            }
        )
    )

    # k=3 caps the window — both expected docs are in the first 3, recall=1.0
    cfg = _make_config(str(ds_path), metrics=[{"recall_at_k": {"k": 3}}])
    result = run_sweep(cfg, no_confirm=True)

    assert result.runs[0].per_question_scores[0]["recall_at_k"] == 1.0
    assert result.runs[0].aggregate_scores["mean_recall_at_k"] == 1.0


def test_run_sweep_aggregate_latency(tmp_path, _patch_adapter):
    """Latency metrics should be populated from metadata."""
    ds_path = tmp_path / "ds.jsonl"
    ds_path.write_text(
        "\n".join(json.dumps({"question": f"Q{i}", "expected_answer": "42"}) for i in range(5))
    )

    cfg = _make_config(str(ds_path))
    result = run_sweep(cfg, no_confirm=True)

    run = result.runs[0]
    assert "latency_p50" in run.aggregate_scores
    assert "latency_p95" in run.aggregate_scores
    # DummyRAGSystem sets latency_ms=100, but orchestrator also sets it
    # via perf_counter — either way it should be > 0
    assert run.aggregate_scores["latency_p50"] >= 0
