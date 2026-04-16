from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ragharness.adapters import create_adapter
from ragharness.adapters.raw import RawRAGSystem
from ragharness.protocol import RAGSystem

# ── Helpers ──────────────────────────────────────────────


def _mock_openai_client() -> MagicMock:
    """Return a mock that quacks like openai.OpenAI()."""
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Paris"))],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10),
    )
    return client


def _mock_anthropic_client() -> MagicMock:
    """Return a mock that quacks like anthropic.Anthropic()."""
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text="Paris")],
        usage=SimpleNamespace(input_tokens=60, output_tokens=12),
    )
    return client


# ── Protocol conformance ────────────────────────────────


def test_raw_satisfies_protocol():
    adapter = RawRAGSystem(llm_provider="openai")
    assert isinstance(adapter, RAGSystem)


# ── OpenAI path ─────────────────────────────────────────


def test_query_openai(monkeypatch):
    adapter = RawRAGSystem(llm_provider="openai", llm_model="gpt-4o", temperature=0.0)
    adapter._client = _mock_openai_client()

    result = adapter.query("Capital of France?")

    assert result.answer == "Paris"
    assert result.metadata["model"] == "gpt-4o"
    assert result.metadata["prompt_tokens"] == 50
    assert result.metadata["completion_tokens"] == 10
    assert result.metadata["latency_ms"] > 0


def test_query_openai_with_retriever():
    adapter = RawRAGSystem(
        llm_provider="openai",
        llm_model="gpt-4o",
        retriever=lambda q, k: [f"doc about {q}"],
        top_k=3,
    )
    adapter._client = _mock_openai_client()

    result = adapter.query("France?")

    assert result.retrieved_docs == ["doc about France?"]
    assert result.metadata["top_k"] == 3


# ── Anthropic path ──────────────────────────────────────


def test_query_anthropic():
    adapter = RawRAGSystem(llm_provider="anthropic", llm_model="claude-sonnet-4-20250514")
    adapter._client = _mock_anthropic_client()

    result = adapter.query("Capital of France?")

    assert result.answer == "Paris"
    assert result.metadata["prompt_tokens"] == 60
    assert result.metadata["completion_tokens"] == 12


# ── Sweep param injection ───────────────────────────────


def test_sweep_params_override():
    adapter = RawRAGSystem(llm_provider="openai", top_k=3, temperature=0.5)
    assert adapter.top_k == 3
    assert adapter.temperature == 0.5


def test_sweep_params_cast_types():
    """Sweep values from YAML arrive as int/float; ensure they're cast."""
    adapter = RawRAGSystem(llm_provider="openai", top_k="10", temperature="0.7")
    assert adapter.top_k == 10
    assert adapter.temperature == 0.7


# ── Seed plumbing ───────────────────────────────────────


def test_seed_forwarded_to_openai():
    """When seed is set it should appear in the OpenAI call kwargs."""
    adapter = RawRAGSystem(llm_provider="openai", llm_model="gpt-4o", seed=123)
    client = _mock_openai_client()
    adapter._client = client

    adapter.query("hi")

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["seed"] == 123


def test_seed_omitted_when_none():
    """When seed is None, no 'seed' kwarg is sent to OpenAI."""
    adapter = RawRAGSystem(llm_provider="openai", llm_model="gpt-4o")
    client = _mock_openai_client()
    adapter._client = client

    adapter.query("hi")

    kwargs = client.chat.completions.create.call_args.kwargs
    assert "seed" not in kwargs


def test_seed_ignored_by_anthropic():
    """Anthropic doesn't accept seed — it must not be forwarded."""
    adapter = RawRAGSystem(llm_provider="anthropic", llm_model="claude-sonnet-4-20250514", seed=99)
    client = _mock_anthropic_client()
    adapter._client = client

    adapter.query("hi")

    kwargs = client.messages.create.call_args.kwargs
    assert "seed" not in kwargs


# ── Unknown provider ────────────────────────────────────


def test_unknown_provider_raises():
    adapter = RawRAGSystem(llm_provider="fake")
    with pytest.raises(ValueError, match="Unknown llm_provider"):
        adapter.query("hi")


# ── Factory function ────────────────────────────────────


def test_create_adapter_raw():
    adapter = create_adapter("raw", {"llm_provider": "openai"})
    assert isinstance(adapter, RawRAGSystem)


def test_create_adapter_with_sweep_overrides():
    adapter = create_adapter(
        "raw",
        {"llm_provider": "openai", "top_k": 3},
        sweep_overrides={"top_k": 10, "temperature": 0.5},
    )
    assert isinstance(adapter, RawRAGSystem)
    assert adapter.top_k == 10
    assert adapter.temperature == 0.5


def test_create_adapter_unknown_raises():
    with pytest.raises(ValueError, match="Unknown adapter type"):
        create_adapter("nonexistent", {})
