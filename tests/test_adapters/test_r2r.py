from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ragharness.adapters import create_adapter
from ragharness.adapters.r2r import R2RRAGSystem
from ragharness.protocol import RAGSystem

# ── Helpers ──────────────────────────────────────────────


def _mock_r2r_client() -> MagicMock:
    """Return a mock that quacks like r2r.R2RClient with a rag result."""
    client = MagicMock()
    client.retrieval.rag.return_value = SimpleNamespace(
        results=SimpleNamespace(
            generated_answer="Paris",
            search_results=SimpleNamespace(
                chunk_search_results=[
                    SimpleNamespace(text="France's capital is Paris."),
                    SimpleNamespace(text="Paris is a city in northern France."),
                ],
            ),
            metadata={"usage": {"input_tokens": 50, "output_tokens": 10}},
        )
    )
    return client


# ── Protocol conformance ────────────────────────────────


def test_r2r_satisfies_protocol():
    adapter = R2RRAGSystem(base_url="http://x")
    assert isinstance(adapter, RAGSystem)


# ── Basic query path ────────────────────────────────────


def test_query_basic():
    adapter = R2RRAGSystem(base_url="http://x", llm_model="openai/gpt-4o-mini")
    adapter._client = _mock_r2r_client()

    result = adapter.query("Capital of France?")

    assert result.answer == "Paris"
    assert result.retrieved_docs == [
        "France's capital is Paris.",
        "Paris is a city in northern France.",
    ]
    assert result.metadata["model"] == "openai/gpt-4o-mini"
    assert result.metadata["prompt_tokens"] == 50
    assert result.metadata["completion_tokens"] == 10
    assert result.metadata["latency_ms"] >= 0


# ── Parameter marshalling ───────────────────────────────


def test_query_passes_top_k_as_limit():
    adapter = R2RRAGSystem(base_url="http://x", top_k=7)
    mock = _mock_r2r_client()
    adapter._client = mock

    adapter.query("hi")

    kwargs = mock.retrieval.rag.call_args.kwargs
    assert kwargs["search_settings"]["limit"] == 7


def test_query_passes_model_and_temperature():
    adapter = R2RRAGSystem(
        base_url="http://x", llm_model="openai/gpt-4o", temperature=0.3
    )
    mock = _mock_r2r_client()
    adapter._client = mock

    adapter.query("hi")

    gen_cfg = mock.retrieval.rag.call_args.kwargs["rag_generation_config"]
    assert gen_cfg["model"] == "openai/gpt-4o"
    assert gen_cfg["temperature"] == 0.3
    assert gen_cfg["max_tokens"] == 1024


def test_query_omits_model_when_none():
    adapter = R2RRAGSystem(base_url="http://x", llm_model=None)
    mock = _mock_r2r_client()
    adapter._client = mock

    adapter.query("hi")

    gen_cfg = mock.retrieval.rag.call_args.kwargs["rag_generation_config"]
    assert "model" not in gen_cfg


def test_query_preserves_passthrough_config():
    """User-supplied extra search_settings/rag_generation_config survive."""
    adapter = R2RRAGSystem(
        base_url="http://x",
        search_settings={"use_hybrid_search": True, "limit": 99},
        rag_generation_config={"top_p": 0.9},
        top_k=5,
    )
    mock = _mock_r2r_client()
    adapter._client = mock

    adapter.query("hi")

    kwargs = mock.retrieval.rag.call_args.kwargs
    assert kwargs["search_settings"]["use_hybrid_search"] is True
    # top_k (sweep) should win over user-provided limit
    assert kwargs["search_settings"]["limit"] == 5
    assert kwargs["rag_generation_config"]["top_p"] == 0.9


# ── Response edge cases ─────────────────────────────────


def test_query_with_no_chunks():
    adapter = R2RRAGSystem(base_url="http://x")
    client = MagicMock()
    client.retrieval.rag.return_value = SimpleNamespace(
        results=SimpleNamespace(
            generated_answer="idk",
            search_results=SimpleNamespace(chunk_search_results=None),
            metadata={},
        )
    )
    adapter._client = client

    result = adapter.query("hi")
    assert result.retrieved_docs == []
    assert result.metadata["prompt_tokens"] == 0
    assert result.metadata["completion_tokens"] == 0


def test_query_without_wrapper():
    """Some SDK paths return the RAGResponse directly, not wrapped."""
    adapter = R2RRAGSystem(base_url="http://x")
    client = MagicMock()
    # No 'results' attribute — fall-through case
    client.retrieval.rag.return_value = SimpleNamespace(
        generated_answer="direct",
        search_results=SimpleNamespace(
            chunk_search_results=[SimpleNamespace(text="doc1")],
        ),
        metadata={},
    )
    adapter._client = client

    result = adapter.query("hi")
    assert result.answer == "direct"
    assert result.retrieved_docs == ["doc1"]


# ── Sweep param casting ─────────────────────────────────


def test_sweep_params_cast_types():
    """Sweep values from YAML arrive as strings; ensure they're cast."""
    adapter = R2RRAGSystem(
        base_url="http://x", top_k="10", temperature="0.7", max_tokens="2048"
    )
    assert adapter.top_k == 10
    assert adapter.temperature == 0.7
    assert adapter.max_tokens == 2048


# ── Pre-built client escape hatch ───────────────────────


def test_pre_built_client_used():
    mock = _mock_r2r_client()
    adapter = R2RRAGSystem(client=mock)
    # base_url default never touched — _get_client returns the mock directly
    assert adapter._get_client() is mock


# ── Import error guard ──────────────────────────────────


def test_import_error_hint(monkeypatch):
    """When r2r is unavailable, the error message points at the extra."""
    adapter = R2RRAGSystem(base_url="http://x")
    # Force the `from r2r import R2RClient` to fail.
    monkeypatch.setitem(sys.modules, "r2r", None)

    with pytest.raises(ImportError, match=r"ragharness\[r2r\]"):
        adapter._get_client()


# ── Factory function ────────────────────────────────────


def test_create_adapter_r2r():
    adapter = create_adapter("r2r", {"base_url": "http://x"})
    assert isinstance(adapter, R2RRAGSystem)


def test_create_adapter_r2r_with_sweep_overrides():
    adapter = create_adapter(
        "r2r",
        {"base_url": "http://x", "top_k": 3},
        sweep_overrides={"top_k": 10, "temperature": 0.5},
    )
    assert isinstance(adapter, R2RRAGSystem)
    assert adapter.top_k == 10
    assert adapter.temperature == 0.5
