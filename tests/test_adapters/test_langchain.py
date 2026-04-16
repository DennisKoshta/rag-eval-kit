from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.retrievers import BaseRetriever  # noqa: E402

from ragharness.adapters import create_adapter  # noqa: E402
from ragharness.adapters.langchain import LangChainRAGSystem  # noqa: E402
from ragharness.protocol import RAGSystem  # noqa: E402

# ── Helpers ──────────────────────────────────────────────


def _mock_llm(
    input_tokens: int = 50, output_tokens: int = 10, content: str = "Paris"
) -> MagicMock:
    """Return a mock that quacks like a LangChain chat model."""
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(
        content=content,
        usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens},
    )
    return llm


# ── Protocol conformance ────────────────────────────────


def test_langchain_satisfies_protocol():
    adapter = LangChainRAGSystem(llm_provider="openai", llm_model="gpt-4o")
    assert isinstance(adapter, RAGSystem)


# ── Internal-build path ─────────────────────────────────


def test_query_openai_with_callable_retriever():
    adapter = LangChainRAGSystem(
        llm_provider="openai",
        llm_model="gpt-4o",
        retriever=lambda q, k: [f"doc about {q}"] * k,
        top_k=3,
    )
    adapter._llm = _mock_llm()

    result = adapter.query("France?")

    assert result.answer == "Paris"
    assert result.retrieved_docs == ["doc about France?"] * 3
    assert result.metadata["model"] == "gpt-4o"
    assert result.metadata["prompt_tokens"] == 50
    assert result.metadata["completion_tokens"] == 10
    assert result.metadata["top_k"] == 3
    assert result.metadata["latency_ms"] > 0


def test_query_with_base_retriever():
    retriever = MagicMock(spec=BaseRetriever)
    retriever.invoke.return_value = [
        Document(page_content="doc 1"),
        Document(page_content="doc 2"),
        Document(page_content="doc 3"),
    ]
    adapter = LangChainRAGSystem(
        llm_provider="openai",
        llm_model="gpt-4o",
        retriever=retriever,
        top_k=2,
    )
    adapter._llm = _mock_llm()

    result = adapter.query("France?")

    assert result.retrieved_docs == ["doc 1", "doc 2"]
    retriever.invoke.assert_called_once_with("France?")


def test_query_no_retriever_pure_llm():
    adapter = LangChainRAGSystem(llm_provider="openai", llm_model="gpt-4o")
    adapter._llm = _mock_llm()

    result = adapter.query("Capital of France?")

    assert result.answer == "Paris"
    assert result.retrieved_docs == []
    # Prompt should have been built with the no-context placeholder
    call_prompt = adapter._llm.invoke.call_args.args[0]
    assert "(no context provided)" in call_prompt


def test_query_anthropic_path():
    adapter = LangChainRAGSystem(llm_provider="anthropic", llm_model="claude-sonnet-4-20250514")
    adapter._llm = _mock_llm(input_tokens=60, output_tokens=12)

    result = adapter.query("Capital of France?")

    assert result.metadata["prompt_tokens"] == 60
    assert result.metadata["completion_tokens"] == 12
    assert result.metadata["model"] == "claude-sonnet-4-20250514"


def test_query_missing_usage_metadata_defaults_to_zero():
    adapter = LangChainRAGSystem(llm_provider="openai", llm_model="gpt-4o")
    adapter._llm = MagicMock()
    adapter._llm.invoke.return_value = SimpleNamespace(content="Paris", usage_metadata=None)

    result = adapter.query("hi")

    assert result.answer == "Paris"
    assert result.metadata["prompt_tokens"] == 0
    assert result.metadata["completion_tokens"] == 0


def test_invalid_retriever_type_raises():
    adapter = LangChainRAGSystem(
        llm_provider="openai", llm_model="gpt-4o", retriever="not a retriever"
    )
    adapter._llm = _mock_llm()
    with pytest.raises(TypeError, match="retriever must be"):
        adapter.query("hi")


def test_unknown_provider_raises():
    adapter = LangChainRAGSystem(llm_provider="fake", llm_model="x")
    with pytest.raises(ValueError, match="Unknown llm_provider"):
        adapter.query("hi")


# ── Sweep param injection ───────────────────────────────


def test_sweep_params_cast_types():
    adapter = LangChainRAGSystem(
        llm_provider="openai", llm_model="gpt-4o", top_k="10", temperature="0.7"
    )
    assert adapter.top_k == 10
    assert adapter.temperature == 0.7


# ── Chain escape hatch ──────────────────────────────────


def test_chain_returns_string():
    chain = MagicMock()
    chain.invoke.return_value = "Paris"
    adapter = LangChainRAGSystem(chain=chain, llm_model="gpt-4o")

    result = adapter.query("Capital of France?")

    assert result.answer == "Paris"
    assert result.retrieved_docs == []
    assert result.metadata["prompt_tokens"] == 0
    chain.invoke.assert_called_once_with("Capital of France?")


def test_chain_returns_aimessage_like():
    chain = MagicMock()
    chain.invoke.return_value = SimpleNamespace(
        content="Paris",
        usage_metadata={"input_tokens": 42, "output_tokens": 7},
    )
    adapter = LangChainRAGSystem(chain=chain, llm_model="gpt-4o")

    result = adapter.query("q?")

    assert result.answer == "Paris"
    assert result.metadata["prompt_tokens"] == 42
    assert result.metadata["completion_tokens"] == 7


def test_chain_returns_dict():
    chain = MagicMock()
    chain.invoke.return_value = {
        "answer": "Paris",
        "retrieved_docs": ["doc 1", Document(page_content="doc 2")],
        "usage": {"input_tokens": 33, "output_tokens": 4},
    }
    adapter = LangChainRAGSystem(chain=chain, llm_model="gpt-4o")

    result = adapter.query("q?")

    assert result.answer == "Paris"
    assert result.retrieved_docs == ["doc 1", "doc 2"]
    assert result.metadata["prompt_tokens"] == 33
    assert result.metadata["completion_tokens"] == 4


# ── Import guard ────────────────────────────────────────


def test_import_error_when_langchain_core_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langchain_core":
            raise ImportError("No module named 'langchain_core'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="langchain-core package required"):
        LangChainRAGSystem(llm_provider="openai", llm_model="gpt-4o")


# ── Factory function ────────────────────────────────────


def test_create_adapter_langchain():
    adapter = create_adapter("langchain", {"llm_provider": "openai", "llm_model": "gpt-4o"})
    assert isinstance(adapter, LangChainRAGSystem)


def test_create_adapter_langchain_with_sweep_overrides():
    adapter = create_adapter(
        "langchain",
        {"llm_provider": "openai", "llm_model": "gpt-4o", "top_k": 3},
        sweep_overrides={"top_k": 10, "temperature": 0.5},
    )
    assert isinstance(adapter, LangChainRAGSystem)
    assert adapter.top_k == 10
    assert adapter.temperature == 0.5
