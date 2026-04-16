from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ragharness.adapters import create_adapter
from ragharness.adapters.llamaindex import LlamaIndexRAGSystem
from ragharness.protocol import RAGSystem

# ── Helpers ──────────────────────────────────────────────


def _mock_response(answer: str = "Paris", docs: list[str] | None = None) -> SimpleNamespace:
    """Return a mock object shaped like a LlamaIndex Response."""
    docs = docs if docs is not None else ["France's capital is Paris."]
    return SimpleNamespace(
        response=answer,
        source_nodes=[
            SimpleNamespace(node=SimpleNamespace(get_content=lambda t=t: t)) for t in docs
        ],
    )


def _mock_query_engine(response: SimpleNamespace | None = None) -> MagicMock:
    engine = MagicMock()
    engine.query.return_value = response or _mock_response()
    return engine


# ── Protocol conformance ────────────────────────────────


def test_llamaindex_satisfies_protocol():
    adapter = LlamaIndexRAGSystem(query_engine=_mock_query_engine())
    assert isinstance(adapter, RAGSystem)


# ── Pre-built query_engine path ─────────────────────────


def test_query_with_pre_built_query_engine():
    adapter = LlamaIndexRAGSystem(query_engine=_mock_query_engine())

    result = adapter.query("Capital of France?")

    assert result.answer == "Paris"
    assert result.retrieved_docs == ["France's capital is Paris."]
    assert result.metadata["latency_ms"] >= 0
    assert result.metadata["top_k"] == 5


def test_query_with_multiple_source_nodes():
    resp = _mock_response(
        answer="multi",
        docs=["doc a", "doc b", "doc c"],
    )
    adapter = LlamaIndexRAGSystem(query_engine=_mock_query_engine(resp))

    result = adapter.query("hi")
    assert result.retrieved_docs == ["doc a", "doc b", "doc c"]


def test_query_handles_missing_source_nodes():
    resp = SimpleNamespace(response="only text", source_nodes=None)
    adapter = LlamaIndexRAGSystem(query_engine=_mock_query_engine(resp))

    result = adapter.query("hi")
    assert result.answer == "only text"
    assert result.retrieved_docs == []


def test_query_answer_falls_back_to_str():
    """If .response attr is missing/empty, adapter falls back to str(response)."""

    class Stringy:
        response = None
        source_nodes = []

        def __str__(self) -> str:
            return "str-fallback"

    adapter = LlamaIndexRAGSystem(query_engine=_mock_query_engine(Stringy()))
    result = adapter.query("hi")
    assert result.answer == "str-fallback"


def test_query_handles_node_without_get_content():
    """TextNode exposes .text; older/alt nodes lack get_content()."""
    resp = SimpleNamespace(
        response="hi",
        source_nodes=[SimpleNamespace(node=SimpleNamespace(text="plain-text"))],
    )
    adapter = LlamaIndexRAGSystem(query_engine=_mock_query_engine(resp))

    result = adapter.query("hi")
    assert result.retrieved_docs == ["plain-text"]


# ── Pre-built index path ────────────────────────────────


def test_query_with_index_invokes_as_query_engine():
    index = MagicMock()
    index.as_query_engine.return_value = _mock_query_engine()

    adapter = LlamaIndexRAGSystem(index=index, top_k=7)
    adapter.query("hi")

    index.as_query_engine.assert_called_once_with(similarity_top_k=7)


def test_query_with_index_and_llm_passes_llm():
    """When llm_provider/llm_model are set, a built LLM is forwarded."""
    index = MagicMock()
    index.as_query_engine.return_value = _mock_query_engine()

    fake_llm = object()
    adapter = LlamaIndexRAGSystem(
        index=index,
        llm_provider="openai",
        llm_model="gpt-4o",
        top_k=5,
    )
    # Stub LLM construction so no real import/ctor is needed
    adapter._build_llm = lambda: fake_llm  # type: ignore[method-assign]

    adapter.query("hi")

    index.as_query_engine.assert_called_once_with(similarity_top_k=5, llm=fake_llm)


# ── Missing source raises ───────────────────────────────


def test_missing_source_raises():
    adapter = LlamaIndexRAGSystem()
    with pytest.raises(ValueError, match="requires one of"):
        adapter.query("hi")


# ── Sweep param casting ─────────────────────────────────


def test_sweep_params_cast_types():
    adapter = LlamaIndexRAGSystem(query_engine=_mock_query_engine(), top_k="10", temperature="0.7")
    assert adapter.top_k == 10
    assert adapter.temperature == 0.7


# ── Documents path builds index (mocked) ────────────────


def _install_fake_llama_index_core(monkeypatch, fake_index, fake_reader) -> None:
    """Register a stubbed llama_index.core so `from llama_index.core import ...` works."""
    fake_module = SimpleNamespace(
        SimpleDirectoryReader=fake_reader,
        VectorStoreIndex=SimpleNamespace(from_documents=lambda _docs: fake_index),
    )
    monkeypatch.setitem(sys.modules, "llama_index", SimpleNamespace(core=fake_module))
    monkeypatch.setitem(sys.modules, "llama_index.core", fake_module)


def test_documents_path_builds_index(monkeypatch):
    """documents_path triggers SimpleDirectoryReader + VectorStoreIndex path."""
    fake_index = MagicMock()
    fake_index.as_query_engine.return_value = _mock_query_engine()
    fake_reader = MagicMock()
    fake_reader.return_value.load_data.return_value = ["doc1"]

    _install_fake_llama_index_core(monkeypatch, fake_index, fake_reader)

    adapter = LlamaIndexRAGSystem(documents_path="./docs")
    adapter.query("hi")

    fake_reader.assert_called_once_with("./docs")
    fake_index.as_query_engine.assert_called_once()


def test_documents_path_index_cached(monkeypatch):
    """Index is built once and reused across multiple queries."""
    fake_index = MagicMock()
    fake_index.as_query_engine.return_value = _mock_query_engine()
    fake_reader = MagicMock()
    fake_reader.return_value.load_data.return_value = ["doc1"]

    _install_fake_llama_index_core(monkeypatch, fake_index, fake_reader)

    adapter = LlamaIndexRAGSystem(documents_path="./docs")
    adapter.query("q1")
    adapter.query("q2")
    adapter.query("q3")

    # Reader only runs once
    fake_reader.assert_called_once()


# ── LLM construction ────────────────────────────────────


def test_build_llm_none_when_provider_missing():
    adapter = LlamaIndexRAGSystem(query_engine=_mock_query_engine(), llm_provider=None)
    assert adapter._build_llm() is None


def test_build_llm_unknown_provider():
    adapter = LlamaIndexRAGSystem(
        query_engine=_mock_query_engine(),
        llm_provider="fake",
        llm_model="something",
    )
    with pytest.raises(ValueError, match="Unknown llm_provider"):
        adapter._build_llm()


# ── Import error guard ──────────────────────────────────


def test_import_error_hint(monkeypatch):
    """When llama_index.core is unavailable, error message points at the extra."""
    adapter = LlamaIndexRAGSystem(documents_path="./docs")
    monkeypatch.setitem(sys.modules, "llama_index.core", None)

    with pytest.raises(ImportError, match=r"ragharness\[llamaindex\]"):
        adapter._load_index_from_documents()


# ── Factory function ────────────────────────────────────


def test_create_adapter_llamaindex():
    # Factory builds the object; no query happens, so no engine needed
    adapter = create_adapter("llamaindex", {"documents_path": "./docs"})
    assert isinstance(adapter, LlamaIndexRAGSystem)


def test_create_adapter_with_sweep_overrides():
    adapter = create_adapter(
        "llamaindex",
        {"documents_path": "./docs", "top_k": 3},
        sweep_overrides={"top_k": 10, "temperature": 0.5},
    )
    assert isinstance(adapter, LlamaIndexRAGSystem)
    assert adapter.top_k == 10
    assert adapter.temperature == 0.5
