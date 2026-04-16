from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ragharness.adapters import create_adapter
from ragharness.adapters.haystack import HaystackRAGSystem
from ragharness.protocol import RAGSystem

# ── Helpers ──────────────────────────────────────────────


def _mock_pipeline_result(
    answer: str = "Paris",
    docs: list[str] | None = None,
    prompt_tokens: int = 40,
    completion_tokens: int = 8,
) -> dict:
    """Shape returned by Pipeline.run() with include_outputs_from={retriever}."""
    docs = docs if docs is not None else ["France's capital is Paris."]
    return {
        "retriever": {"documents": [SimpleNamespace(content=t) for t in docs]},
        "generator": {
            "replies": [answer],
            "meta": [
                {
                    "model": "gpt-4o-mini",
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                }
            ],
        },
    }


def _mock_pipeline(result: dict | None = None) -> MagicMock:
    pipe = MagicMock()
    pipe.run.return_value = result if result is not None else _mock_pipeline_result()
    return pipe


# ── Protocol conformance ────────────────────────────────


def test_haystack_satisfies_protocol():
    adapter = HaystackRAGSystem(pipeline=_mock_pipeline())
    assert isinstance(adapter, RAGSystem)


# ── Pre-built pipeline path ─────────────────────────────


def test_query_with_pre_built_pipeline():
    adapter = HaystackRAGSystem(pipeline=_mock_pipeline())

    result = adapter.query("Capital of France?")

    assert result.answer == "Paris"
    assert result.retrieved_docs == ["France's capital is Paris."]
    assert result.metadata["prompt_tokens"] == 40
    assert result.metadata["completion_tokens"] == 8
    assert result.metadata["latency_ms"] >= 0


def test_query_default_run_data_shape():
    """Default routing: {retriever: {query, top_k}, prompt_builder: {question}}."""
    pipe = _mock_pipeline()
    adapter = HaystackRAGSystem(pipeline=pipe, top_k=7)

    adapter.query("Capital?")

    call_args = pipe.run.call_args
    run_data = call_args.args[0]
    assert run_data["retriever"] == {"query": "Capital?", "top_k": 7}
    assert run_data["prompt_builder"] == {"question": "Capital?"}


def test_query_include_outputs_from_retriever():
    pipe = _mock_pipeline()
    adapter = HaystackRAGSystem(pipeline=pipe)

    adapter.query("hi")

    include = pipe.run.call_args.kwargs["include_outputs_from"]
    assert include == {"retriever"}


def test_custom_run_data_builder_used():
    pipe = _mock_pipeline()
    calls: list[tuple[str, int]] = []

    def builder(q: str, k: int) -> dict:
        calls.append((q, k))
        return {"text_embedder": {"text": q}, "prompt_builder": {"question": q}}

    adapter = HaystackRAGSystem(pipeline=pipe, run_data_builder=builder, top_k=5)
    adapter.query("hello")

    assert calls == [("hello", 5)]
    run_data = pipe.run.call_args.args[0]
    assert run_data == {
        "text_embedder": {"text": "hello"},
        "prompt_builder": {"question": "hello"},
    }


def test_custom_component_names():
    pipe = _mock_pipeline(
        result={
            "my_retriever": {"documents": [SimpleNamespace(content="doc-x")]},
            "my_llm": {"replies": ["hi"], "meta": [{"usage": {}}]},
        }
    )
    adapter = HaystackRAGSystem(
        pipeline=pipe,
        retriever_component="my_retriever",
        generator_component="my_llm",
    )

    result = adapter.query("q?")
    assert result.answer == "hi"
    assert result.retrieved_docs == ["doc-x"]


# ── Response edge cases ─────────────────────────────────


def test_query_with_chat_message_reply():
    """ChatGenerator returns ChatMessage-shaped replies (.text attr)."""
    pipe = _mock_pipeline(
        result={
            "retriever": {"documents": []},
            "generator": {
                "replies": [SimpleNamespace(text="chat-answer")],
                "meta": [{"usage": {}}],
            },
        }
    )
    adapter = HaystackRAGSystem(pipeline=pipe)

    result = adapter.query("hi")
    assert result.answer == "chat-answer"


def test_query_no_replies_empty_answer():
    pipe = _mock_pipeline(
        result={"retriever": {"documents": []}, "generator": {"replies": [], "meta": []}}
    )
    adapter = HaystackRAGSystem(pipeline=pipe)

    result = adapter.query("hi")
    assert result.answer == ""
    assert result.metadata["prompt_tokens"] == 0


def test_query_without_retriever_output():
    """include_outputs_from may fail / user pipeline may skip — retrieved_docs stays []."""
    pipe = _mock_pipeline(result={"generator": {"replies": ["ok"], "meta": [{"usage": {}}]}})
    adapter = HaystackRAGSystem(pipeline=pipe)

    result = adapter.query("hi")
    assert result.retrieved_docs == []
    assert result.answer == "ok"


# ── Missing source raises ───────────────────────────────


def test_missing_source_raises():
    adapter = HaystackRAGSystem()
    with pytest.raises(ValueError, match="requires one of"):
        adapter.query("hi")


# ── Sweep param casting ─────────────────────────────────


def test_sweep_params_cast_types():
    adapter = HaystackRAGSystem(pipeline=_mock_pipeline(), top_k="10", temperature="0.7")
    assert adapter.top_k == 10
    assert adapter.temperature == 0.7


# ── documents_path builds pipeline (mocked) ─────────────


def _install_fake_haystack(monkeypatch, built_pipeline):
    """Register stubbed haystack modules so the internal-build path runs."""
    fake_pipeline_cls = MagicMock(return_value=built_pipeline)

    def _make_doc(content, meta=None):
        return SimpleNamespace(content=content, meta=meta)

    fake_doc_cls = MagicMock(side_effect=_make_doc)
    fake_store = MagicMock()
    fake_store_cls = MagicMock(return_value=fake_store)
    fake_retriever_cls = MagicMock()
    fake_prompt_builder_cls = MagicMock()
    fake_generator_cls = MagicMock()

    modules = {
        "haystack": SimpleNamespace(Pipeline=fake_pipeline_cls),
        "haystack.components": SimpleNamespace(),
        "haystack.components.builders": SimpleNamespace(PromptBuilder=fake_prompt_builder_cls),
        "haystack.components.generators": SimpleNamespace(OpenAIGenerator=fake_generator_cls),
        "haystack.components.retrievers": SimpleNamespace(),
        "haystack.components.retrievers.in_memory": SimpleNamespace(
            InMemoryBM25Retriever=fake_retriever_cls
        ),
        "haystack.dataclasses": SimpleNamespace(Document=fake_doc_cls),
        "haystack.document_stores": SimpleNamespace(),
        "haystack.document_stores.in_memory": SimpleNamespace(
            InMemoryDocumentStore=fake_store_cls
        ),
    }
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)

    return SimpleNamespace(
        pipeline_cls=fake_pipeline_cls,
        store=fake_store,
        doc_cls=fake_doc_cls,
        retriever_cls=fake_retriever_cls,
        generator_cls=fake_generator_cls,
    )


def test_documents_path_builds_pipeline(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("Paris is the capital of France.")
    (tmp_path / "b.txt").write_text("Rome is the capital of Italy.")

    built_pipeline = _mock_pipeline()
    fakes = _install_fake_haystack(monkeypatch, built_pipeline)

    adapter = HaystackRAGSystem(documents_path=str(tmp_path), llm_model="gpt-4o-mini")
    adapter.query("hi")

    # Pipeline constructed and used
    fakes.pipeline_cls.assert_called_once()
    # Generator configured with the right model
    assert fakes.generator_cls.call_args.kwargs["model"] == "gpt-4o-mini"
    # Document store received 2 documents
    write_call = fakes.store.write_documents.call_args
    assert len(write_call.args[0]) == 2


def test_documents_path_pipeline_cached(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("text")

    built_pipeline = _mock_pipeline()
    fakes = _install_fake_haystack(monkeypatch, built_pipeline)

    adapter = HaystackRAGSystem(documents_path=str(tmp_path))
    adapter.query("q1")
    adapter.query("q2")
    adapter.query("q3")

    fakes.pipeline_cls.assert_called_once()


# ── Import error guard ──────────────────────────────────


def test_import_error_hint(monkeypatch):
    adapter = HaystackRAGSystem(documents_path="./docs")
    monkeypatch.setitem(sys.modules, "haystack", None)

    with pytest.raises(ImportError, match=r"ragharness\[haystack\]"):
        adapter._build_pipeline_from_documents()


# ── Factory function ────────────────────────────────────


def test_create_adapter_haystack():
    adapter = create_adapter("haystack", {"documents_path": "./docs"})
    assert isinstance(adapter, HaystackRAGSystem)


def test_create_adapter_with_sweep_overrides():
    adapter = create_adapter(
        "haystack",
        {"documents_path": "./docs", "top_k": 3},
        sweep_overrides={"top_k": 10, "temperature": 0.5},
    )
    assert isinstance(adapter, HaystackRAGSystem)
    assert adapter.top_k == 10
    assert adapter.temperature == 0.5
