from __future__ import annotations

import time
from typing import Any

from ragharness.protocol import RAGResult


class LlamaIndexRAGSystem:
    """LlamaIndex-backed RAG adapter.

    LlamaIndex is Python-first — the primary path is to hand in a
    pre-built ``index`` or ``query_engine``; the adapter just drives it.

    Three ways to configure, in priority order:

    1. **Pre-built query_engine (Python only):** pass
       ``query_engine=<BaseQueryEngine>``. The adapter invokes it
       verbatim. ``top_k`` / ``llm_model`` on the adapter are ignored
       since they are already baked into the engine.
    2. **Pre-built index (Python only):** pass ``index=<BaseIndex>``.
       The adapter calls ``index.as_query_engine(similarity_top_k=top_k,
       llm=<llm>)`` per query, so sweep params take effect.
    3. **Documents directory (YAML-friendly):** set
       ``documents_path="./docs/"``. The adapter builds a
       ``VectorStoreIndex`` from the directory on first query and caches
       it. Requires LlamaIndex's default embeddings provider
       (``OPENAI_API_KEY`` by default).

    Parameters
    ----------
    query_engine, index:
        Pre-built LlamaIndex objects; see modes above.
    documents_path:
        Directory readable by ``SimpleDirectoryReader``.
    llm_provider:
        ``"openai"`` or ``"anthropic"``. Used when building a query
        engine from an index or documents.
    llm_model:
        Model identifier (e.g. ``"gpt-4o"``, ``"claude-sonnet-4-20250514"``).
    top_k:
        ``similarity_top_k`` forwarded to the query engine.
    temperature:
        Sampling temperature forwarded to the LLM.
    """

    def __init__(
        self,
        query_engine: Any = None,
        index: Any = None,
        documents_path: str | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        top_k: int = 5,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self._query_engine = query_engine
        self._index = index
        self.documents_path = documents_path
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self._extra = kwargs

    def _build_llm(self) -> Any:
        """Construct a LlamaIndex LLM wrapper based on llm_provider/model."""
        if not self.llm_provider or not self.llm_model:
            return None

        if self.llm_provider == "openai":
            try:
                from llama_index.llms.openai import OpenAI
            except ImportError:
                raise ImportError(
                    "llama-index-llms-openai package required. "
                    "Install with: pip install ragharness[llamaindex]"
                ) from None
            return OpenAI(model=self.llm_model, temperature=self.temperature)

        if self.llm_provider == "anthropic":
            try:
                from llama_index.llms.anthropic import Anthropic
            except ImportError:
                raise ImportError(
                    "llama-index-llms-anthropic package required. "
                    "Install with: pip install llama-index-llms-anthropic"
                ) from None
            return Anthropic(model=self.llm_model, temperature=self.temperature)

        raise ValueError(f"Unknown llm_provider: {self.llm_provider!r}")

    def _load_index_from_documents(self) -> Any:
        try:
            from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
        except ImportError:
            raise ImportError(
                "llama-index package required. Install with: pip install ragharness[llamaindex]"
            ) from None

        documents = SimpleDirectoryReader(self.documents_path).load_data()
        return VectorStoreIndex.from_documents(documents)

    def _get_query_engine(self) -> Any:
        if self._query_engine is not None:
            return self._query_engine

        if self._index is None:
            if self.documents_path is None:
                raise ValueError(
                    "LlamaIndexRAGSystem requires one of: query_engine=..., "
                    "index=..., or documents_path=... . None were provided."
                )
            self._index = self._load_index_from_documents()

        llm = self._build_llm()
        if llm is not None:
            return self._index.as_query_engine(similarity_top_k=self.top_k, llm=llm)
        return self._index.as_query_engine(similarity_top_k=self.top_k)

    def query(self, question: str) -> RAGResult:
        engine = self._get_query_engine()

        start = time.perf_counter()
        response = engine.query(question)
        elapsed_ms = (time.perf_counter() - start) * 1000

        answer = getattr(response, "response", None)
        if not answer:
            answer = str(response)

        source_nodes = getattr(response, "source_nodes", None) or []
        docs: list[str] = []
        for n in source_nodes:
            node = getattr(n, "node", n)
            if hasattr(node, "get_content"):
                docs.append(str(node.get_content()))
            else:
                docs.append(str(getattr(node, "text", node)))

        return RAGResult(
            answer=str(answer),
            retrieved_docs=docs,
            metadata={
                "latency_ms": elapsed_ms,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "model": self.llm_model or "llamaindex-default",
                "top_k": self.top_k,
            },
        )
