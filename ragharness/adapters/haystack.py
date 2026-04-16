from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ragharness.protocol import RAGResult

DEFAULT_PROMPT_TEMPLATE = """\
Given the following documents, answer the question.

Documents:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Question: {{ question }}
Answer:
"""


class HaystackRAGSystem:
    """Haystack 2.x-backed RAG adapter.

    Haystack's core primitive is a ``Pipeline`` composed of named
    components (retriever, prompt_builder, generator, ...). The adapter
    either drives a pre-built pipeline or builds a simple BM25 +
    OpenAIGenerator pipeline on demand.

    Two modes of operation:

    1. **Escape hatch (Python only):** pass ``pipeline=<Pipeline>``.
       The adapter invokes ``pipeline.run(...)`` per query. Component
       names default to ``retriever`` / ``prompt_builder`` /
       ``generator`` but can be overridden. For non-standard input
       shapes (e.g. embedding pipelines), pass ``run_data_builder`` —
       a callable ``(question, top_k) -> dict`` that returns the full
       ``pipeline.run`` input.
    2. **Internal-build (YAML-friendly):** set ``documents_path`` to a
       directory; the adapter reads each ``*.txt`` as a Haystack
       ``Document`` and wires a BM25 retriever → PromptBuilder →
       OpenAIGenerator pipeline. Requires ``OPENAI_API_KEY``.

    Parameters
    ----------
    pipeline:
        Pre-built Haystack ``Pipeline``.
    documents_path:
        Directory of ``*.txt`` files to index with BM25.
    llm_model:
        OpenAI model identifier used by the built-in pipeline.
    top_k:
        Forwarded to the retriever as ``top_k``.
    temperature:
        Sampling temperature for the built-in generator.
    retriever_component, prompt_builder_component, generator_component:
        Names of the components in a user-supplied pipeline.
    run_data_builder:
        Callable ``(question, top_k) -> dict`` used for custom pipeline
        shapes. Defaults to the retriever/prompt_builder convention.
    prompt_template:
        Jinja template used by the built-in ``PromptBuilder``.
    """

    def __init__(
        self,
        pipeline: Any = None,
        documents_path: str | None = None,
        llm_model: str = "gpt-4o-mini",
        top_k: int = 5,
        temperature: float = 0.0,
        retriever_component: str = "retriever",
        prompt_builder_component: str = "prompt_builder",
        generator_component: str = "generator",
        run_data_builder: Callable[[str, int], dict[str, Any]] | None = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        **kwargs: Any,
    ) -> None:
        self._pipeline = pipeline
        self.documents_path = documents_path
        self.llm_model = llm_model
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.retriever_component = retriever_component
        self.prompt_builder_component = prompt_builder_component
        self.generator_component = generator_component
        self.run_data_builder = run_data_builder
        self.prompt_template = prompt_template
        self._extra = kwargs

    def _build_pipeline_from_documents(self) -> Any:
        try:
            from haystack import Pipeline
            from haystack.components.builders import PromptBuilder
            from haystack.components.generators import OpenAIGenerator
            from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
            from haystack.dataclasses import Document
            from haystack.document_stores.in_memory import InMemoryDocumentStore
        except ImportError:
            raise ImportError(
                "haystack-ai package required. Install with: pip install ragharness[haystack]"
            ) from None

        import os

        store = InMemoryDocumentStore()
        docs: list[Any] = []
        if self.documents_path is not None:
            for root, _dirs, files in os.walk(self.documents_path):
                for fname in files:
                    if fname.endswith(".txt"):
                        path = os.path.join(root, fname)
                        with open(path, encoding="utf-8") as f:
                            docs.append(Document(content=f.read(), meta={"source": path}))
        store.write_documents(docs)

        generator_kwargs: dict[str, Any] = {
            "model": self.llm_model,
            "generation_kwargs": {"temperature": self.temperature},
        }

        pipe = Pipeline()
        pipe.add_component(self.retriever_component, InMemoryBM25Retriever(store))
        pipe.add_component(
            self.prompt_builder_component, PromptBuilder(template=self.prompt_template)
        )
        pipe.add_component(self.generator_component, OpenAIGenerator(**generator_kwargs))

        pipe.connect(
            f"{self.retriever_component}.documents",
            f"{self.prompt_builder_component}.documents",
        )
        pipe.connect(
            f"{self.prompt_builder_component}.prompt",
            f"{self.generator_component}.prompt",
        )
        return pipe

    def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        if self.documents_path is None:
            raise ValueError(
                "HaystackRAGSystem requires one of: pipeline=... or documents_path=... . "
                "Neither was provided."
            )

        self._pipeline = self._build_pipeline_from_documents()
        return self._pipeline

    def _default_run_data(self, question: str) -> dict[str, Any]:
        return {
            self.retriever_component: {"query": question, "top_k": self.top_k},
            self.prompt_builder_component: {"question": question},
        }

    def query(self, question: str) -> RAGResult:
        pipeline = self._get_pipeline()

        if self.run_data_builder is not None:
            run_data = self.run_data_builder(question, self.top_k)
        else:
            run_data = self._default_run_data(question)

        start = time.perf_counter()
        result = pipeline.run(run_data, include_outputs_from={self.retriever_component})
        elapsed_ms = (time.perf_counter() - start) * 1000

        gen_out = result.get(self.generator_component, {}) or {}
        replies = gen_out.get("replies") or []
        if replies:
            reply = replies[0]
            answer = getattr(reply, "text", None) or getattr(reply, "content", None) or str(reply)
        else:
            answer = ""

        retriever_out = result.get(self.retriever_component, {}) or {}
        raw_docs = retriever_out.get("documents") or []
        docs = [getattr(d, "content", str(d)) for d in raw_docs]

        meta = gen_out.get("meta") or []
        usage = meta[0].get("usage", {}) if meta and isinstance(meta[0], dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens", 0)) if isinstance(usage, dict) else 0
        completion_tokens = (
            int(usage.get("completion_tokens", 0)) if isinstance(usage, dict) else 0
        )

        return RAGResult(
            answer=str(answer),
            retrieved_docs=docs,
            metadata={
                "latency_ms": elapsed_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model": self.llm_model,
                "top_k": self.top_k,
            },
        )
