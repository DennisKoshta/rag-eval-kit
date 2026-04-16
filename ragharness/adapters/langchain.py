from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, cast

from ragharness.protocol import RAGResult

DEFAULT_RAG_PROMPT = """\
Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""


class LangChainRAGSystem:
    """LangChain-backed RAG adapter.

    Two modes of operation:

    1. **Internal-build (YAML-friendly):** specify ``llm_provider`` and
       ``llm_model``; the adapter constructs a ``ChatOpenAI`` or
       ``ChatAnthropic`` LLM, runs retrieval (via callable or
       ``BaseRetriever``), formats a prompt, and calls the LLM.
    2. **Escape hatch (Python only):** pass a pre-built LangChain
       ``Runnable`` via ``chain=...``; ``query`` invokes it directly and
       normalises the output. Supported outputs: ``str``, ``AIMessage``,
       or ``dict`` with ``answer`` / ``retrieved_docs`` keys.

    Parameters
    ----------
    llm_provider:
        ``"openai"`` or ``"anthropic"``. Required unless ``chain`` is given.
    llm_model:
        Model identifier (e.g. ``"gpt-4o"``, ``"claude-sonnet-4-20250514"``).
    retriever:
        Either a callable ``(query, top_k) -> list[str]`` or a LangChain
        ``BaseRetriever``. ``None`` disables retrieval.
    prompt_template:
        Format string with ``{context}`` and ``{question}`` placeholders.
    top_k:
        Number of documents to retrieve.
    temperature:
        Sampling temperature forwarded to the LLM.
    chain:
        Pre-built LangChain ``Runnable``. When provided, bypasses the
        internal build path.
    """

    def __init__(
        self,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        retriever: Any = None,
        prompt_template: str = DEFAULT_RAG_PROMPT,
        top_k: int = 5,
        temperature: float = 0.0,
        chain: Any = None,
        **kwargs: Any,
    ) -> None:
        try:
            import langchain_core  # noqa: F401
        except ImportError:
            raise ImportError(
                "langchain-core package required. Install with: pip install ragharness[langchain]"
            ) from None

        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.retriever = retriever
        self.prompt_template = prompt_template
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self._chain = chain
        self._extra = kwargs
        self._llm: Any = None

    def _get_llm(self) -> Any:
        if self._llm is not None:
            return self._llm

        if not self.llm_model:
            raise ValueError("llm_model is required when chain is not provided")

        if self.llm_provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise ImportError(
                    "langchain-openai package required. "
                    "Install with: pip install ragharness[langchain]"
                ) from None
            self._llm = ChatOpenAI(model=self.llm_model, temperature=self.temperature)
        elif self.llm_provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError:
                raise ImportError(
                    "langchain-anthropic package required. "
                    "Install with: pip install ragharness[langchain]"
                ) from None
            self._llm = ChatAnthropic(
                model_name=self.llm_model,
                temperature=self.temperature,
                max_tokens_to_sample=1024,
                timeout=None,
                stop=None,
            )
        else:
            raise ValueError(f"Unknown llm_provider: {self.llm_provider!r}")

        return self._llm

    def _retrieve(self, question: str) -> list[str]:
        if self.retriever is None:
            return []

        from langchain_core.retrievers import BaseRetriever

        if isinstance(self.retriever, BaseRetriever):
            docs = self.retriever.invoke(question)[: self.top_k]
            return [getattr(d, "page_content", str(d)) for d in docs]

        if callable(self.retriever):
            fn = cast(Callable[[str, int], list[str]], self.retriever)
            return list(fn(question, self.top_k))

        raise TypeError(
            f"retriever must be a callable or BaseRetriever, got {type(self.retriever).__name__}"
        )

    def query(self, question: str) -> RAGResult:
        start = time.perf_counter()

        if self._chain is not None:
            answer, docs, prompt_tokens, completion_tokens = self._invoke_chain(question)
        else:
            docs = self._retrieve(question)
            context = "\n\n".join(docs) if docs else "(no context provided)"
            prompt = self.prompt_template.format(context=context, question=question)

            msg = self._get_llm().invoke(prompt)
            answer = getattr(msg, "content", str(msg)) or ""
            usage = getattr(msg, "usage_metadata", None) or {}
            prompt_tokens = int(usage.get("input_tokens", 0))
            completion_tokens = int(usage.get("output_tokens", 0))

        elapsed_ms = (time.perf_counter() - start) * 1000

        return RAGResult(
            answer=answer,
            retrieved_docs=docs,
            metadata={
                "latency_ms": elapsed_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model": self.llm_model or "unknown",
                "top_k": self.top_k,
            },
        )

    def _invoke_chain(self, question: str) -> tuple[str, list[str], int, int]:
        """Invoke a user-provided Runnable and normalise the output.

        Token usage is best-effort: populated when the chain returns an
        ``AIMessage`` with ``usage_metadata`` or a dict with a ``usage``
        key. Otherwise zero.
        """
        output = self._chain.invoke(question)

        if isinstance(output, str):
            return output, [], 0, 0

        if isinstance(output, dict):
            answer = str(output.get("answer", ""))
            docs_raw = output.get("retrieved_docs", []) or []
            docs = [
                d if isinstance(d, str) else getattr(d, "page_content", str(d)) for d in docs_raw
            ]
            usage = output.get("usage") or {}
            return (
                answer,
                docs,
                int(usage.get("input_tokens", 0)),
                int(usage.get("output_tokens", 0)),
            )

        content = getattr(output, "content", None)
        if content is not None:
            usage = getattr(output, "usage_metadata", None) or {}
            return (
                str(content),
                [],
                int(usage.get("input_tokens", 0)),
                int(usage.get("output_tokens", 0)),
            )

        return str(output), [], 0, 0
