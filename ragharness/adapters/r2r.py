from __future__ import annotations

import time
from typing import Any

from ragharness.protocol import RAGResult


class R2RRAGSystem:
    """R2R (SciPhi) adapter: wraps ``r2r.R2RClient``.

    R2R is a server-side RAG system — retrieval and generation both run
    on a remote (or local) R2R instance that has been pre-ingested with
    documents. The adapter marshals sweep parameters (``top_k``,
    ``temperature``, ``llm_model``) into the SDK's ``search_settings``
    and ``rag_generation_config`` shapes.

    Two modes of operation:

    1. **YAML-friendly (default):** specify ``base_url`` and optionally
       ``llm_model``. The adapter constructs ``R2RClient(base_url=...)``
       and invokes ``client.retrieval.rag(...)`` per query.
    2. **Escape hatch (Python only):** pass a pre-built ``R2RClient`` via
       ``client=...``. Useful for cloud auth / session login cases where
       the client needs to carry state that YAML can't express.

    Parameters
    ----------
    base_url:
        URL of the running R2R server. Ignored when ``client`` is given.
    llm_model:
        LiteLLM-style model identifier (e.g. ``"openai/gpt-4o-mini"``).
        Forwarded to R2R as ``rag_generation_config["model"]``. When
        ``None``, R2R falls back to its server-side default.
    top_k:
        Number of chunks to retrieve. Forwarded as
        ``search_settings["limit"]``.
    temperature:
        Sampling temperature forwarded to the LLM.
    max_tokens:
        Upper bound on generated tokens.
    search_mode:
        R2R search mode (``"custom"`` / ``"advanced"`` / ``"basic"``).
    search_settings:
        Extra ``search_settings`` passed through verbatim. ``top_k`` /
        ``limit`` set here is overridden by the sweep's ``top_k``.
    rag_generation_config:
        Extra ``rag_generation_config`` passed through verbatim.
        ``temperature`` / ``max_tokens`` / ``model`` set here are
        overridden by the sweep's values when provided.
    client:
        Pre-built ``R2RClient``. When given, bypasses the internal build
        path entirely.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7272",
        llm_model: str | None = None,
        top_k: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        search_mode: str = "custom",
        search_settings: dict[str, Any] | None = None,
        rag_generation_config: dict[str, Any] | None = None,
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        self.base_url = base_url
        self.llm_model = llm_model
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.search_mode = search_mode
        self.search_settings = search_settings or {}
        self.rag_generation_config = rag_generation_config or {}
        self._client: Any = client
        self._extra = kwargs

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from r2r import R2RClient
        except ImportError:
            raise ImportError(
                "r2r package required. Install with: pip install ragharness[r2r]"
            ) from None

        self._client = R2RClient(base_url=self.base_url)
        return self._client

    def query(self, question: str) -> RAGResult:
        search_settings: dict[str, Any] = {**self.search_settings, "limit": self.top_k}
        gen_config: dict[str, Any] = {
            **self.rag_generation_config,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.llm_model is not None:
            gen_config["model"] = self.llm_model

        client = self._get_client()

        start = time.perf_counter()
        wrapped = client.retrieval.rag(
            query=question,
            search_settings=search_settings,
            rag_generation_config=gen_config,
            search_mode=self.search_mode,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        resp = getattr(wrapped, "results", wrapped)

        answer = getattr(resp, "generated_answer", "") or ""

        search_results = getattr(resp, "search_results", None)
        chunks = getattr(search_results, "chunk_search_results", None) or []
        docs = [getattr(c, "text", str(c)) for c in chunks]

        metadata = getattr(resp, "metadata", {}) or {}
        usage = metadata.get("usage") if isinstance(metadata, dict) else None
        prompt_tokens = int(usage.get("input_tokens", 0)) if isinstance(usage, dict) else 0
        completion_tokens = int(usage.get("output_tokens", 0)) if isinstance(usage, dict) else 0

        return RAGResult(
            answer=str(answer),
            retrieved_docs=docs,
            metadata={
                "latency_ms": elapsed_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model": self.llm_model or "r2r-default",
                "top_k": self.top_k,
            },
        )
