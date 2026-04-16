from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ragharness.protocol import RAGResult

DEFAULT_RAG_PROMPT = """\
Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""


class RawRAGSystem:
    """Direct-API adapter: wires a retriever callable to an LLM client.

    This is the "escape hatch" adapter — users supply their own retriever
    function and an LLM provider, and this class glues them together
    behind the ``RAGSystem`` protocol.

    Parameters
    ----------
    llm_provider:
        ``"openai"`` or ``"anthropic"``.
    llm_model:
        Model identifier (e.g. ``"gpt-4o"``, ``"claude-sonnet-4-20250514"``).
    retriever:
        Optional callable ``(query, top_k) -> list[str]``.  When *None*
        the adapter runs in pure-generation mode (no retrieval).
    prompt_template:
        A format string with ``{context}`` and ``{question}`` placeholders.
    top_k:
        Number of documents to retrieve.
    temperature:
        Sampling temperature forwarded to the LLM.
    seed:
        Optional integer seed for reproducibility. Forwarded to OpenAI's
        ``chat.completions.create(seed=...)``; silently ignored by Anthropic,
        which does not expose a seed parameter.
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        retriever: Callable[[str, int], list[str]] | None = None,
        prompt_template: str = DEFAULT_RAG_PROMPT,
        top_k: int = 5,
        temperature: float = 0.0,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.retriever = retriever
        self.prompt_template = prompt_template
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.seed = int(seed) if seed is not None else None
        self._extra = kwargs
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if self.llm_provider == "openai":
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install ragharness[openai]"
                ) from None
            self._client = openai.OpenAI()
        elif self.llm_provider == "anthropic":
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: pip install ragharness[anthropic]"
                ) from None
            self._client = anthropic.Anthropic()
        else:
            raise ValueError(f"Unknown llm_provider: {self.llm_provider!r}")

        return self._client

    def query(self, question: str) -> RAGResult:
        start = time.perf_counter()

        # Retrieval
        docs: list[str] = []
        if self.retriever is not None:
            docs = self.retriever(question, self.top_k)

        # Build prompt
        context = "\n\n".join(docs) if docs else "(no context provided)"
        prompt = self.prompt_template.format(context=context, question=question)

        # LLM call
        client = self._get_client()
        answer, prompt_tokens, completion_tokens = self._call_llm(client, prompt)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return RAGResult(
            answer=answer,
            retrieved_docs=docs,
            metadata={
                "latency_ms": elapsed_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model": self.llm_model,
                "top_k": self.top_k,
            },
        )

    def _call_llm(self, client: Any, prompt: str) -> tuple[str, int, int]:
        """Call the LLM and return (answer, prompt_tokens, completion_tokens)."""
        if self.llm_provider == "openai":
            openai_kwargs: dict[str, Any] = {
                "model": self.llm_model,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if self.seed is not None:
                openai_kwargs["seed"] = self.seed
            resp = client.chat.completions.create(**openai_kwargs)
            answer = resp.choices[0].message.content or ""
            usage = resp.usage
            prompt_toks = usage.prompt_tokens if usage else 0
            completion_toks = usage.completion_tokens if usage else 0
            return answer, prompt_toks, completion_toks

        if self.llm_provider == "anthropic":
            resp = client.messages.create(
                model=self.llm_model,
                max_tokens=1024,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = resp.content[0].text if resp.content else ""
            return answer, resp.usage.input_tokens, resp.usage.output_tokens

        raise ValueError(f"Unknown llm_provider: {self.llm_provider!r}")
