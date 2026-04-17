from __future__ import annotations

import json
import logging
from typing import Any

from ragharness.dataset import EvalItem
from ragharness.protocol import RAGResult

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """\
You are evaluating a RAG system's answer to a question.

Question: {question}
Expected Answer: {expected_answer}
System Answer: {system_answer}

Rate the system answer's correctness on a scale from 0.0 to 1.0:
- 1.0: Fully correct, captures all key information
- 0.7-0.9: Mostly correct, minor omissions or imprecisions
- 0.3-0.6: Partially correct, significant gaps
- 0.0-0.2: Incorrect or irrelevant

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<brief explanation>"}}"""

FAITHFULNESS_SYSTEM_PROMPT = """\
You are evaluating whether a RAG system's answer is faithful to its retrieved context.

Retrieved Context:
{context}

System Answer: {system_answer}

Rate how much of the answer is directly supported by the retrieved context on a
scale from 0.0 to 1.0:
- 1.0: Every claim in the answer is grounded in the context
- 0.7-0.9: Mostly grounded, at most minor unsupported details
- 0.3-0.6: Some claims grounded, others fabricated or unsupported
- 0.0-0.2: Answer contradicts the context or is entirely fabricated

Do not penalise correctness or completeness — only grounding. A short answer
that is fully supported scores 1.0.

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<brief explanation>"}}"""


class _LLMScorer:
    """Shared LLM-client machinery for judge-style metrics.

    Subclasses define a ``__call__(item, result) -> float`` that builds a
    prompt and feeds it through :meth:`_call_llm` and :meth:`_parse_score`.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 2,
    ) -> None:
        self.provider = provider
        self.model = model
        self.max_retries = max_retries
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package required for LLM judge with provider='openai'. "
                    "Install with: pip install ragharness[openai]"
                ) from None
            self._client = openai.OpenAI()
        elif self.provider == "anthropic":
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required for LLM judge with provider='anthropic'. "
                    "Install with: pip install ragharness[anthropic]"
                ) from None
            self._client = anthropic.Anthropic()
        else:
            raise ValueError(f"Unknown LLM judge provider: {self.provider!r}")

        return self._client

    def _call_llm(self, prompt: str) -> str:
        client = self._get_client()

        if self.provider == "openai":
            resp = client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""

        # anthropic
        resp = client.messages.create(
            model=self.model,
            max_tokens=256,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return str(resp.content[0].text)

    def _parse_score(self, prompt: str) -> float:
        """Call the LLM, parse the JSON ``score`` field, clamp to [0, 1].

        Retries up to ``self.max_retries`` times on JSON / type / value
        errors. Returns NaN if every attempt fails.
        """
        for attempt in range(1, self.max_retries + 2):  # +2 = 1 initial + max_retries
            try:
                raw = self._call_llm(prompt)
                parsed = json.loads(raw)
                score = float(parsed["score"])
                return max(0.0, min(1.0, score))
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning(
                    "LLM scorer parse failure (attempt %d/%d): %s",
                    attempt,
                    self.max_retries + 1,
                    e,
                )

        logger.error("LLM scorer failed after %d attempts, returning NaN", self.max_retries + 1)
        return float("nan")


class LLMJudge(_LLMScorer):
    """LLM-based answer correctness judge.

    Callable that scores a RAG answer against the expected answer. Conforms
    to the per-question metric signature ``(EvalItem, RAGResult) -> float``.
    """

    def __call__(self, item: EvalItem, result: RAGResult) -> float:
        prompt = JUDGE_SYSTEM_PROMPT.format(
            question=item.question,
            expected_answer=item.expected_answer,
            system_answer=result.answer,
        )
        return self._parse_score(prompt)


class LLMFaithfulness(_LLMScorer):
    """LLM-based faithfulness / groundedness judge.

    Scores how well the system answer is supported by ``result.retrieved_docs``
    — a hallucination detector rather than a correctness check. The
    expected answer is *not* used; faithfulness asks "is this grounded?",
    not "is this right?".
    """

    def __call__(self, item: EvalItem, result: RAGResult) -> float:
        context = "\n\n".join(result.retrieved_docs) if result.retrieved_docs else "(no context)"
        prompt = FAITHFULNESS_SYSTEM_PROMPT.format(
            context=context,
            system_answer=result.answer,
        )
        return self._parse_score(prompt)
