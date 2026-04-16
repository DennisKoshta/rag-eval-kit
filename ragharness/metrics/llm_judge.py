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


class LLMJudge:
    """LLM-based answer quality judge.

    Callable that scores a RAG answer against an expected answer using
    an LLM.  Conforms to the per-question metric signature
    ``(EvalItem, RAGResult) -> float``.
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

    def __call__(self, item: EvalItem, result: RAGResult) -> float:
        prompt = JUDGE_SYSTEM_PROMPT.format(
            question=item.question,
            expected_answer=item.expected_answer,
            system_answer=result.answer,
        )

        for attempt in range(1, self.max_retries + 2):  # +2 = 1 initial + max_retries
            try:
                raw = self._call_llm(prompt)
                parsed = json.loads(raw)
                score = float(parsed["score"])
                return max(0.0, min(1.0, score))
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning(
                    "LLM judge parse failure (attempt %d/%d): %s",
                    attempt,
                    self.max_retries + 1,
                    e,
                )

        logger.error("LLM judge failed after %d attempts, returning NaN", self.max_retries + 1)
        return float("nan")

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
