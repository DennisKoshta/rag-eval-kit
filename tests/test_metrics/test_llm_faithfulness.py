"""Tests for the LLM faithfulness judge — mirror of test_llm_judge.py."""

from __future__ import annotations

import math

import pytest

from ragharness.dataset import EvalItem
from ragharness.metrics.llm_judge import LLMFaithfulness
from ragharness.protocol import RAGResult


def _item() -> EvalItem:
    return EvalItem(question="Capital of France?", expected_answer="Paris")


def _result(answer: str = "Paris", docs: list[str] | None = None) -> RAGResult:
    return RAGResult(answer=answer, retrieved_docs=docs or ["Paris is the capital of France."])


def test_parses_valid_json_score(monkeypatch):
    judge = LLMFaithfulness(provider="openai")
    monkeypatch.setattr(
        judge, "_call_llm", lambda prompt: '{"score": 0.9, "reasoning": "grounded"}'
    )
    assert judge(_item(), _result()) == 0.9


def test_clamps_score_above_one(monkeypatch):
    judge = LLMFaithfulness(provider="openai")
    monkeypatch.setattr(judge, "_call_llm", lambda prompt: '{"score": 1.5, "reasoning": "x"}')
    assert judge(_item(), _result()) == 1.0


def test_clamps_score_below_zero(monkeypatch):
    judge = LLMFaithfulness(provider="openai")
    monkeypatch.setattr(judge, "_call_llm", lambda prompt: '{"score": -0.5, "reasoning": "x"}')
    assert judge(_item(), _result()) == 0.0


def test_retries_on_bad_json(monkeypatch):
    call_count = 0

    def _fake_llm(prompt):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return "not json"
        return '{"score": 0.6, "reasoning": "ok"}'

    judge = LLMFaithfulness(provider="openai", max_retries=2)
    monkeypatch.setattr(judge, "_call_llm", _fake_llm)

    assert judge(_item(), _result()) == 0.6
    assert call_count == 3


def test_returns_nan_after_exhausted_retries(monkeypatch):
    judge = LLMFaithfulness(provider="openai", max_retries=1)
    monkeypatch.setattr(judge, "_call_llm", lambda prompt: "garbage")
    assert math.isnan(judge(_item(), _result()))


def test_unknown_provider_raises():
    judge = LLMFaithfulness(provider="fake_provider")
    with pytest.raises(ValueError, match="Unknown LLM judge provider"):
        judge._get_client()


def test_prompt_contains_retrieved_docs_not_expected_answer(monkeypatch):
    """Faithfulness judges grounding — expected_answer must not be in the prompt."""
    seen: dict[str, str] = {}

    def _capture(prompt: str) -> str:
        seen["prompt"] = prompt
        return '{"score": 1.0, "reasoning": "x"}'

    judge = LLMFaithfulness(provider="openai")
    monkeypatch.setattr(judge, "_call_llm", _capture)

    item = EvalItem(question="q", expected_answer="SECRET_EXPECTED")
    result = RAGResult(answer="some answer", retrieved_docs=["doc one.", "doc two."])
    judge(item, result)

    assert "doc one." in seen["prompt"]
    assert "doc two." in seen["prompt"]
    assert "some answer" in seen["prompt"]
    assert "SECRET_EXPECTED" not in seen["prompt"]


def test_no_retrieved_docs_still_scores(monkeypatch):
    """With empty retrieved_docs the prompt still renders (with a placeholder)."""
    judge = LLMFaithfulness(provider="openai")
    monkeypatch.setattr(judge, "_call_llm", lambda prompt: '{"score": 0.0, "reasoning": "none"}')
    item = EvalItem(question="q", expected_answer="a")
    result = RAGResult(answer="some answer", retrieved_docs=[])
    assert judge(item, result) == 0.0
