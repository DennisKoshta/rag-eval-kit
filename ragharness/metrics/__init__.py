from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ragharness.dataset import EvalItem
from ragharness.metrics.answer import contains, f1_token, rouge_l
from ragharness.metrics.cost import token_cost
from ragharness.metrics.exact_match import exact_match
from ragharness.metrics.latency import latency_p50, latency_p95
from ragharness.metrics.retrieval import (
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from ragharness.protocol import RAGResult

PerQuestionMetric = Callable[[EvalItem, RAGResult], float]
AggregateMetric = Callable[..., float]

PER_QUESTION_REGISTRY: dict[str, PerQuestionMetric] = {
    "exact_match": exact_match,
    "contains": contains,
    "f1_token": f1_token,
    "rouge_l": rouge_l,
    "precision_at_k": precision_at_k,
    "recall_at_k": recall_at_k,
    "hit_rate_at_k": hit_rate_at_k,
    "mrr": mrr,
    "ndcg_at_k": ndcg_at_k,
    # llm_judge / llm_faithfulness are registered lazily — they need config to instantiate
}

AGGREGATE_REGISTRY: dict[str, AggregateMetric] = {
    "latency_p50": latency_p50,
    "latency_p95": latency_p95,
    "token_cost": token_cost,
}


def get_per_question_metric(name: str, **kwargs: Any) -> PerQuestionMetric:
    """Look up a per-question metric by name.

    Per-question metrics have the signature ``(EvalItem, RAGResult) -> float``
    and are called once per dataset item per sweep configuration.

    The names ``llm_judge`` and ``llm_faithfulness`` are handled lazily
    because they instantiate an LLM client — *kwargs* are forwarded to
    :class:`ragharness.metrics.llm_judge.LLMJudge` or
    :class:`ragharness.metrics.llm_judge.LLMFaithfulness` respectively.

    Raises ``ValueError`` if the name is unknown. Register new metrics by
    mutating ``PER_QUESTION_REGISTRY`` before calling ``run_sweep``.
    """
    if name == "llm_judge":
        from ragharness.metrics.llm_judge import LLMJudge

        return LLMJudge(**kwargs)
    if name == "llm_faithfulness":
        from ragharness.metrics.llm_judge import LLMFaithfulness

        return LLMFaithfulness(**kwargs)

    if name not in PER_QUESTION_REGISTRY:
        raise ValueError(f"Unknown per-question metric: {name!r}")
    return PER_QUESTION_REGISTRY[name]


def get_aggregate_metric(name: str) -> AggregateMetric:
    """Look up an aggregate metric by name.

    Aggregate metrics have the signature ``(list[RAGResult], **kwargs) -> float``
    and are called once per sweep configuration after all per-question
    scoring is complete. Register new metrics by mutating
    ``AGGREGATE_REGISTRY`` before calling ``run_sweep``.

    Raises ``ValueError`` if the name is unknown.
    """
    if name not in AGGREGATE_REGISTRY:
        raise ValueError(f"Unknown aggregate metric: {name!r}")
    return AGGREGATE_REGISTRY[name]
