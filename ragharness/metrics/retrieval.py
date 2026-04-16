"""Retrieval-quality metrics."""

from __future__ import annotations

from ragharness.dataset import EvalItem
from ragharness.protocol import RAGResult


def precision_at_k(item: EvalItem, result: RAGResult, *, k: int = 5) -> float:
    """Fraction of the top-k retrieved documents that appear in ``item.expected_docs``.

    Docs are compared by exact string equality (whitespace-sensitive). If
    your retriever returns chunk IDs, store chunk IDs in ``expected_docs``;
    if it returns raw text, store raw text. Mixing the two will score 0.

    Edge cases:

    - returns 0.0 when ``item.expected_docs`` is empty (no ground truth)
    - returns 0.0 when ``result.retrieved_docs`` is empty
    - k is capped by the retrieved-docs length — we never pad with zeros

    The denominator is ``len(top_k)``, not ``k``, so under-retrieval is
    not penalised. Use ``recall_at_k`` (coming soon) for that.
    """
    if not item.expected_docs:
        return 0.0
    top_k = result.retrieved_docs[:k]
    if not top_k:
        return 0.0
    expected_set = set(item.expected_docs)
    hits = sum(1 for doc in top_k if doc in expected_set)
    return hits / len(top_k)
