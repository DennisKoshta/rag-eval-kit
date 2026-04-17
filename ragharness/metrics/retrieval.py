"""Retrieval-quality metrics.

All metrics in this module compare ``result.retrieved_docs`` against
``item.expected_docs`` by exact string equality. If your retriever
returns chunk IDs, store chunk IDs in ``expected_docs``; if it returns
raw text, store raw text. Mixing the two will score 0 across the board.
"""

from __future__ import annotations

import math

from ragharness.dataset import EvalItem
from ragharness.protocol import RAGResult


def _matches(expected: list[str], retrieved: list[str], k: int) -> list[bool]:
    """Per-position hit vector for the first *k* retrieved docs.

    Returns a list of length ``min(k, len(retrieved))`` where the i-th
    entry is True iff ``retrieved[i]`` is in the *expected* set.
    """
    expected_set = set(expected)
    return [doc in expected_set for doc in retrieved[:k]]


def precision_at_k(item: EvalItem, result: RAGResult, *, k: int = 5) -> float:
    """Fraction of the top-k retrieved documents that appear in ``item.expected_docs``.

    Edge cases:

    - returns 0.0 when ``item.expected_docs`` is empty (no ground truth)
    - returns 0.0 when ``result.retrieved_docs`` is empty
    - k is capped by the retrieved-docs length — we never pad with zeros

    The denominator is ``len(top_k)``, not ``k``, so under-retrieval is
    not penalised. Use :func:`recall_at_k` for that.
    """
    if not item.expected_docs:
        return 0.0
    hits = _matches(item.expected_docs, result.retrieved_docs, k)
    if not hits:
        return 0.0
    return sum(hits) / len(hits)


def recall_at_k(item: EvalItem, result: RAGResult, *, k: int = 5) -> float:
    """Fraction of expected docs that appear anywhere in the top-k retrieved.

    Counts *unique* expected docs that were hit, so duplicates in
    ``retrieved_docs`` do not inflate recall. Returns 0.0 when
    ``expected_docs`` is empty.
    """
    if not item.expected_docs:
        return 0.0
    expected_set = set(item.expected_docs)
    retrieved_set = set(result.retrieved_docs[:k])
    return len(expected_set & retrieved_set) / len(expected_set)


def hit_rate_at_k(item: EvalItem, result: RAGResult, *, k: int = 5) -> float:
    """1.0 if any expected doc appears in the top-k, else 0.0.

    The binary "at least something useful made it" signal — very sensitive
    on small datasets where percentages don't smooth out. Returns 0.0 when
    ``expected_docs`` is empty.
    """
    if not item.expected_docs:
        return 0.0
    return 1.0 if any(_matches(item.expected_docs, result.retrieved_docs, k)) else 0.0


def mrr(item: EvalItem, result: RAGResult) -> float:
    """Reciprocal rank of the first retrieved doc that hits ``expected_docs``.

    Scans the full ranked list (no *k* cap — MRR is conventionally
    computed over all retrieved items). Returns 0.0 if nothing hits or
    either side is empty.
    """
    if not item.expected_docs or not result.retrieved_docs:
        return 0.0
    expected_set = set(item.expected_docs)
    for i, doc in enumerate(result.retrieved_docs, start=1):
        if doc in expected_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(item: EvalItem, result: RAGResult, *, k: int = 5) -> float:
    """Binary-relevance nDCG at *k*.

    Uses the standard formula ``DCG = Σ rel_i / log2(i + 1)`` for
    positions 1..k, and ``IDCG`` computed against the ideal ranking of
    the actually-expected doc count capped at *k*. Returns 0.0 when
    either side is empty.
    """
    if not item.expected_docs or not result.retrieved_docs:
        return 0.0
    hits = _matches(item.expected_docs, result.retrieved_docs, k)
    dcg = sum(1.0 / math.log2(i + 2) for i, hit in enumerate(hits) if hit)
    ideal_hits = min(len(set(item.expected_docs)), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
