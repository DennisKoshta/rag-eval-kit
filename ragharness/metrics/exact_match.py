"""Exact-match answer scoring."""

from __future__ import annotations

from ragharness.dataset import EvalItem
from ragharness.protocol import RAGResult


def exact_match(item: EvalItem, result: RAGResult) -> float:
    """Binary 1.0/0.0 score comparing ``result.answer`` to ``item.expected_answer``.

    Normalization applied to both sides before comparison:

    - surrounding whitespace is stripped
    - comparison is case-insensitive

    This metric is intentionally strict — substring containment, partial
    matches, and paraphrases all score 0.0. Use ``llm_judge`` for softer
    semantic matching.

    Returns 1.0 on match, 0.0 otherwise.
    """
    return 1.0 if result.answer.strip().lower() == item.expected_answer.strip().lower() else 0.0
