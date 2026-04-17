"""Tests for the retrieval-pack metrics: recall / hit_rate / mrr / ndcg."""

from __future__ import annotations

import math

import pytest

from ragharness.dataset import EvalItem
from ragharness.metrics.retrieval import (
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from ragharness.protocol import RAGResult


def _item(expected: list[str] | None) -> EvalItem:
    return EvalItem(question="q", expected_answer="a", expected_docs=expected)


def _result(docs: list[str]) -> RAGResult:
    return RAGResult(answer="a", retrieved_docs=docs)


# ── recall_at_k ─────────────────────────────────────────


def test_recall_perfect():
    assert recall_at_k(_item(["a", "b"]), _result(["a", "b", "c"]), k=3) == 1.0


def test_recall_partial():
    assert recall_at_k(_item(["a", "b", "c"]), _result(["a", "b"]), k=5) == pytest.approx(2 / 3)


def test_recall_none_hit():
    assert recall_at_k(_item(["x"]), _result(["a", "b"]), k=5) == 0.0


def test_recall_k_caps_retrieved():
    # Only "a" appears in first 2 retrieved → recall = 1/2
    assert recall_at_k(_item(["a", "b"]), _result(["a", "x", "b"]), k=2) == 0.5


def test_recall_no_expected_is_zero():
    assert recall_at_k(_item(None), _result(["a"]), k=5) == 0.0


def test_recall_empty_retrieved_is_zero():
    assert recall_at_k(_item(["a"]), _result([]), k=5) == 0.0


def test_recall_k_larger_than_retrieved_does_not_error():
    assert recall_at_k(_item(["a"]), _result(["a"]), k=100) == 1.0


# ── hit_rate_at_k ───────────────────────────────────────


def test_hit_rate_any_hit_is_one():
    assert hit_rate_at_k(_item(["a", "b"]), _result(["x", "a"]), k=5) == 1.0


def test_hit_rate_no_hit_is_zero():
    assert hit_rate_at_k(_item(["a"]), _result(["b", "c"]), k=5) == 0.0


def test_hit_rate_k_caps_window():
    # Only "x" in first 2; "a" at index 3 is out of window
    assert hit_rate_at_k(_item(["a"]), _result(["x", "y", "a"]), k=2) == 0.0


def test_hit_rate_no_expected_is_zero():
    assert hit_rate_at_k(_item(None), _result(["a"]), k=5) == 0.0


def test_hit_rate_empty_retrieved_is_zero():
    assert hit_rate_at_k(_item(["a"]), _result([]), k=5) == 0.0


# ── mrr ─────────────────────────────────────────────────


def test_mrr_first_position_is_one():
    assert mrr(_item(["a"]), _result(["a", "b", "c"])) == 1.0


def test_mrr_third_position_is_one_third():
    assert mrr(_item(["c"]), _result(["a", "b", "c"])) == pytest.approx(1 / 3)


def test_mrr_takes_first_hit_only():
    # "b" at pos 2 hits first; "a" at pos 3 is ignored
    assert mrr(_item(["a", "b"]), _result(["x", "b", "a"])) == 0.5


def test_mrr_no_hit_is_zero():
    assert mrr(_item(["z"]), _result(["a", "b", "c"])) == 0.0


def test_mrr_no_expected_is_zero():
    assert mrr(_item(None), _result(["a"])) == 0.0


def test_mrr_empty_retrieved_is_zero():
    assert mrr(_item(["a"]), _result([])) == 0.0


# ── ndcg_at_k ───────────────────────────────────────────


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(_item(["a", "b"]), _result(["a", "b"]), k=2) == pytest.approx(1.0)


def test_ndcg_all_hits_out_of_order_still_one():
    # Both expected docs in top-k, just reordered — IDCG uses ideal
    # positions but with all expected in-window, hits fill the prefix too.
    assert ndcg_at_k(_item(["a", "b"]), _result(["b", "a"]), k=2) == pytest.approx(1.0)


def test_ndcg_no_hits_is_zero():
    assert ndcg_at_k(_item(["x"]), _result(["a", "b", "c"]), k=3) == 0.0


def test_ndcg_ranking_matters():
    # Expected: {"a"}; retrieved: ["x", "a"] at k=2
    # DCG = 1 / log2(3) ≈ 0.6309
    # IDCG (one expected doc, ideal pos 1) = 1 / log2(2) = 1
    # nDCG ≈ 0.6309
    score = ndcg_at_k(_item(["a"]), _result(["x", "a"]), k=2)
    assert score == pytest.approx(1 / math.log2(3))


def test_ndcg_ideal_capped_at_k():
    # 5 expected docs but k=2 → IDCG is over the top 2 ideal positions
    # retrieved ["a", "b"] both hit → DCG = 1 + 1/log2(3)
    # IDCG(k=2, 5 expected) = same = 1 + 1/log2(3)
    # ratio = 1.0
    score = ndcg_at_k(
        _item(["a", "b", "c", "d", "e"]),
        _result(["a", "b"]),
        k=2,
    )
    assert score == pytest.approx(1.0)


def test_ndcg_no_expected_is_zero():
    assert ndcg_at_k(_item(None), _result(["a"]), k=5) == 0.0


def test_ndcg_empty_retrieved_is_zero():
    assert ndcg_at_k(_item(["a"]), _result([]), k=5) == 0.0
