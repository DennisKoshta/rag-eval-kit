"""Tests for the answer-quality metric pack (contains / f1_token / rouge_l)."""

from __future__ import annotations

import pytest

from ragharness.dataset import EvalItem
from ragharness.metrics.answer import contains, f1_token, rouge_l
from ragharness.protocol import RAGResult


def _pair(expected: str, answer: str) -> tuple[EvalItem, RAGResult]:
    return EvalItem(question="q", expected_answer=expected), RAGResult(answer=answer)


# ── contains ────────────────────────────────────────────


def test_contains_exact_match():
    item, result = _pair("Paris", "Paris")
    assert contains(item, result) == 1.0


def test_contains_substring_match():
    item, result = _pair("Paris", "The capital is Paris, France.")
    assert contains(item, result) == 1.0


def test_contains_case_insensitive():
    item, result = _pair("paris", "PARIS IS THE ANSWER")
    assert contains(item, result) == 1.0


def test_contains_whitespace_stripped():
    item, result = _pair("  Paris  ", "\tParis\n")
    assert contains(item, result) == 1.0


def test_contains_no_match():
    item, result = _pair("Paris", "London is the capital.")
    assert contains(item, result) == 0.0


def test_contains_empty_expected_returns_zero():
    item, result = _pair("", "Paris")
    assert contains(item, result) == 0.0


def test_contains_empty_answer_with_nonempty_expected():
    item, result = _pair("Paris", "")
    assert contains(item, result) == 0.0


# ── f1_token ────────────────────────────────────────────


def test_f1_identical_answers_is_one():
    item, result = _pair("the quick brown fox", "the quick brown fox")
    assert f1_token(item, result) == pytest.approx(1.0)


def test_f1_disjoint_tokens_is_zero():
    item, result = _pair("cat dog bird", "shark whale octopus")
    assert f1_token(item, result) == 0.0


def test_f1_partial_overlap_matches_squad_formula():
    # expected tokens: {the, quick, brown, fox}  (4)
    # pred tokens:     {the, brown, fox, jumped} (4)
    # overlap: {the, brown, fox} = 3
    # P = 3/4, R = 3/4 → F = 3/4 = 0.75
    item, result = _pair("the quick brown fox", "the brown fox jumped")
    assert f1_token(item, result) == pytest.approx(0.75)


def test_f1_punctuation_stripped():
    item, result = _pair("hello, world!", "hello world")
    assert f1_token(item, result) == pytest.approx(1.0)


def test_f1_case_insensitive():
    item, result = _pair("Hello World", "hello world")
    assert f1_token(item, result) == pytest.approx(1.0)


def test_f1_multiset_intersection_no_over_credit():
    # pred has "the" 3 times, ref only 1 time — overlap should be 1
    # P = 1/3, R = 1/1 = 1.0 → F = 2·(1/3)·1/(1/3 + 1) = (2/3)/(4/3) = 0.5
    item, result = _pair("the", "the the the")
    assert f1_token(item, result) == pytest.approx(0.5)


def test_f1_empty_expected_is_zero():
    item, result = _pair("", "anything")
    assert f1_token(item, result) == 0.0


def test_f1_empty_answer_is_zero():
    item, result = _pair("something", "")
    assert f1_token(item, result) == 0.0


# ── rouge_l ─────────────────────────────────────────────


def test_rouge_l_identical_is_one():
    item, result = _pair("the quick brown fox", "the quick brown fox")
    assert rouge_l(item, result) == pytest.approx(1.0)


def test_rouge_l_disjoint_is_zero():
    item, result = _pair("cat dog", "shark whale")
    assert rouge_l(item, result) == 0.0


def test_rouge_l_known_lcs_value():
    # ref: a b c d e  (5 tokens)
    # pred: a x b y c z d (7 tokens)
    # LCS = [a, b, c, d] → length 4
    # P = 4/7, R = 4/5 → F = 2·(4/7)·(4/5)/(4/7 + 4/5)
    #   = (32/35) / (20/35 + 28/35) = (32/35)/(48/35) = 32/48 = 2/3
    item, result = _pair("a b c d e", "a x b y c z d")
    assert rouge_l(item, result) == pytest.approx(2 / 3)


def test_rouge_l_respects_token_order():
    # Same tokens reversed — LCS is just 1 token (since all 4 are distinct)
    # ref: a b c d, pred: d c b a → LCS = 1
    # P = 1/4, R = 1/4 → F = 1/4
    item, result = _pair("a b c d", "d c b a")
    assert rouge_l(item, result) == pytest.approx(0.25)


def test_rouge_l_empty_expected_is_zero():
    item, result = _pair("", "anything")
    assert rouge_l(item, result) == 0.0


def test_rouge_l_empty_answer_is_zero():
    item, result = _pair("something", "")
    assert rouge_l(item, result) == 0.0
