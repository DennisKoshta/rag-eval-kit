"""Tests for tag-based metric grouping."""

from __future__ import annotations

import pytest

from ragharness.dataset import EvalItem
from ragharness.tag_grouping import compute_tag_scores


def _item(tags: dict | None = None) -> EvalItem:
    return EvalItem(question="q", expected_answer="a", tags=tags)


# ── Basic grouping ──────────────────────────────────────


def test_single_tag_key_two_values():
    items = [
        _item({"topic": "physics"}),
        _item({"topic": "physics"}),
        _item({"topic": "history"}),
    ]
    scores = [
        {"exact_match": 1.0},
        {"exact_match": 0.0},
        {"exact_match": 1.0},
    ]
    result = compute_tag_scores(items, scores)
    assert result["topic"]["physics"]["exact_match"] == pytest.approx(0.5)
    assert result["topic"]["history"]["exact_match"] == pytest.approx(1.0)


def test_multiple_tag_keys():
    items = [
        _item({"topic": "physics", "difficulty": "hard"}),
        _item({"topic": "history", "difficulty": "easy"}),
    ]
    scores = [{"f1": 0.8}, {"f1": 0.4}]
    result = compute_tag_scores(items, scores)
    assert "topic" in result
    assert "difficulty" in result
    assert result["difficulty"]["hard"]["f1"] == pytest.approx(0.8)
    assert result["difficulty"]["easy"]["f1"] == pytest.approx(0.4)


def test_multiple_metrics():
    items = [_item({"t": "a"}), _item({"t": "a"})]
    scores = [
        {"exact_match": 1.0, "f1": 0.9},
        {"exact_match": 0.0, "f1": 0.7},
    ]
    result = compute_tag_scores(items, scores)
    assert result["t"]["a"]["exact_match"] == pytest.approx(0.5)
    assert result["t"]["a"]["f1"] == pytest.approx(0.8)


# ── Edge cases ──────────────────────────────────────────


def test_no_tags_returns_empty():
    items = [_item(None), _item(None)]
    scores = [{"exact_match": 1.0}, {"exact_match": 0.0}]
    assert compute_tag_scores(items, scores) == {}


def test_partial_tags_only_groups_tagged_items():
    items = [
        _item({"topic": "physics"}),
        _item(None),
        _item({"topic": "physics"}),
    ]
    scores = [
        {"exact_match": 1.0},
        {"exact_match": 0.0},
        {"exact_match": 0.5},
    ]
    result = compute_tag_scores(items, scores)
    # Only items 0 and 2 contribute
    assert result["topic"]["physics"]["exact_match"] == pytest.approx(0.75)


def test_single_item_per_group():
    items = [_item({"k": "v"})]
    scores = [{"m": 0.42}]
    result = compute_tag_scores(items, scores)
    assert result["k"]["v"]["m"] == pytest.approx(0.42)


def test_empty_inputs():
    assert compute_tag_scores([], []) == {}


def test_tag_values_converted_to_str():
    items = [_item({"level": 3})]
    scores = [{"m": 1.0}]
    result = compute_tag_scores(items, scores)
    assert "3" in result["level"]


def test_results_sorted_by_key():
    items = [
        _item({"z": "1", "a": "1"}),
    ]
    scores = [{"m": 1.0}]
    result = compute_tag_scores(items, scores)
    assert list(result.keys()) == ["a", "z"]
