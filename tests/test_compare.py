"""Tests for the compare reporter."""

from __future__ import annotations

import csv

import pytest

from ragharness.reporters.compare_reporter import (
    CompareResult,
    ConfigComparison,
    MetricDelta,
    _compute_delta,
    _match_configs,
    _parse_row,
    compare_results,
    format_comparison_table,
    write_comparison_csv,
)

# ── _parse_row ──────────────────────────────────────────


def test_parse_row_splits_config_and_metrics():
    row = {"adapter": "raw", "mean_exact_match": "0.8500", "latency_p50": "120.0"}
    config, metrics = _parse_row(row)
    assert config == {"adapter": "raw"}
    assert metrics == {"mean_exact_match": 0.85, "latency_p50": 120.0}


def test_parse_row_all_numeric():
    row = {"latency_p50": "100.0", "latency_p95": "200.0"}
    config, metrics = _parse_row(row)
    assert config == {}
    assert len(metrics) == 2


def test_parse_row_no_numeric():
    row = {"adapter": "raw", "model": "gpt-4o"}
    config, metrics = _parse_row(row)
    assert config == {"adapter": "raw", "model": "gpt-4o"}
    assert metrics == {}


# ── _match_configs ──────────────────────────────────────


def test_match_configs_exact_match():
    rows_a = [{"adapter": "raw", "mean_exact_match": "0.8"}]
    rows_b = [{"adapter": "raw", "mean_exact_match": "0.9"}]
    matched, ua, ub = _match_configs(rows_a, rows_b)
    assert len(matched) == 1
    assert ua == []
    assert ub == []
    _, metrics_a, metrics_b = matched[0]
    assert metrics_a["mean_exact_match"] == 0.8
    assert metrics_b["mean_exact_match"] == 0.9


def test_match_configs_with_unmatched():
    rows_a = [
        {"adapter": "raw", "score": "0.8"},
        {"adapter": "langchain", "score": "0.7"},
    ]
    rows_b = [
        {"adapter": "raw", "score": "0.85"},
        {"adapter": "llamaindex", "score": "0.9"},
    ]
    matched, ua, ub = _match_configs(rows_a, rows_b)
    assert len(matched) == 1
    assert len(ua) == 1  # langchain only in A
    assert len(ub) == 1  # llamaindex only in B


def test_match_configs_empty_inputs():
    matched, ua, ub = _match_configs([], [])
    assert matched == []
    assert ua == []
    assert ub == []


# ── _compute_delta ──────────────────────────────────────


def test_compute_delta_improvement_higher_is_better():
    d = _compute_delta("mean_exact_match", 0.8, 0.9, 0.05)
    assert d.direction == "improved"
    assert d.absolute_delta == pytest.approx(0.1)
    assert d.pct_change == pytest.approx(12.5)


def test_compute_delta_regression_higher_is_better():
    d = _compute_delta("mean_exact_match", 0.9, 0.7, 0.05)
    assert d.direction == "regressed"


def test_compute_delta_improvement_lower_is_better():
    # latency decrease = improvement
    d = _compute_delta("latency_p50", 200.0, 150.0, 0.05)
    assert d.direction == "improved"
    assert d.absolute_delta == pytest.approx(-50.0)


def test_compute_delta_regression_lower_is_better():
    d = _compute_delta("latency_p50", 100.0, 200.0, 0.05)
    assert d.direction == "regressed"


def test_compute_delta_unchanged_within_threshold():
    d = _compute_delta("mean_exact_match", 0.80, 0.82, 0.05)
    assert d.direction == "unchanged"


def test_compute_delta_zero_baseline_no_pct():
    d = _compute_delta("token_cost", 0.0, 0.5, 0.05)
    assert d.pct_change is None
    assert d.direction == "regressed"  # cost increased, lower is better


def test_compute_delta_missing_value_a():
    d = _compute_delta("new_metric", None, 0.5, 0.05)
    assert d.direction == "n/a"
    assert d.absolute_delta is None


def test_compute_delta_missing_value_b():
    d = _compute_delta("old_metric", 0.5, None, 0.05)
    assert d.direction == "n/a"


# ── compare_results end-to-end ──────────────────────────


def _write_csv(path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_compare_results_end_to_end(tmp_path):
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    _write_csv(
        csv_a,
        [{"adapter": "raw", "mean_exact_match": "0.8000", "latency_p50": "100.0000"}],
    )
    _write_csv(
        csv_b,
        [{"adapter": "raw", "mean_exact_match": "0.9000", "latency_p50": "80.0000"}],
    )

    result = compare_results(csv_a, csv_b, threshold=0.05)
    assert len(result.comparisons) == 1
    comp = result.comparisons[0]
    assert comp.config_label == "adapter=raw"

    deltas_by_name = {d.metric: d for d in comp.deltas}
    assert deltas_by_name["mean_exact_match"].direction == "improved"
    assert deltas_by_name["latency_p50"].direction == "improved"


# ── format_comparison_table ─────────────────────────────


def test_format_comparison_table_contains_labels():
    result = CompareResult(
        path_a="a.csv",
        path_b="b.csv",
        comparisons=[
            ConfigComparison(
                config_label="baseline",
                config_params={},
                deltas=[
                    MetricDelta("m", 0.5, 0.6, 0.1, 20.0, "improved"),
                ],
            )
        ],
    )
    text = format_comparison_table(result)
    assert "a.csv" in text
    assert "b.csv" in text
    assert "baseline" in text


# ── write_comparison_csv ────────────────────────────────


def test_write_comparison_csv(tmp_path):
    result = CompareResult(
        path_a="a.csv",
        path_b="b.csv",
        comparisons=[
            ConfigComparison(
                config_label="top_k=3",
                config_params={"top_k": "3"},
                deltas=[
                    MetricDelta("mean_exact_match", 0.8, 0.9, 0.1, 12.5, "improved"),
                    MetricDelta("latency_p50", 100.0, 80.0, -20.0, -20.0, "improved"),
                ],
            )
        ],
    )
    out = write_comparison_csv(result, tmp_path / "cmp.csv")
    assert out.exists()

    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["metric"] == "mean_exact_match"
    assert rows[0]["direction"] == "improved"
    assert rows[0]["top_k"] == "3"
