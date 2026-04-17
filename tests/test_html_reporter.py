"""Tests for the HTML reporter."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from ragharness.dataset import EvalItem
from ragharness.orchestrator import RunResult, SweepResult
from ragharness.protocol import RAGResult
from ragharness.reporters.compare_reporter import CompareResult, ConfigComparison, MetricDelta
from ragharness.reporters.html_reporter import (
    _fig_to_base64,
    _render_detail_table,
    _render_summary_table,
    _render_tag_tables,
    write_comparison_html,
    write_html,
)


def _make_sweep_result() -> SweepResult:
    """Build a small SweepResult with 2 configs x 2 questions, items have tags."""
    items = [
        EvalItem(question="Q1", expected_answer="A1", tags={"topic": "physics"}),
        EvalItem(question="Q2", expected_answer="A2", tags={"topic": "history"}),
    ]

    def _run(params: dict, answers: list[str]) -> RunResult:
        results = [
            RAGResult(
                answer=ans,
                retrieved_docs=["doc1"],
                metadata={
                    "latency_ms": 100.0 + i * 50,
                    "model": "m",
                    "prompt_tokens": 50,
                    "completion_tokens": 20,
                },
            )
            for i, ans in enumerate(answers)
        ]
        per_q = [
            {"exact_match": 1.0 if ans == exp.expected_answer else 0.0}
            for ans, exp in zip(answers, items)
        ]
        agg = {
            "latency_p50": 100.0,
            "latency_p95": 150.0,
            "mean_exact_match": sum(s["exact_match"] for s in per_q) / len(per_q),
        }
        return RunResult(
            config_params=params,
            per_question_scores=per_q,
            aggregate_scores=agg,
            raw_results=results,
            items=items,
        )

    return SweepResult(
        runs=[
            _run({"top_k": 3}, ["A1", "A2"]),
            _run({"top_k": 5}, ["A1", "wrong"]),
        ]
    )


# ── _fig_to_base64 ─────────────────────────────────────


def test_fig_to_base64_returns_data_uri():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    uri = _fig_to_base64(fig)
    assert uri.startswith("data:image/png;base64,")
    assert len(uri) > 100


# ── _render_summary_table ──────────────────────────────


def test_render_summary_table_contains_metrics():
    result = _make_sweep_result()
    html = _render_summary_table(result)
    assert "mean_exact_match" in html
    assert "latency_p50" in html


def test_render_summary_table_contains_config_params():
    result = _make_sweep_result()
    html = _render_summary_table(result)
    assert "top_k" in html


def test_render_summary_table_empty():
    html = _render_summary_table(SweepResult())
    assert "No results" in html


# ── _render_detail_table ───────────────────────────────


def test_render_detail_table_row_count():
    result = _make_sweep_result()
    html = _render_detail_table(result)
    # 2 configs x 2 questions = 4 <tr> in tbody
    assert html.count("<tr>") == 4 + 1  # +1 for header


def test_render_detail_table_empty_scores():
    sr = SweepResult(
        runs=[
            RunResult(
                config_params={},
                per_question_scores=[],
                aggregate_scores={},
                raw_results=[],
                items=[],
            )
        ]
    )
    html = _render_detail_table(sr)
    assert "No per-question data" in html


# ── _render_tag_tables ──────────────────────────────────


def test_render_tag_tables_structure():
    tag_scores = {
        "topic": {
            "physics": {"exact_match": 0.6},
            "history": {"exact_match": 0.9},
        }
    }
    html = _render_tag_tables(tag_scores)
    assert "topic" in html
    assert "physics" in html
    assert "history" in html
    assert "0.6000" in html


def test_render_tag_tables_empty():
    assert _render_tag_tables({}) == ""


# ── write_html ──────────────────────────────────────────


def test_write_html_creates_file(tmp_path):
    result = _make_sweep_result()
    path = write_html(result, tmp_path / "report.html")
    assert path.exists()
    assert path.stat().st_size > 0


def test_write_html_self_contained(tmp_path):
    result = _make_sweep_result()
    path = write_html(result, tmp_path / "report.html")
    content = path.read_text(encoding="utf-8")
    assert "<link " not in content
    assert 'src="http' not in content
    assert "<style>" in content
    assert "<script>" in content


def test_write_html_contains_base64_images(tmp_path):
    result = _make_sweep_result()
    path = write_html(result, tmp_path / "report.html")
    content = path.read_text(encoding="utf-8")
    assert "data:image/png;base64," in content


def test_write_html_with_tag_scores(tmp_path):
    result = _make_sweep_result()
    tags = {"topic": {"physics": {"exact_match": 0.7}, "history": {"exact_match": 0.9}}}
    path = write_html(result, tmp_path / "report.html", tag_scores=tags)
    content = path.read_text(encoding="utf-8")
    assert "Tag Breakdown" in content
    assert "physics" in content


def test_write_html_empty_sweep(tmp_path):
    path = write_html(SweepResult(), tmp_path / "empty.html")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "No results" in content


# ── write_comparison_html ──────────────────────────────


def test_write_comparison_html_creates_file(tmp_path):
    cmp = CompareResult(
        path_a="a.csv",
        path_b="b.csv",
        comparisons=[
            ConfigComparison(
                config_label="baseline",
                config_params={},
                deltas=[
                    MetricDelta("mean_exact_match", 0.8, 0.9, 0.1, 12.5, "improved"),
                ],
            )
        ],
    )
    path = write_comparison_html(cmp, tmp_path / "cmp.html")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Comparison Report" in content
    assert "improved" in content
    assert "mean_exact_match" in content
