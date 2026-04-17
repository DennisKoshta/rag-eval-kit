"""Self-contained HTML report generator.

All CSS and JS are inlined — no external dependencies. Charts are
embedded as base64-encoded PNGs via matplotlib's in-memory buffer.
"""

from __future__ import annotations

import base64
import html
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

matplotlib.use("Agg")

if TYPE_CHECKING:
    from ragharness.orchestrator import SweepResult
    from ragharness.reporters.compare_reporter import CompareResult


# ── Inline assets ───────────────────────────────────────

_CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, sans-serif; padding: 2rem; color: #1a1a1a;
       max-width: 1200px; margin: 0 auto; line-height: 1.5; }
h1 { margin-bottom: .5rem; }
h2 { margin: 2rem 0 .75rem; border-bottom: 2px solid #e0e0e0; padding-bottom: .25rem; }
h3 { margin: 1.25rem 0 .5rem; }
table { border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; font-size: .875rem; }
th, td { padding: .4rem .6rem; text-align: left; border: 1px solid #d0d0d0; }
th { background: #f5f5f5; position: sticky; top: 0; cursor: pointer; user-select: none; }
th:hover { background: #e8e8e8; }
tr:nth-child(even) { background: #fafafa; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.improved { color: #16a34a; font-weight: 600; }
.regressed { color: #dc2626; font-weight: 600; }
.unchanged { color: #9ca3af; }
.chart-grid { display: flex; flex-wrap: wrap; gap: 1.5rem; }
.chart-grid img { max-width: 600px; width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; }
details { margin-bottom: 1rem; }
summary { cursor: pointer; font-weight: 600; padding: .25rem 0; }
input#filter { padding: .4rem .6rem; width: 300px; margin-bottom: .75rem;
               border: 1px solid #d0d0d0; border-radius: 4px; font-size: .875rem; }
.meta { color: #6b7280; font-size: .85rem; margin-bottom: 1.5rem; }
"""

_JS = """\
document.querySelectorAll('table.sortable th').forEach(th => {
  th.addEventListener('click', () => {
    const table = th.closest('table');
    const idx = Array.from(th.parentNode.children).indexOf(th);
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const dir = th.dataset.dir === 'asc' ? 'desc' : 'asc';
    th.parentNode.querySelectorAll('th').forEach(h => delete h.dataset.dir);
    th.dataset.dir = dir;
    rows.sort((a, b) => {
      const va = a.children[idx].textContent;
      const vb = b.children[idx].textContent;
      const na = parseFloat(va), nb = parseFloat(vb);
      if (!isNaN(na) && !isNaN(nb)) return dir === 'asc' ? na - nb : nb - na;
      return dir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
    });
    rows.forEach(r => tbody.appendChild(r));
  });
});
const filterInput = document.getElementById('filter');
if (filterInput) {
  filterInput.addEventListener('input', () => {
    const q = filterInput.value.toLowerCase();
    document.querySelectorAll('#detail-table tbody tr').forEach(r => {
      r.style.display = r.textContent.toLowerCase().includes(q) ? '' : 'none';
    });
  });
}
"""


# ── Helpers ─────────────────────────────────────────────


def _esc(text: Any) -> str:
    return html.escape(str(text))


def _fig_to_base64(fig: Figure) -> str:
    """Render a matplotlib figure to a base64 PNG data URI."""
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _html_page(*, title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "<meta charset='utf-8'>\n"
        f"<title>{_esc(title)}</title>\n"
        f"<style>\n{_CSS}</style>\n"
        "</head>\n<body>\n"
        f"{body}\n"
        f"<script>\n{_JS}</script>\n"
        "</body>\n</html>"
    )


# ── Sweep report sections ──────────────────────────────


def _render_summary_table(sweep_result: SweepResult) -> str:
    if not sweep_result.runs:
        return "<p>No results.</p>"

    first = sweep_result.runs[0]
    config_keys = sorted(first.config_params.keys()) if first.config_params else []
    metric_keys = sorted(first.aggregate_scores.keys())

    header = "".join(f"<th>{_esc(k)}</th>" for k in config_keys + metric_keys)
    rows: list[str] = []
    for run in sweep_result.runs:
        cells = "".join(f"<td>{_esc(run.config_params.get(k, ''))}</td>" for k in config_keys)
        cells += "".join(
            f"<td class='num'>{run.aggregate_scores.get(k, 0.0):.4f}</td>" for k in metric_keys
        )
        rows.append(f"<tr>{cells}</tr>")

    return (
        "<table class='sortable'><thead><tr>"
        f"{header}</tr></thead><tbody>\n" + "\n".join(rows) + "\n</tbody></table>"
    )


def _render_detail_table(sweep_result: SweepResult) -> str:
    if not sweep_result.runs or not sweep_result.runs[0].per_question_scores:
        return "<p>No per-question data available.</p>"

    first = sweep_result.runs[0]
    config_keys = sorted(first.config_params.keys()) if first.config_params else []
    metric_keys = sorted(first.per_question_scores[0].keys()) if first.per_question_scores else []
    columns = config_keys + ["question", "expected_answer", "answer"] + metric_keys

    header = "".join(f"<th>{_esc(c)}</th>" for c in columns)
    rows: list[str] = []
    for run in sweep_result.runs:
        for item, result, scores in zip(run.items, run.raw_results, run.per_question_scores):
            cells = "".join(f"<td>{_esc(run.config_params.get(k, ''))}</td>" for k in config_keys)
            cells += f"<td>{_esc(item.question)}</td>"
            cells += f"<td>{_esc(item.expected_answer)}</td>"
            cells += f"<td>{_esc(result.answer)}</td>"
            cells += "".join(f"<td class='num'>{scores.get(k, 0.0):.4f}</td>" for k in metric_keys)
            rows.append(f"<tr>{cells}</tr>")

    return (
        "<table class='sortable' id='detail-table'><thead><tr>"
        f"{header}</tr></thead><tbody>\n" + "\n".join(rows) + "\n</tbody></table>"
    )


def _render_tag_tables(
    tag_scores: dict[str, dict[str, dict[str, float]]],
) -> str:
    if not tag_scores:
        return ""

    sections: list[str] = []
    for tag_key, values in sorted(tag_scores.items()):
        metric_keys = sorted({m for scores in values.values() for m in scores})
        header = f"<th>{_esc(tag_key)}</th>" + "".join(f"<th>{_esc(m)}</th>" for m in metric_keys)
        rows: list[str] = []
        for tag_val in sorted(values):
            cells = f"<td>{_esc(tag_val)}</td>"
            cells += "".join(
                f"<td class='num'>{values[tag_val].get(m, 0.0):.4f}</td>" for m in metric_keys
            )
            rows.append(f"<tr>{cells}</tr>")

        table = (
            f"<table class='sortable'><thead><tr>{header}</tr></thead>"
            f"<tbody>\n" + "\n".join(rows) + "\n</tbody></table>"
        )
        sections.append(f"<details open><summary>{_esc(tag_key)}</summary>\n{table}\n</details>")

    return "\n".join(sections)


def _generate_charts_base64(sweep_result: SweepResult) -> list[tuple[str, str]]:
    """Generate charts as (title, base64_data_uri) pairs."""
    from ragharness.reporters.chart_reporter import (
        _config_labels,
        _make_accuracy_fig,
        _make_cost_vs_accuracy_fig,
        _make_latency_fig,
        _make_per_metric_figs,
    )

    labels = _config_labels(sweep_result)
    charts: list[tuple[str, str]] = []

    for make_fn, title in [
        (_make_accuracy_fig, "Accuracy vs Configuration"),
        (_make_latency_fig, "Latency Distribution"),
        (_make_cost_vs_accuracy_fig, "Cost vs Accuracy"),
    ]:
        fig = make_fn(sweep_result, labels)
        if fig is not None:
            charts.append((title, _fig_to_base64(fig)))

    for fig, filename in _make_per_metric_figs(sweep_result, labels):
        title = filename.replace("chart_", "").replace(".png", "").replace("_", " ").title()
        charts.append((title, _fig_to_base64(fig)))

    return charts


# ── Public API: sweep report ───────────────────────────


def write_html(
    sweep_result: SweepResult,
    output_path: str | Path,
    tag_scores: dict[str, dict[str, dict[str, float]]] | None = None,
) -> Path:
    """Generate a self-contained HTML report from sweep results."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts: list[str] = [
        "<h1>ragharness Evaluation Report</h1>",
        f"<p class='meta'>Generated: {timestamp}</p>",
        "<h2>Summary</h2>",
        _render_summary_table(sweep_result),
    ]

    if tag_scores:
        parts.append("<h2>Tag Breakdown</h2>")
        parts.append(_render_tag_tables(tag_scores))

    if sweep_result.runs and sweep_result.runs[0].raw_results:
        charts = _generate_charts_base64(sweep_result)
        if charts:
            parts.append("<h2>Charts</h2>")
            parts.append("<div class='chart-grid'>")
            for title, data_uri in charts:
                parts.append(f"<img src='{data_uri}' alt='{_esc(title)}' title='{_esc(title)}'>")
            parts.append("</div>")

    parts.append("<h2>Per-Question Detail</h2>")
    parts.append("<input type='text' id='filter' placeholder='Filter questions...'>")
    parts.append(_render_detail_table(sweep_result))

    content = _html_page(title="ragharness Report", body="\n".join(parts))
    path.write_text(content, encoding="utf-8")
    return path


# ── Public API: comparison report ──────────────────────


def write_comparison_html(
    compare_result: CompareResult,
    output_path: str | Path,
) -> Path:
    """Generate an HTML report for a comparison between two result CSVs."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts: list[str] = [
        "<h1>ragharness Comparison Report</h1>",
        f"<p class='meta'>Generated: {timestamp}</p>",
        f"<p><strong>A:</strong> {_esc(compare_result.path_a)}<br>"
        f"<strong>B:</strong> {_esc(compare_result.path_b)}</p>",
    ]

    for comp in compare_result.comparisons:
        parts.append(f"<h2>{_esc(comp.config_label)}</h2>")

        header = "<th>Metric</th><th>A</th><th>B</th><th>Delta</th><th>%</th><th>Direction</th>"
        rows: list[str] = []
        for d in comp.deltas:
            va = f"{d.value_a:.4f}" if d.value_a is not None else "-"
            vb = f"{d.value_b:.4f}" if d.value_b is not None else "-"
            delta = f"{d.absolute_delta:+.4f}" if d.absolute_delta is not None else "-"
            pct = f"{d.pct_change:+.1f}%" if d.pct_change is not None else "-"
            css = d.direction if d.direction in ("improved", "regressed", "unchanged") else ""
            rows.append(
                f"<tr><td>{_esc(d.metric)}</td>"
                f"<td class='num'>{va}</td><td class='num'>{vb}</td>"
                f"<td class='num'>{delta}</td><td class='num'>{pct}</td>"
                f"<td class='{css}'>{_esc(d.direction)}</td></tr>"
            )

        parts.append(
            f"<table class='sortable'><thead><tr>{header}</tr></thead>"
            f"<tbody>\n" + "\n".join(rows) + "\n</tbody></table>"
        )

    if compare_result.unmatched_a:
        parts.append("<h2>Configs only in A</h2><ul>")
        for cfg in compare_result.unmatched_a:
            label = ", ".join(f"{k}={v}" for k, v in sorted(cfg.items())) or "baseline"
            parts.append(f"<li>{_esc(label)}</li>")
        parts.append("</ul>")

    if compare_result.unmatched_b:
        parts.append("<h2>Configs only in B</h2><ul>")
        for cfg in compare_result.unmatched_b:
            label = ", ".join(f"{k}={v}" for k, v in sorted(cfg.items())) or "baseline"
            parts.append(f"<li>{_esc(label)}</li>")
        parts.append("</ul>")

    content = _html_page(title="ragharness Comparison", body="\n".join(parts))
    path.write_text(content, encoding="utf-8")
    return path
