"""Compare two ragharness result CSVs side by side.

Reads two ``results_summary.csv`` files, matches configs by parameter
equality, and computes per-metric deltas with directional indicators.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click

_LOWER_IS_BETTER = {"latency_p50", "latency_p95", "token_cost"}


# ── Data structures ─────────────────────────────────────


@dataclass
class MetricDelta:
    metric: str
    value_a: float | None
    value_b: float | None
    absolute_delta: float | None
    pct_change: float | None
    direction: str  # "improved" | "regressed" | "unchanged" | "n/a"


@dataclass
class ConfigComparison:
    config_label: str
    config_params: dict[str, str]
    deltas: list[MetricDelta] = field(default_factory=list)


@dataclass
class CompareResult:
    path_a: str
    path_b: str
    comparisons: list[ConfigComparison] = field(default_factory=list)
    unmatched_a: list[dict[str, str]] = field(default_factory=list)
    unmatched_b: list[dict[str, str]] = field(default_factory=list)


# ── CSV reading ─────────────────────────────────────────


def read_summary_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a ``results_summary.csv`` into a list of row dicts."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_row(
    row: dict[str, str],
) -> tuple[dict[str, str], dict[str, float]]:
    """Split a CSV row into (config_params, metric_scores).

    Values that parse as ``float`` go to metrics; everything else is a
    config parameter. Mirrors the heuristic in ``ragharness.cli.report``.
    """
    config: dict[str, str] = {}
    metrics: dict[str, float] = {}
    for key, value in row.items():
        try:
            metrics[key] = float(value)
        except (ValueError, TypeError):
            config[key] = str(value)
    return config, metrics


# ── Matching ────────────────────────────────────────────


def _config_key(config: dict[str, str]) -> frozenset[tuple[str, str]]:
    return frozenset(config.items())


def _match_configs(
    rows_a: list[dict[str, str]],
    rows_b: list[dict[str, str]],
) -> tuple[
    list[tuple[dict[str, str], dict[str, float], dict[str, float]]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    """Match configs between two CSVs by config_params equality.

    Returns ``(matched, unmatched_from_a, unmatched_from_b)`` where each
    matched entry is ``(config_params, metrics_a, metrics_b)``.
    """
    parsed_a = [_parse_row(r) for r in rows_a]
    parsed_b = [_parse_row(r) for r in rows_b]

    index_b: dict[frozenset[tuple[str, str]], tuple[dict[str, str], dict[str, float]]] = {}
    for config, metrics in parsed_b:
        index_b[_config_key(config)] = (config, metrics)

    matched: list[tuple[dict[str, str], dict[str, float], dict[str, float]]] = []
    unmatched_a: list[dict[str, str]] = []
    seen_b_keys: set[frozenset[tuple[str, str]]] = set()

    for config_a, metrics_a in parsed_a:
        key = _config_key(config_a)
        if key in index_b:
            _, metrics_b = index_b[key]
            matched.append((config_a, metrics_a, metrics_b))
            seen_b_keys.add(key)
        else:
            unmatched_a.append(config_a)

    unmatched_b = [config for config, _ in parsed_b if _config_key(config) not in seen_b_keys]
    return matched, unmatched_a, unmatched_b


# ── Delta computation ───────────────────────────────────


def _compute_delta(
    metric: str,
    val_a: float | None,
    val_b: float | None,
    threshold: float,
) -> MetricDelta:
    if val_a is None or val_b is None:
        return MetricDelta(
            metric=metric,
            value_a=val_a,
            value_b=val_b,
            absolute_delta=None,
            pct_change=None,
            direction="n/a",
        )

    delta = val_b - val_a
    pct = (delta / val_a * 100) if val_a != 0 else None

    if abs(delta) < threshold:
        direction = "unchanged"
    elif metric in _LOWER_IS_BETTER:
        direction = "improved" if delta < 0 else "regressed"
    else:
        direction = "improved" if delta > 0 else "regressed"

    return MetricDelta(
        metric=metric,
        value_a=val_a,
        value_b=val_b,
        absolute_delta=delta,
        pct_change=pct,
        direction=direction,
    )


# ── Main entry point ───────────────────────────────────


def compare_results(
    path_a: str | Path,
    path_b: str | Path,
    threshold: float = 0.05,
) -> CompareResult:
    """Read two ``results_summary.csv`` files and produce a comparison."""
    rows_a = read_summary_csv(path_a)
    rows_b = read_summary_csv(path_b)
    matched, unmatched_a, unmatched_b = _match_configs(rows_a, rows_b)

    comparisons: list[ConfigComparison] = []
    for config_params, metrics_a, metrics_b in matched:
        label = ", ".join(f"{k}={v}" for k, v in sorted(config_params.items())) or "baseline"
        all_metrics = sorted(set(metrics_a) | set(metrics_b))
        deltas = [
            _compute_delta(m, metrics_a.get(m), metrics_b.get(m), threshold) for m in all_metrics
        ]
        comparisons.append(
            ConfigComparison(
                config_label=label,
                config_params=config_params,
                deltas=deltas,
            )
        )

    return CompareResult(
        path_a=str(path_a),
        path_b=str(path_b),
        comparisons=comparisons,
        unmatched_a=unmatched_a,
        unmatched_b=unmatched_b,
    )


# ── Terminal formatting ─────────────────────────────────


def _direction_symbol(d: MetricDelta) -> str:
    if d.direction == "improved":
        return click.style("  improved", fg="green")
    if d.direction == "regressed":
        return click.style("  regressed", fg="red")
    if d.direction == "unchanged":
        return click.style("  unchanged", dim=True)
    return click.style("  n/a", dim=True)


def format_comparison_table(result: CompareResult) -> str:
    """Format a :class:`CompareResult` as an ANSI-coloured terminal table."""
    lines: list[str] = []
    lines.append(f"Comparing: A = {result.path_a}")
    lines.append(f"           B = {result.path_b}")
    lines.append("")

    for comp in result.comparisons:
        lines.append(f"[{comp.config_label}]")
        lines.append(
            f"  {'Metric':<30s} {'A':>12s} {'B':>12s} {'Delta':>12s} {'%':>8s}  Direction"
        )
        lines.append(f"  {'-' * 90}")
        for d in comp.deltas:
            va = f"{d.value_a:.4f}" if d.value_a is not None else "-"
            vb = f"{d.value_b:.4f}" if d.value_b is not None else "-"
            delta = f"{d.absolute_delta:+.4f}" if d.absolute_delta is not None else "-"
            pct = f"{d.pct_change:+.1f}%" if d.pct_change is not None else "-"
            sym = _direction_symbol(d)
            lines.append(f"  {d.metric:<30s} {va:>12s} {vb:>12s} {delta:>12s} {pct:>8s}{sym}")
        lines.append("")

    if result.unmatched_a:
        lines.append("Configs only in A:")
        for cfg in result.unmatched_a:
            label = ", ".join(f"{k}={v}" for k, v in sorted(cfg.items())) or "baseline"
            lines.append(f"  {label}")
        lines.append("")

    if result.unmatched_b:
        lines.append("Configs only in B:")
        for cfg in result.unmatched_b:
            label = ", ".join(f"{k}={v}" for k, v in sorted(cfg.items())) or "baseline"
            lines.append(f"  {label}")
        lines.append("")

    return "\n".join(lines)


# ── CSV output ──────────────────────────────────────────


def write_comparison_csv(
    result: CompareResult,
    output_path: str | Path,
) -> Path:
    """Write comparison deltas to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_keys: list[str] = []
    if result.comparisons:
        config_keys = sorted(result.comparisons[0].config_params.keys())

    fieldnames = config_keys + [
        "metric",
        "value_a",
        "value_b",
        "delta",
        "pct_change",
        "direction",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for comp in result.comparisons:
            for d in comp.deltas:
                row: dict[str, Any] = {**comp.config_params}
                row["metric"] = d.metric
                row["value_a"] = f"{d.value_a:.4f}" if d.value_a is not None else ""
                row["value_b"] = f"{d.value_b:.4f}" if d.value_b is not None else ""
                row["delta"] = f"{d.absolute_delta:+.4f}" if d.absolute_delta is not None else ""
                row["pct_change"] = f"{d.pct_change:+.1f}%" if d.pct_change is not None else ""
                row["direction"] = d.direction
                writer.writerow(row)

    return path
