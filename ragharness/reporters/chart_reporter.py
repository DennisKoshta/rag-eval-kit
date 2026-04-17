from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

matplotlib.use("Agg")

if TYPE_CHECKING:
    from ragharness.orchestrator import SweepResult


def write_charts(sweep_result: SweepResult, output_dir: str | Path) -> Path:
    """Generate comparison charts from sweep results.

    Returns the output directory path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sweep_result.runs:
        return output_dir

    config_labels = _config_labels(sweep_result)

    for make_fn, name in [
        (_make_accuracy_fig, "chart_accuracy.png"),
        (_make_latency_fig, "chart_latency_distribution.png"),
        (_make_cost_vs_accuracy_fig, "chart_cost_vs_accuracy.png"),
    ]:
        fig = make_fn(sweep_result, config_labels)
        if fig is not None:
            fig.savefig(output_dir / name, dpi=150)
            plt.close(fig)

    for fig, name in _make_per_metric_figs(sweep_result, config_labels):
        fig.savefig(output_dir / name, dpi=150)
        plt.close(fig)

    return output_dir


def _config_labels(sweep_result: SweepResult) -> list[str]:
    labels = []
    for run in sweep_result.runs:
        label = ", ".join(f"{k}={v}" for k, v in sorted(run.config_params.items())) or "baseline"
        labels.append(label)
    return labels


def _make_accuracy_fig(sweep_result: SweepResult, labels: list[str]) -> Figure | None:
    """Create accuracy bar chart figure. Returns None if no data."""
    mean_keys = sorted(k for k in sweep_result.runs[0].aggregate_scores if k.startswith("mean_"))
    if not mean_keys:
        return None

    x = np.arange(len(labels))
    width = 0.8 / max(len(mean_keys), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    for i, key in enumerate(mean_keys):
        values = [run.aggregate_scores.get(key, 0.0) for run in sweep_result.runs]
        offset = (i - len(mean_keys) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=key.removeprefix("mean_"))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Accuracy vs Configuration")
    ax.legend()
    fig.tight_layout()
    return fig


def _make_latency_fig(sweep_result: SweepResult, labels: list[str]) -> Figure | None:
    """Create latency box plot figure. Returns None if no data."""
    latency_data = [
        [r.metadata.get("latency_ms", 0.0) for r in run.raw_results] for run in sweep_result.runs
    ]
    if not any(latency_data):
        return None

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    ax.boxplot(latency_data, tick_labels=labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Distribution")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


def _make_cost_vs_accuracy_fig(sweep_result: SweepResult, labels: list[str]) -> Figure | None:
    """Create cost-vs-accuracy scatter figure. Returns None if no data."""
    costs = [run.aggregate_scores.get("token_cost", 0.0) for run in sweep_result.runs]
    accuracy_key = "mean_llm_judge"
    if accuracy_key not in sweep_result.runs[0].aggregate_scores:
        accuracy_key = "mean_exact_match"
    if accuracy_key not in sweep_result.runs[0].aggregate_scores:
        return None

    accuracies = [run.aggregate_scores.get(accuracy_key, 0.0) for run in sweep_result.runs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(costs, accuracies)
    for i, label in enumerate(labels):
        ax.annotate(label, (costs[i], accuracies[i]), fontsize=7)
    ax.set_xlabel("Cost ($)")
    ax.set_ylabel(accuracy_key.removeprefix("mean_"))
    ax.set_title("Cost vs Accuracy Tradeoff")
    fig.tight_layout()
    return fig


def _make_per_metric_figs(
    sweep_result: SweepResult, labels: list[str]
) -> list[tuple[Figure, str]]:
    """Create one bar chart per aggregate metric. Returns (fig, filename) pairs."""
    skip = {"token_cost"}
    agg_keys = sorted(
        k
        for k in sweep_result.runs[0].aggregate_scores
        if k not in skip and not k.startswith("mean_")
    )
    figs: list[tuple[Figure, str]] = []
    for metric_name in agg_keys:
        values = [run.aggregate_scores.get(metric_name, 0.0) for run in sweep_result.runs]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
        ax.bar(range(len(values)), values)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} across configurations")
        fig.tight_layout()
        figs.append((fig, f"chart_{metric_name}.png"))
    return figs
