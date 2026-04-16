from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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

    _chart_accuracy(sweep_result, config_labels, output_dir)
    _chart_latency_distribution(sweep_result, config_labels, output_dir)
    _chart_cost_vs_accuracy(sweep_result, config_labels, output_dir)
    _chart_per_metric_bars(sweep_result, config_labels, output_dir)

    return output_dir


def _config_labels(sweep_result: SweepResult) -> list[str]:
    labels = []
    for run in sweep_result.runs:
        label = ", ".join(f"{k}={v}" for k, v in sorted(run.config_params.items())) or "baseline"
        labels.append(label)
    return labels


def _chart_accuracy(sweep_result: SweepResult, labels: list[str], output_dir: Path) -> None:
    """Grouped bar chart of mean per-question metric scores."""
    # Gather all mean_* keys
    mean_keys = sorted(k for k in sweep_result.runs[0].aggregate_scores if k.startswith("mean_"))
    if not mean_keys:
        return

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
    fig.savefig(output_dir / "chart_accuracy.png", dpi=150)
    plt.close(fig)


def _chart_latency_distribution(
    sweep_result: SweepResult, labels: list[str], output_dir: Path
) -> None:
    """Box plot of per-query latency across configurations."""
    latency_data = [
        [r.metadata.get("latency_ms", 0.0) for r in run.raw_results] for run in sweep_result.runs
    ]
    if not any(latency_data):
        return

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    ax.boxplot(latency_data, tick_labels=labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Distribution")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "chart_latency_distribution.png", dpi=150)
    plt.close(fig)


def _chart_cost_vs_accuracy(
    sweep_result: SweepResult, labels: list[str], output_dir: Path
) -> None:
    """Scatter plot: total cost (x) vs mean accuracy (y)."""
    costs = [run.aggregate_scores.get("token_cost", 0.0) for run in sweep_result.runs]
    # Use mean_llm_judge if available, fall back to mean_exact_match
    accuracy_key = "mean_llm_judge"
    if accuracy_key not in sweep_result.runs[0].aggregate_scores:
        accuracy_key = "mean_exact_match"
    if accuracy_key not in sweep_result.runs[0].aggregate_scores:
        return

    accuracies = [run.aggregate_scores.get(accuracy_key, 0.0) for run in sweep_result.runs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(costs, accuracies)
    for i, label in enumerate(labels):
        ax.annotate(label, (costs[i], accuracies[i]), fontsize=7)
    ax.set_xlabel("Cost ($)")
    ax.set_ylabel(accuracy_key.removeprefix("mean_"))
    ax.set_title("Cost vs Accuracy Tradeoff")
    fig.tight_layout()
    fig.savefig(output_dir / "chart_cost_vs_accuracy.png", dpi=150)
    plt.close(fig)


def _chart_per_metric_bars(sweep_result: SweepResult, labels: list[str], output_dir: Path) -> None:
    """One bar chart per aggregate metric (excluding mean_ and token_cost)."""
    skip = {"token_cost"}
    agg_keys = sorted(
        k
        for k in sweep_result.runs[0].aggregate_scores
        if k not in skip and not k.startswith("mean_")
    )

    for metric_name in agg_keys:
        values = [run.aggregate_scores.get(metric_name, 0.0) for run in sweep_result.runs]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
        ax.bar(range(len(values)), values)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} across configurations")
        fig.tight_layout()
        fig.savefig(output_dir / f"chart_{metric_name}.png", dpi=150)
        plt.close(fig)
