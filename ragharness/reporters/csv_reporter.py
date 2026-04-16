from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragharness.orchestrator import SweepResult


def write_csv(sweep_result: SweepResult, output_dir: str | Path) -> Path:
    """Write per-question detail and aggregate summary CSVs.

    Returns the output directory path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sweep_result.runs:
        return output_dir

    _write_detail(sweep_result, output_dir / "results_detail.csv")
    _write_summary(sweep_result, output_dir / "results_summary.csv")

    return output_dir


def _write_detail(sweep_result: SweepResult, path: Path) -> None:
    first_run = sweep_result.runs[0]
    config_keys = sorted(first_run.config_params.keys()) if first_run.config_params else []
    metric_keys = (
        sorted(first_run.per_question_scores[0].keys())
        if first_run.per_question_scores
        else []
    )

    fieldnames = (
        config_keys
        + ["question", "expected_answer", "answer"]
        + metric_keys
        + ["latency_ms"]
    )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in sweep_result.runs:
            for item, result, scores in zip(
                run.items, run.raw_results, run.per_question_scores
            ):
                row: dict[str, object] = {**run.config_params}
                row["question"] = item.question
                row["expected_answer"] = item.expected_answer
                row["answer"] = result.answer
                row["latency_ms"] = f"{result.metadata.get('latency_ms', 0.0):.2f}"
                row.update(scores)
                writer.writerow(row)


def _write_summary(sweep_result: SweepResult, path: Path) -> None:
    first_run = sweep_result.runs[0]
    config_keys = sorted(first_run.config_params.keys()) if first_run.config_params else []
    agg_keys = sorted(first_run.aggregate_scores.keys())
    fieldnames = config_keys + agg_keys

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in sweep_result.runs:
            row: dict[str, object] = {**run.config_params}
            for k, v in run.aggregate_scores.items():
                row[k] = f"{v:.4f}"
            writer.writerow(row)
