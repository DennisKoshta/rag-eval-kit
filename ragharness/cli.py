from __future__ import annotations

import csv as csv_module
import logging
import sys
from pathlib import Path

import click

from ragharness._version import __version__


@click.group()
@click.version_option(version=__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def main(verbose: bool) -> None:
    """ragharness — pluggable RAG evaluation framework."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


# ── run ──────────────────────────────────────────────────


@main.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Print run plan without executing.")
@click.option("--output-dir", type=click.Path(), default=None, help="Override output directory.")
@click.option("--filter", "filter_str", default=None, help='Filter configs (e.g. "top_k=5").')
@click.option("--no-confirm", is_flag=True, help="Skip cost confirmation prompt.")
@click.option(
    "--verbose",
    "verbose_queries",
    is_flag=True,
    help="Show per-question results as they complete.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Integer seed for reproducibility, injected into adapter_config. "
    "Honoured by OpenAI; silently ignored by providers without seed support.",
)
@click.option(
    "--concurrency",
    type=int,
    default=None,
    help="Number of parallel queries per config (overrides YAML). Requires thread-safe adapter.",
)
@click.option(
    "--checkpoint",
    "checkpoint_path",
    type=click.Path(),
    default=None,
    help="JSONL checkpoint path for resumable runs (overrides YAML output.checkpoint).",
)
def run(
    config: str,
    dry_run: bool,
    output_dir: str | None,
    no_confirm: bool,
    filter_str: str | None,
    verbose_queries: bool,
    seed: int | None,
    concurrency: int | None,
    checkpoint_path: str | None,
) -> None:
    """Run an evaluation sweep from a config file."""
    from ragharness.auth import MissingAPIKeyError, load_dotenv
    from ragharness.config import load_config
    from ragharness.orchestrator import run_sweep
    from ragharness.reporters import write_charts, write_csv

    load_dotenv()
    cfg = load_config(config)

    if output_dir:
        cfg.output.csv = str(Path(output_dir) / "results.csv")
        cfg.output.charts = str(Path(output_dir) / "charts")

    # Optional sweep filter: --filter "top_k=5"
    if filter_str:
        key, _, value = filter_str.partition("=")
        if key in cfg.sweep:
            cfg.sweep[key] = [v for v in cfg.sweep[key] if str(v) == value]

    if seed is not None:
        cfg.system.adapter_config["seed"] = seed

    if concurrency is not None:
        if concurrency < 1:
            click.echo("Error: --concurrency must be >= 1", err=True)
            raise SystemExit(1)
        cfg.concurrency = concurrency

    if checkpoint_path is not None:
        cfg.output.checkpoint = checkpoint_path

    try:
        result = run_sweep(
            cfg,
            dry_run=dry_run,
            no_confirm=no_confirm,
            verbose=verbose_queries,
        )
    except MissingAPIKeyError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from None

    if dry_run or not result.runs:
        return

    # Write outputs
    out_dir = Path(output_dir or "results")
    if cfg.output.csv:
        csv_dir = write_csv(result, out_dir)
        click.echo(f"\nCSV results written to {csv_dir}")
    if cfg.output.charts:
        chart_dir = write_charts(result, out_dir / "charts")
        click.echo(f"Charts written to {chart_dir}")

    if cfg.output.html:
        from ragharness.reporters.html_reporter import write_html

        tag_scores: dict[str, dict[str, dict[str, float]]] | None = None
        if any(run_result.tag_scores for run_result in result.runs):
            merged: dict[str, dict[str, dict[str, float]]] = {}
            for run_result in result.runs:
                for tag_key, tag_vals in run_result.tag_scores.items():
                    merged.setdefault(tag_key, {}).update(tag_vals)
            tag_scores = merged
        html_path = write_html(result, cfg.output.html, tag_scores=tag_scores)
        click.echo(f"HTML report written to {html_path}")

    # Print summary table
    _print_summary(result)


def _config_label(run_result: object) -> str:
    from ragharness.orchestrator import RunResult

    assert isinstance(run_result, RunResult)
    return ", ".join(f"{k}={v}" for k, v in sorted(run_result.config_params.items())) or "baseline"


def _print_summary(result: object) -> None:
    from ragharness.orchestrator import SweepResult

    assert isinstance(result, SweepResult)

    click.echo(f"\n{'=' * 60}")
    click.echo("Summary")
    click.echo(f"{'=' * 60}")
    for run_result in result.runs:
        label = _config_label(run_result)
        click.echo(f"\n  [{label}]")
        for metric, value in sorted(run_result.aggregate_scores.items()):
            click.echo(f"    {metric}: {value:.4f}")


# ── validate ─────────────────────────────────────────────


@main.command()
@click.argument("config", type=click.Path(exists=True))
def validate(config: str) -> None:
    """Validate a config file without running."""
    from ragharness.config import load_config
    from ragharness.orchestrator import expand_sweep

    try:
        cfg = load_config(config)
    except Exception as e:
        click.echo(f"Config validation failed: {e}", err=True)
        raise SystemExit(1) from None

    n_sweep = len(expand_sweep(cfg.sweep))

    click.echo("Config is valid.")
    click.echo(f"  Adapter:         {cfg.system.adapter}")
    click.echo(f"  Dataset:         {cfg.dataset.path or cfg.dataset.name}")
    click.echo(f"  Sweep configs:   {n_sweep}")
    click.echo(f"  Metrics:         {cfg.metrics}")


# ── report ───────────────────────────────────────────────


@main.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Output directory for charts.",
)
@click.option(
    "--html",
    "html_path",
    type=click.Path(),
    default=None,
    help="Generate a self-contained HTML report.",
)
def report(csv_path: str, output_dir: str | None, html_path: str | None) -> None:
    """Re-generate charts from an existing results_summary.csv."""
    from ragharness.orchestrator import RunResult, SweepResult
    from ragharness.reporters import write_charts

    csv_file = Path(csv_path)
    out_dir = Path(output_dir) if output_dir else csv_file.parent

    runs: list[RunResult] = []
    with csv_file.open(newline="") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            config_params: dict[str, str] = {}
            aggregate_scores: dict[str, float] = {}
            for key, value in row.items():
                try:
                    aggregate_scores[key] = float(value)
                except ValueError:
                    config_params[key] = value
            runs.append(
                RunResult(
                    config_params=config_params,
                    per_question_scores=[],
                    aggregate_scores=aggregate_scores,
                    raw_results=[],
                    items=[],
                )
            )

    sweep_result = SweepResult(runs=runs)
    chart_dir = write_charts(sweep_result, out_dir)
    click.echo(f"Charts regenerated in {chart_dir}")

    if html_path:
        from ragharness.reporters.html_reporter import write_html

        hp = write_html(sweep_result, html_path)
        click.echo(f"HTML report written to {hp}")


# ── compare ──────────────────────────────────────────────


@main.command()
@click.argument("csv_a", type=click.Path(exists=True))
@click.argument("csv_b", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Write comparison CSV.",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=0.05,
    help="Min absolute delta to flag as changed (default 0.05).",
)
@click.option(
    "--html",
    "html_path",
    type=click.Path(),
    default=None,
    help="Write an HTML comparison report.",
)
def compare(
    csv_a: str,
    csv_b: str,
    output: str | None,
    threshold: float,
    html_path: str | None,
) -> None:
    """Compare two results_summary.csv files side by side."""
    from ragharness.reporters.compare_reporter import (
        compare_results,
        format_comparison_table,
        write_comparison_csv,
    )

    result = compare_results(csv_a, csv_b, threshold=threshold)
    click.echo(format_comparison_table(result))

    if output:
        write_comparison_csv(result, output)
        click.echo(f"Comparison CSV written to {output}")

    if html_path:
        from ragharness.reporters.html_reporter import write_comparison_html

        write_comparison_html(result, html_path)
        click.echo(f"Comparison HTML written to {html_path}")
