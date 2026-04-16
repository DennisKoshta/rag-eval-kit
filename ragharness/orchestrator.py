from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import click
from tqdm import tqdm

from ragharness.adapters import create_adapter
from ragharness.auth import check_api_key
from ragharness.config import RagHarnessConfig
from ragharness.dataset import EvalDataset, EvalItem
from ragharness.metrics import (
    AGGREGATE_REGISTRY,
    PER_QUESTION_REGISTRY,
    PerQuestionMetric,
    get_aggregate_metric,
    get_per_question_metric,
)
from ragharness.protocol import RAGResult

logger = logging.getLogger(__name__)


# ── Result types ─────────────────────────────────────────


@dataclass
class RunResult:
    """Results for a single sweep configuration."""

    config_params: dict[str, Any]
    per_question_scores: list[dict[str, float]]
    aggregate_scores: dict[str, float]
    raw_results: list[RAGResult]
    items: list[EvalItem]


@dataclass
class SweepResult:
    """Results across all sweep configurations."""

    runs: list[RunResult] = field(default_factory=list)


# ── Pure helpers ─────────────────────────────────────────


def expand_sweep(sweep_params: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand sweep params into a list of config dicts via Cartesian product."""
    if not sweep_params:
        return [{}]
    keys = list(sweep_params.keys())
    values = [sweep_params[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def estimate_cost(
    n_questions: int,
    n_configs: int,
    *,
    avg_prompt_tokens: int = 500,
    avg_completion_tokens: int = 200,
    input_per_1k: float = 0.003,
    output_per_1k: float = 0.015,
) -> float:
    """Rough cost estimate for the sweep in USD."""
    total_queries = n_questions * n_configs
    prompt_cost = total_queries * (avg_prompt_tokens / 1000) * input_per_1k
    completion_cost = total_queries * (avg_completion_tokens / 1000) * output_per_1k
    return prompt_cost + completion_cost


# ── Metric resolution ───────────────────────────────────


def _resolve_metrics(
    metrics_config: list[str | dict[str, Any]],
) -> tuple[dict[str, PerQuestionMetric], dict[str, Any]]:
    """Turn the config metrics list into ready-to-call metric dicts.

    Returns (per_question_metrics, aggregate_metrics).
    """
    pq_metrics: dict[str, PerQuestionMetric] = {}
    agg_metrics: dict[str, Any] = {}

    for entry in metrics_config:
        if isinstance(entry, str):
            name, params = entry, {}
        else:
            name = next(iter(entry))
            params = entry[name]

        if name in PER_QUESTION_REGISTRY or name == "llm_judge":
            if name == "precision_at_k" and params:
                pq_metrics[name] = partial(get_per_question_metric(name), **params)
            else:
                pq_metrics[name] = get_per_question_metric(name, **params)
        elif name in AGGREGATE_REGISTRY:
            fn = get_aggregate_metric(name)
            if params:
                agg_metrics[name] = partial(fn, **params)
            else:
                agg_metrics[name] = fn
        else:
            logger.warning("Unknown metric %r, skipping", name)

    return pq_metrics, agg_metrics


# ── Main entry point ────────────────────────────────────


def run_sweep(
    config: RagHarnessConfig,
    *,
    dry_run: bool = False,
    no_confirm: bool = False,
    verbose: bool = False,
) -> SweepResult:
    """Execute the full evaluation sweep defined by *config*."""

    # 1. Load dataset
    ds_cfg = config.dataset
    if ds_cfg.source == "jsonl":
        if ds_cfg.path is None:
            raise ValueError(f"dataset.path is required for source={ds_cfg.source!r}")
        dataset = EvalDataset.from_jsonl(ds_cfg.path)
        source_label = ds_cfg.path
    elif ds_cfg.source == "csv":
        if ds_cfg.path is None:
            raise ValueError(f"dataset.path is required for source={ds_cfg.source!r}")
        dataset = EvalDataset.from_csv(ds_cfg.path)
        source_label = ds_cfg.path
    elif ds_cfg.source == "huggingface":
        if ds_cfg.name is None:
            raise ValueError("dataset.name is required for source='huggingface'")
        dataset = EvalDataset.from_huggingface(
            ds_cfg.name,
            split=ds_cfg.split,
            config_name=ds_cfg.config_name,
            question_field=ds_cfg.question_field,
            answer_field=ds_cfg.answer_field,
            docs_field=ds_cfg.docs_field,
            trust_remote_code=ds_cfg.trust_remote_code,
        )
        source_label = f"{ds_cfg.name}:{ds_cfg.split}"
    else:
        raise ValueError(f"Unsupported dataset source: {ds_cfg.source!r}")

    if ds_cfg.limit is not None:
        dataset = EvalDataset(dataset._items[: ds_cfg.limit])

    logger.info("Loaded %d evaluation items from %s", len(dataset), source_label)

    # 2. Expand sweep matrix
    sweep_configs = expand_sweep(config.sweep)
    n_configs = len(sweep_configs)
    n_questions = len(dataset)
    total_queries = n_configs * n_questions

    # 3. Print run plan
    click.echo(f"\n{'=' * 60}")
    click.echo("RAG Evaluation Sweep")
    click.echo(f"{'=' * 60}")
    click.echo(f"  Dataset:        {source_label} ({n_questions} questions)")
    click.echo(f"  Adapter:        {config.system.adapter}")
    click.echo(f"  Configurations: {n_configs}")
    click.echo(f"  Total queries:  {total_queries}")

    # 4. Cost estimate
    est_cost = estimate_cost(n_questions, n_configs)
    if est_cost > 0:
        click.echo(f"  Est. cost:      ${est_cost:.2f}")

    if dry_run:
        click.echo("\n  [DRY RUN] Would execute the above. Exiting.")
        for sc in sweep_configs:
            click.echo(f"    Config: {sc or '(baseline)'}")
        return SweepResult()

    # Fail fast if the adapter needs an API key we don't have.
    check_api_key(config.system.adapter_config)

    if est_cost > 1.0 and not no_confirm:
        if not click.confirm(
            f"\nEstimated cost ${est_cost:.2f} exceeds $1. Continue?"
        ):
            click.echo("Aborted.")
            raise SystemExit(0)

    click.echo(f"{'=' * 60}\n")

    # 5. Resolve metrics
    pq_metrics, agg_metrics = _resolve_metrics(config.metrics)

    # 6. Run each config
    sweep_result = SweepResult()

    for cfg_idx, sweep_params in enumerate(sweep_configs):
        label = (
            ", ".join(f"{k}={v}" for k, v in sorted(sweep_params.items()))
            or "(baseline)"
        )
        click.echo(f"Config {cfg_idx + 1}/{n_configs}: {label}")

        system = create_adapter(
            config.system.adapter, config.system.adapter_config, sweep_params
        )

        results: list[RAGResult] = []
        per_q_scores: list[dict[str, float]] = []

        for item in tqdm(dataset, desc="  Queries", leave=False):
            # Time the query
            start = time.perf_counter()
            result = system.query(item.question)
            elapsed_ms = (time.perf_counter() - start) * 1000
            result.metadata.setdefault("latency_ms", elapsed_ms)
            results.append(result)

            # Per-question metrics
            scores: dict[str, float] = {}
            for metric_name, metric_fn in pq_metrics.items():
                scores[metric_name] = metric_fn(item, result)
            per_q_scores.append(scores)

            if verbose:
                click.echo(f"    {item.question[:60]}  →  {scores}")

        # 7. Aggregate metrics
        agg_scores: dict[str, float] = {}
        for metric_name, metric_fn in agg_metrics.items():
            try:
                agg_scores[metric_name] = metric_fn(results)
            except TypeError:
                # token_cost needs pricing kwarg — caller should have
                # wrapped it via partial; log and skip if not.
                logger.warning("Aggregate metric %r failed, skipping", metric_name)

        # Auto-compute means of per-question metrics
        for pq_name in pq_metrics:
            values = [s[pq_name] for s in per_q_scores]
            agg_scores[f"mean_{pq_name}"] = sum(values) / len(values) if values else 0.0

        click.echo(f"  Results: {agg_scores}")

        sweep_result.runs.append(
            RunResult(
                config_params=sweep_params,
                per_question_scores=per_q_scores,
                aggregate_scores=agg_scores,
                raw_results=results,
                items=list(dataset),
            )
        )

    return sweep_result
