from __future__ import annotations

import itertools
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import click
from tqdm import tqdm

from ragharness.adapters import create_adapter
from ragharness.auth import check_api_key
from ragharness.checkpoint import (
    CheckpointMap,
    CheckpointWriter,
    load_checkpoint,
    row_to_result,
)
from ragharness.config import DatasetConfig, RagHarnessConfig, SystemConfig
from ragharness.cost_utils import (
    estimate_sweep_cost,
    resolve_model_from_config,
    resolve_pricing_from_metrics,
)
from ragharness.dataset import EvalDataset, EvalItem
from ragharness.metrics import (
    AGGREGATE_REGISTRY,
    PER_QUESTION_REGISTRY,
    PerQuestionMetric,
    get_aggregate_metric,
    get_per_question_metric,
)
from ragharness.protocol import RAGResult, RAGSystem

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
    """Rough cost estimate for the sweep in USD.

    Retained for backwards compatibility. :func:`run_sweep` itself uses
    :func:`ragharness.cost_utils.estimate_sweep_cost`, which derives
    prompt-token counts from the real dataset.
    """
    total_queries = n_questions * n_configs
    prompt_cost = total_queries * (avg_prompt_tokens / 1000) * input_per_1k
    completion_cost = total_queries * (avg_completion_tokens / 1000) * output_per_1k
    return prompt_cost + completion_cost


# ── Dataset loading ──────────────────────────────────────


def _load_dataset(ds_cfg: DatasetConfig) -> tuple[EvalDataset, str]:
    """Load a dataset from a DatasetConfig, returning (dataset, human-readable label)."""
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

    return dataset, source_label


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

        if name in ("llm_judge", "llm_faithfulness"):
            pq_metrics[name] = get_per_question_metric(name, **params)
        elif name in PER_QUESTION_REGISTRY:
            fn = get_per_question_metric(name)
            pq_metrics[name] = partial(fn, **params) if params else fn
        elif name in AGGREGATE_REGISTRY:
            fn = get_aggregate_metric(name)
            if params:
                agg_metrics[name] = partial(fn, **params)
            else:
                agg_metrics[name] = fn
        else:
            logger.warning("Unknown metric %r, skipping", name)

    return pq_metrics, agg_metrics


# ── Plan display ────────────────────────────────────────


def _print_run_plan(
    *,
    source_label: str,
    n_questions: int,
    adapter: str,
    n_configs: int,
    total_queries: int,
    est_cost: float,
    concurrency: int,
    checkpoint: str | None,
) -> None:
    """Print the sweep banner shown before execution."""
    click.echo(f"\n{'=' * 60}")
    click.echo("RAG Evaluation Sweep")
    click.echo(f"{'=' * 60}")
    click.echo(f"  Dataset:        {source_label} ({n_questions} questions)")
    click.echo(f"  Adapter:        {adapter}")
    click.echo(f"  Configurations: {n_configs}")
    click.echo(f"  Total queries:  {total_queries}")
    click.echo(f"  Concurrency:    {concurrency}")
    if checkpoint:
        click.echo(f"  Checkpoint:     {checkpoint}")
    if est_cost > 0:
        click.echo(f"  Est. cost:      ${est_cost:.2f}")


# ── Single-item worker ──────────────────────────────────


def _score_item(
    item: EvalItem,
    system: RAGSystem,
    pq_metrics: dict[str, PerQuestionMetric],
) -> tuple[RAGResult, dict[str, float]]:
    """Run *system* on *item* and compute per-question scores."""
    start = time.perf_counter()
    result = system.query(item.question)
    elapsed_ms = (time.perf_counter() - start) * 1000
    result.metadata.setdefault("latency_ms", elapsed_ms)
    scores: dict[str, float] = {
        metric_name: metric_fn(item, result) for metric_name, metric_fn in pq_metrics.items()
    }
    return result, scores


def _load_completed_from_checkpoint(
    checkpoint_rows: CheckpointMap,
    cfg_idx: int,
    n_items: int,
    sweep_params: dict[str, Any],
) -> dict[int, tuple[RAGResult, dict[str, float]]]:
    """Return {item_idx: (result, scores)} for rows matching *sweep_params*."""
    done: dict[int, tuple[RAGResult, dict[str, float]]] = {}
    for item_idx in range(n_items):
        row = checkpoint_rows.get((cfg_idx, item_idx))
        if row is None:
            continue
        if row.get("config_params") != sweep_params:
            logger.warning(
                "Checkpoint row (config_idx=%d, item_idx=%d) has mismatched "
                "config_params; re-running",
                cfg_idx,
                item_idx,
            )
            continue
        scores = {k: float(v) for k, v in (row.get("scores") or {}).items()}
        done[item_idx] = (row_to_result(row), scores)
    return done


# ── Single-config execution ─────────────────────────────


def _run_single_config(
    *,
    cfg_idx: int,
    n_configs: int,
    sweep_params: dict[str, Any],
    dataset: EvalDataset,
    system_cfg: SystemConfig,
    pq_metrics: dict[str, PerQuestionMetric],
    agg_metrics: dict[str, Any],
    verbose: bool,
    concurrency: int = 1,
    checkpoint_writer: CheckpointWriter | None = None,
    checkpoint_rows: CheckpointMap | None = None,
) -> RunResult:
    """Run every dataset item against a single expanded sweep config."""
    label = ", ".join(f"{k}={v}" for k, v in sorted(sweep_params.items())) or "(baseline)"
    click.echo(f"Config {cfg_idx + 1}/{n_configs}: {label}")

    items = list(dataset)
    n_items = len(items)

    done_results: dict[int, tuple[RAGResult, dict[str, float]]] = {}
    if checkpoint_rows:
        done_results = _load_completed_from_checkpoint(
            checkpoint_rows, cfg_idx, n_items, sweep_params
        )
        if done_results:
            click.echo(
                f"  Resuming from checkpoint: {len(done_results)}/{n_items} items already done"
            )

    pending_indices = [i for i in range(n_items) if i not in done_results]

    pending_results: dict[int, tuple[RAGResult, dict[str, float]]] = {}
    if pending_indices:
        system = create_adapter(system_cfg.adapter, system_cfg.adapter_config, sweep_params)

        def _run_one(item_idx: int) -> tuple[int, RAGResult, dict[str, float]]:
            result, scores = _score_item(items[item_idx], system, pq_metrics)
            return item_idx, result, scores

        def _record(idx: int, result: RAGResult, scores: dict[str, float]) -> None:
            pending_results[idx] = (result, scores)
            if checkpoint_writer is not None:
                checkpoint_writer.write(
                    config_idx=cfg_idx,
                    item_idx=idx,
                    config_params=sweep_params,
                    result=result,
                    scores=scores,
                )
            if verbose:
                click.echo(f"    {items[idx].question[:60]}  →  {scores}")

        if concurrency == 1:
            for item_idx in tqdm(pending_indices, desc="  Queries", leave=False):
                idx, result, scores = _run_one(item_idx)
                _record(idx, result, scores)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = executor.map(_run_one, pending_indices)
                for idx, result, scores in tqdm(
                    futures,
                    desc="  Queries",
                    total=len(pending_indices),
                    leave=False,
                ):
                    _record(idx, result, scores)

    results: list[RAGResult] = []
    per_q_scores: list[dict[str, float]] = []
    for item_idx in range(n_items):
        entry = done_results.get(item_idx) or pending_results[item_idx]
        results.append(entry[0])
        per_q_scores.append(entry[1])

    agg_scores: dict[str, float] = {}
    for metric_name, metric_fn in agg_metrics.items():
        try:
            agg_scores[metric_name] = metric_fn(results)
        except TypeError:
            logger.warning("Aggregate metric %r failed, skipping", metric_name)

    for pq_name in pq_metrics:
        values = [s[pq_name] for s in per_q_scores if pq_name in s]
        agg_scores[f"mean_{pq_name}"] = sum(values) / len(values) if values else 0.0

    click.echo(f"  Results: {agg_scores}")

    return RunResult(
        config_params=sweep_params,
        per_question_scores=per_q_scores,
        aggregate_scores=agg_scores,
        raw_results=results,
        items=items,
    )


# ── Main entry point ────────────────────────────────────


def run_sweep(
    config: RagHarnessConfig,
    *,
    dry_run: bool = False,
    no_confirm: bool = False,
    verbose: bool = False,
) -> SweepResult:
    """Execute the full evaluation sweep defined by *config*."""
    dataset, source_label = _load_dataset(config.dataset)
    logger.info("Loaded %d evaluation items from %s", len(dataset), source_label)

    sweep_configs = expand_sweep(config.sweep)
    n_configs = len(sweep_configs)
    n_questions = len(dataset)
    total_queries = n_configs * n_questions

    model = resolve_model_from_config(config.system.adapter_config)
    input_per_1k, output_per_1k = resolve_pricing_from_metrics(config.metrics, model)
    est_cost = estimate_sweep_cost(
        dataset,
        sweep_configs,
        model=model,
        input_per_1k=input_per_1k,
        output_per_1k=output_per_1k,
    )

    _print_run_plan(
        source_label=source_label,
        n_questions=n_questions,
        adapter=config.system.adapter,
        n_configs=n_configs,
        total_queries=total_queries,
        est_cost=est_cost,
        concurrency=config.concurrency,
        checkpoint=config.output.checkpoint,
    )

    if dry_run:
        click.echo("\n  [DRY RUN] Would execute the above. Exiting.")
        for sc in sweep_configs:
            click.echo(f"    Config: {sc or '(baseline)'}")
        return SweepResult()

    check_api_key(config.system.adapter_config)

    if est_cost > 1.0 and not no_confirm:
        if not click.confirm(f"\nEstimated cost ${est_cost:.2f} exceeds $1. Continue?"):
            click.echo("Aborted.")
            raise SystemExit(0)

    click.echo(f"{'=' * 60}\n")

    pq_metrics, agg_metrics = _resolve_metrics(config.metrics)

    checkpoint_rows: CheckpointMap = {}
    checkpoint_writer: CheckpointWriter | None = None
    if config.output.checkpoint:
        checkpoint_rows = load_checkpoint(config.output.checkpoint)
        checkpoint_writer = CheckpointWriter(config.output.checkpoint)

    try:
        sweep_result = SweepResult()
        for cfg_idx, sweep_params in enumerate(sweep_configs):
            sweep_result.runs.append(
                _run_single_config(
                    cfg_idx=cfg_idx,
                    n_configs=n_configs,
                    sweep_params=sweep_params,
                    dataset=dataset,
                    system_cfg=config.system,
                    pq_metrics=pq_metrics,
                    agg_metrics=agg_metrics,
                    verbose=verbose,
                    concurrency=config.concurrency,
                    checkpoint_writer=checkpoint_writer,
                    checkpoint_rows=checkpoint_rows,
                )
            )
    finally:
        if checkpoint_writer is not None:
            checkpoint_writer.close()

    return sweep_result
