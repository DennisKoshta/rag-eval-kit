# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-04-16

### Added
- **Retrieval metric pack**: `recall_at_k`, `hit_rate_at_k`, `mrr`, and `ndcg_at_k` — standard IR metrics that sit alongside the existing `precision_at_k`. All pure-Python, zero new dependencies.
- **Answer-quality metric pack**: `contains` (substring), `f1_token` (SQuAD-style token F1), and `rouge_l` (LCS-based F-measure) bridge the gap between the strict `exact_match` and the expensive `llm_judge`.
- `llm_faithfulness` — LLM-based hallucination-detection judge that scores how well the system answer is grounded in `result.retrieved_docs` (as opposed to `llm_judge`, which scores correctness against `expected_answer`).
- `examples/answer_quality_config.yaml` and `examples/retrieval_quality_config.yaml` demonstrating the new metric packs.
- "Choosing a metric" decision table in [docs/writing_metrics.md](docs/writing_metrics.md).

### Changed
- `LLMJudge` now inherits from an internal `_LLMScorer` base class that holds the provider/client/parse-score plumbing; `LLMJudge`'s public signature is unchanged. `LLMFaithfulness` is a sibling subclass sharing the same machinery.
- `orchestrator._resolve_metrics` now wraps *any* parameterised per-question metric with `functools.partial` (previously only `precision_at_k` had the wrap; other parameterised metrics would have raised). LLM-based judges are still constructed with kwargs via the special-cased lazy path.

## [0.2.0] - 2026-04-16

### Added
- Parallel query execution within each sweep config via `concurrency:` in YAML or `--concurrency N` on the CLI. I/O-bound LLM calls now overlap through a `ThreadPoolExecutor`; ordering is preserved.
- Resumable sweeps via a JSONL checkpoint. Set `output.checkpoint:` or `--checkpoint PATH`; interrupted runs skip items already recorded. Config fingerprints are verified per row — edited sweep parameters trigger a warning and force re-run.
- `tiktoken`-backed cost estimation. `ragharness[cost]` extra installs `tiktoken`; pre-run estimates now count real prompt tokens from the loaded dataset and consult a built-in per-model pricing table (overridable via the `token_cost` metric).
- `ragharness.cost_utils` module exposing `count_tokens` and `estimate_sweep_cost` for programmatic use.
- `ragharness.checkpoint` module with a thread-safe `CheckpointWriter` and `load_checkpoint` loader.
- Thread-safety section in [docs/writing_adapters.md](docs/writing_adapters.md) covering the contract for custom adapters at `concurrency > 1`.

### Changed
- `RawRAGSystem` and `R2RRAGSystem` lazy client init now uses a double-checked `threading.Lock`, preventing a TOCTOU race at `concurrency > 1`.
- `run_sweep` cost estimate line now reflects dataset-derived token counts instead of a flat 500-token heuristic. The legacy `estimate_cost(n_questions, n_configs, ...)` function is retained for backwards compatibility.

## [0.1.0] - 2026-04-16

Initial public release.

### Added
- `RAGSystem` protocol and `RAGResult` dataclass for zero-inheritance custom systems.
- `EvalDataset` / `EvalItem` with JSONL, CSV, and HuggingFace loaders (dotted-path field access).
- Pydantic-backed YAML config (`dataset`, `system`, `sweep`, `metrics`, `output`) with validation.
- Sweep orchestrator with Cartesian product expansion, per-question and aggregate metrics, tqdm progress.
- Adapters: `raw` (OpenAI/Anthropic), `langchain`, `llamaindex`, `r2r`, `haystack`.
- Metrics: `exact_match`, `precision_at_k`, `llm_judge`, `latency_p50`, `latency_p95`, `token_cost`.
- Reporters: per-question and summary CSV, matplotlib charts (accuracy bars, latency box plots, cost vs accuracy scatter, per-metric bars).
- CLI: `ragharness run`, `ragharness validate`, `ragharness report`.
- Auth helper with `.env` loading and user-friendly `MissingAPIKeyError`.
- Strict mypy, ruff, and a pytest suite with adapter/metric/reporter/config coverage.

[Unreleased]: https://github.com/DennisKoshta/ragharness/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/DennisKoshta/ragharness/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/DennisKoshta/ragharness/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DennisKoshta/ragharness/releases/tag/v0.1.0
