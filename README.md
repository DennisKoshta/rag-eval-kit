# ragharness

[![PyPI version](https://img.shields.io/pypi/v/ragharness.svg)](https://pypi.org/project/ragharness/)
[![Python versions](https://img.shields.io/pypi/pyversions/ragharness.svg)](https://pypi.org/project/ragharness/)
[![CI](https://github.com/DennisKoshta/ragharness/actions/workflows/ci.yml/badge.svg)](https://github.com/DennisKoshta/ragharness/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Pluggable RAG evaluation framework. Run any RAG system against a labeled Q/A dataset and get accuracy, latency, and cost metrics across a configuration sweep.

## Why ragharness?

Most teams end up writing one-off eval scripts per project: a loop that calls their RAG pipeline, compares outputs to a golden set, and prints a number. That works once — but the moment you want to sweep across `top_k`, temperatures, or chunking strategies, or compare LangChain vs LlamaIndex on the same dataset, those scripts turn into glue code. ragharness replaces that glue with a YAML config + a single `RAGSystem` protocol (one method: `query`), so the same eval loop drives any system, any metric, any config matrix — and spits out CSVs and charts teams can actually share.

## Installation

```bash
pip install ragharness

# With optional dependencies:
pip install ragharness[anthropic]     # Anthropic Claude
pip install ragharness[openai]        # OpenAI
pip install ragharness[langchain]     # LangChain adapter
pip install ragharness[llamaindex]    # LlamaIndex adapter
pip install ragharness[r2r]           # SciPhi R2R server client
pip install ragharness[haystack]      # Haystack 2.x adapter
pip install ragharness[huggingface]   # HuggingFace datasets
pip install ragharness[cost]          # tiktoken for real pre-run cost estimates
pip install ragharness[all]           # Everything
```

For development:

```bash
git clone https://github.com/DennisKoshta/ragharness.git
cd ragharness
uv venv && uv pip install -e ".[dev,anthropic]"
```

## Quick Start

### 1. Create a dataset (JSONL)

```jsonl
{"question": "What is the capital of France?", "expected_answer": "Paris"}
{"question": "Who wrote Romeo and Juliet?", "expected_answer": "William Shakespeare"}
```

### 2. Write a config

```yaml
dataset:
  source: jsonl
  path: ./data/questions.jsonl

system:
  adapter: raw
  adapter_config:
    llm_provider: anthropic
    llm_model: claude-sonnet-4-20250514

sweep:
  top_k: [3, 5, 10]
  temperature: [0.0, 0.3]

metrics:
  - exact_match
  - latency_p50
  - latency_p95
  - token_cost:
      pricing:
        claude-sonnet-4-20250514:
          input_per_1k: 0.003
          output_per_1k: 0.015

output:
  csv: ./results/run.csv
  charts: ./results/charts/
  html: ./results/report.html
```

### 3. Run

```bash
ragharness run config.yaml
```

This expands the sweep matrix (3 top_k x 2 temperature = 6 configs), runs each against every question, and outputs:

- **CSV** with per-question scores and aggregate summary
- **Charts** (accuracy bars, latency box plots, cost vs accuracy scatter)
- **Summary table** printed to stdout

## CLI Reference

```
ragharness run CONFIG        Run an evaluation sweep
  --dry-run                  Print plan without executing
  --output-dir DIR           Override output directory
  --filter TEXT              Filter configs (e.g. "top_k=5")
  --no-confirm               Skip cost confirmation prompt
  --verbose                  Show per-question results
  --seed N                   Inject reproducibility seed into adapter_config
  --concurrency N            Parallel queries per config (default 1)
  --checkpoint PATH          JSONL checkpoint for resumable runs

ragharness validate CONFIG   Validate a config file without running

ragharness report CSV_PATH   Re-generate charts from existing results
  --output-dir DIR           Output directory for charts
  --html PATH                Generate a self-contained HTML report

ragharness compare CSV_A CSV_B  Compare two results_summary.csv files
  --output PATH              Write comparison CSV
  --threshold FLOAT          Min absolute delta to flag (default 0.05)
  --html PATH                Write an HTML comparison report
```

### Parallel sweeps

LLM calls are I/O-bound, so `--concurrency 8` usually gets a ~6–8× speedup on real datasets with no code changes. Sweep configs still run sequentially (one at a time) so progress output stays readable; parallelism fans out across dataset items within each config. All bundled adapters are thread-safe — if you write a custom adapter, see [docs/writing_adapters.md](docs/writing_adapters.md#thread-safety-when-concurrency--1).

### Resumable runs

Pass `--checkpoint path.jsonl` (or set `output.checkpoint` in YAML) and every completed `(config, question)` pair appends to that file immediately. If the process dies or you Ctrl-C out, re-running with the same checkpoint skips the completed items — no wasted tokens. The checkpoint records each row's `config_params`; if you edit your sweep between runs, mismatched rows are flagged and re-run.

### HTML reports

Set `output.html: ./results/report.html` in your config (or pass `--html` to `ragharness report`) and get a single self-contained HTML file with sortable tables, inline charts, and a text filter for per-question results. No external CSS/JS — the file is fully portable and can be shared as-is.

### Comparing runs

After tweaking a retriever and re-running, use `ragharness compare` to diff two result CSVs:

```bash
ragharness compare results_v1/results_summary.csv results_v2/results_summary.csv --html diff.html
```

Configs are matched by parameter equality. Each metric gets an absolute delta, percentage change, and directional indicator (improved/regressed/unchanged). Latency and cost metrics are direction-aware — a decrease is an improvement.

### Tag-based breakdown

If your dataset includes tags on `EvalItem` (e.g. `{"topic": "physics"}`), ragharness automatically groups per-question scores by tag and reports per-tag averages in `results_tags.csv` and the HTML report's "Tag Breakdown" section. No config changes needed — tags are detected from the data.

## Python API

```python
from ragharness import RAGSystem, RAGResult, EvalDataset
from ragharness.config import load_config
from ragharness.orchestrator import run_sweep
from ragharness.reporters import write_csv, write_charts, write_html

config = load_config("config.yaml")
result = run_sweep(config, no_confirm=True)
write_csv(result, "results/")
write_charts(result, "results/charts/")
write_html(result, "results/report.html")
```

## Custom RAG Systems

Implement the `RAGSystem` protocol -- a single `query` method:

```python
from ragharness import RAGSystem, RAGResult

class MyRAGSystem:
    def query(self, question: str) -> RAGResult:
        # Your retrieval + generation logic here
        docs = my_retriever.search(question, top_k=5)
        answer = my_llm.generate(question, context=docs)
        return RAGResult(
            answer=answer,
            retrieved_docs=docs,
            metadata={
                "latency_ms": elapsed_ms,
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "model": "my-model",
                "top_k": 5,
            },
        )
```

No inheritance required. Any object with a conforming `query` method works.

## Metrics

| Metric | Type | Description |
|---|---|---|
| `exact_match` | Per-question | 1.0 if answer matches expected (case-insensitive) |
| `contains` | Per-question | 1.0 if expected answer appears as a substring of the answer |
| `f1_token` | Per-question | SQuAD-style token F1 between answer and expected |
| `rouge_l` | Per-question | ROUGE-L F-measure based on longest common subsequence |
| `llm_judge` | Per-question | LLM scores correctness 0.0-1.0 |
| `llm_faithfulness` | Per-question | LLM scores how grounded the answer is in retrieved docs |
| `precision_at_k` | Per-question | Fraction of retrieved docs in expected set |
| `recall_at_k` | Per-question | Fraction of expected docs found in top-k retrieved |
| `hit_rate_at_k` | Per-question | 1.0 if any expected doc appears in top-k |
| `mrr` | Per-question | Reciprocal rank of the first retrieved doc that hits |
| `ndcg_at_k` | Per-question | Binary-relevance nDCG over the top-k retrieved |
| `latency_p50` | Aggregate | Median query latency |
| `latency_p95` | Aggregate | 95th percentile query latency |
| `token_cost` | Aggregate | Total estimated cost from token counts |

## Configuration Reference

| Section | Key | Description |
|---|---|---|
| `dataset` | `source` | `jsonl`, `csv`, or `huggingface` |
| `dataset` | `path` | Path to dataset file |
| `dataset` | `limit` | Max questions to evaluate |
| `system` | `adapter` | `raw`, `langchain`, `llamaindex`, `r2r`, or `haystack` |
| `system` | `adapter_config` | Adapter-specific parameters |
| `sweep` | *(any key)* | Lists of values to sweep (Cartesian product) |
| `metrics` | | List of metric names (strings or dicts with params) |
| `output` | `csv` | CSV output path |
| `output` | `charts` | Charts output directory |
| `output` | `html` | Self-contained HTML report path |
| `output` | `checkpoint` | JSONL checkpoint path for resumable runs |
| `concurrency` | | Parallel queries per config (default `1`) |

## Adapters

| Adapter | Status | Install |
|---|---|---|
| `raw` | Implemented | `pip install ragharness[anthropic]` or `[openai]` |
| `langchain` | Implemented | `pip install ragharness[langchain]` |
| `llamaindex` | Implemented | `pip install ragharness[llamaindex]` |
| `r2r` | Implemented | `pip install ragharness[r2r]` |
| `haystack` | Implemented | `pip install ragharness[haystack]` |

## Development

```bash
uv venv && uv pip install -e ".[dev]"
pytest                    # Run tests
ruff check .              # Lint
mypy ragharness/          # Type check
```

## License

[MIT](LICENSE) © 2026 Dennis Koshta
