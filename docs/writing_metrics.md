# Writing a custom metric

## Choosing a built-in metric

Pick the metric that matches your task shape before you write a new one.

| Task shape | Recommended metric(s) | Why |
|---|---|---|
| Exact-string QA ("Paris", "42", "Au") | `exact_match`, `contains` | Strict vs. substring — `contains` handles "The capital is Paris." |
| SQuAD-style span QA | `f1_token` | Token-level precision/recall that shrugs off word-order noise |
| Summarisation / long-form | `rouge_l` | LCS-based F-measure; rewards shared ordered phrases |
| Open-ended / creative | `llm_judge` | Semantic correctness when string overlap isn't enough |
| Hallucination detection | `llm_faithfulness` | Judges grounding in `retrieved_docs`, not correctness |
| Retrieval ablation (which retriever?) | `precision_at_k`, `recall_at_k`, `hit_rate_at_k`, `mrr`, `ndcg_at_k` | Standard IR metrics — combine for a full picture |
| Latency-sensitive systems | `latency_p50`, `latency_p95` | Aggregate latency from `metadata.latency_ms` |
| Cost-sensitive sweeps | `token_cost` | Aggregate USD from token counts + pricing |

**Rule of thumb:** stack a cheap string metric with `llm_judge` so you always have a fast sanity-check signal alongside the expensive semantic one. For retrieval work, `recall@k` and `mrr` together tell you coverage *and* ranking quality.

## Writing new metrics

ragharness has two metric kinds, and they plug in through two different registries in [ragharness/metrics/__init__.py](../ragharness/metrics/__init__.py).

| Kind | Signature | Called | Example |
|---|---|---|---|
| **Per-question** | `(EvalItem, RAGResult) -> float` | once per item per config | `exact_match`, `precision_at_k`, `llm_judge` |
| **Aggregate** | `(list[RAGResult], **kwargs) -> float` | once per config after all items run | `latency_p50`, `latency_p95`, `token_cost` |

Means of per-question metrics are auto-computed by the orchestrator as `mean_<name>`, so you typically only need to register a per-question metric to also get its average in the summary CSV and charts.

## 1. Write the function

### Per-question example: semantic similarity

```python
# ragharness/metrics/similarity.py
from __future__ import annotations
from ragharness.dataset import EvalItem
from ragharness.protocol import RAGResult

_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def semantic_similarity(item: EvalItem, result: RAGResult) -> float:
    """Cosine similarity between answer and expected_answer embeddings."""
    from sentence_transformers import util
    model = _get_model()
    emb1 = model.encode(result.answer, convert_to_tensor=True)
    emb2 = model.encode(item.expected_answer, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2).item())
```

Notes:

- Import heavy / optional deps **inside** the function (lazy) so importing ragharness stays fast and doesn't require the extra.
- Return a float in `[0, 1]` where possible — charts and summary tables assume this.
- Use `nan` to signal "could not score this item"; the orchestrator's mean computation will need handling if you do this, so prefer a defined fallback value.

### Aggregate example: total tokens

```python
# ragharness/metrics/tokens.py
from __future__ import annotations
from ragharness.protocol import RAGResult


def total_tokens(results: list[RAGResult]) -> float:
    return float(sum(
        r.metadata.get("prompt_tokens", 0) + r.metadata.get("completion_tokens", 0)
        for r in results
    ))
```

Accept `**kwargs` if you need configurable params — users can then pass them in the config:

```yaml
metrics:
  - total_tokens
  - my_aggregate:
      threshold: 0.8
```

`params` arriving from YAML get wrapped via `functools.partial` by the resolver.

## 2. Register it

Edit [ragharness/metrics/__init__.py](../ragharness/metrics/__init__.py):

```python
from ragharness.metrics.similarity import semantic_similarity
from ragharness.metrics.tokens import total_tokens

PER_QUESTION_REGISTRY["semantic_similarity"] = semantic_similarity
AGGREGATE_REGISTRY["total_tokens"] = total_tokens
```

## 3. Whitelist in the config validator

[ragharness/config.py](../ragharness/config.py) validates metric names up-front so `ragharness validate` catches typos. Add your name to the allowlist there.

## 4. Tests

Put tests under `tests/test_metrics/test_<name>.py`. Stub out any LLM/embedding calls — unit tests should not hit the network. Cover:

- the happy path (known inputs → known outputs)
- empty / missing data edge cases
- the `**kwargs` plumbing if applicable

## 5. Document the metric

Add one paragraph docstring that covers:

- what it measures (plain-English, no jargon)
- which `metadata` or `EvalItem` fields it reads
- what edge cases return (empty lists, missing metadata)
- range of the return value

See [ragharness/metrics/cost.py](../ragharness/metrics/cost.py) for a representative example.

## Registering metrics from user code (no PR needed)

The registries are plain dicts. If you don't want to upstream a metric, register it from your own script before calling `run_sweep`:

```python
from ragharness.metrics import PER_QUESTION_REGISTRY
from ragharness.orchestrator import run_sweep

def my_metric(item, result):
    return 1.0 if "yes" in result.answer.lower() else 0.0

PER_QUESTION_REGISTRY["my_metric"] = my_metric

# now reference "my_metric" in your YAML's metrics list and run_sweep
```

The config validator will complain about the unknown name, so you'll also need to relax the allowlist check or construct `RagHarnessConfig` in Python and skip the YAML path.
