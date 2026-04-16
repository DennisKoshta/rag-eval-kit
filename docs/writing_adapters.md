# Writing a custom adapter

An adapter is the bridge between a RAG framework (LangChain, LlamaIndex, R2R, …) or a bespoke stack and ragharness's evaluation loop. The contract is deliberately tiny: anything with a `query(self, question: str) -> RAGResult` method satisfies the [`RAGSystem`](../ragharness/protocol.py) protocol. There is no base class, no registration step at import time, and no required `__init__` signature.

This guide walks through adding a first-class adapter (shipped with the package and selectable via `system.adapter: <name>` in YAML). If you only need a one-off for your own codebase, skip ahead to [Ad-hoc custom systems](#ad-hoc-custom-systems).

## 1. Write the adapter class

Create `ragharness/adapters/<name>.py`. The class takes everything it needs as keyword arguments — the orchestrator merges sweep parameters on top of `adapter_config` before instantiation, so any keyword that appears in the YAML `sweep:` block is available here.

```python
# ragharness/adapters/myvendor.py
from __future__ import annotations
import time
from typing import Any
from ragharness.protocol import RAGResult


class MyVendorRAGSystem:
    def __init__(
        self,
        api_url: str,
        top_k: int = 5,
        temperature: float = 0.0,
        **kwargs: Any,  # soak up future sweep params you haven't declared yet
    ) -> None:
        try:
            import myvendor_sdk
        except ImportError:
            raise ImportError(
                "myvendor-sdk required. Install with: pip install ragharness[myvendor]"
            ) from None

        self.client = myvendor_sdk.Client(api_url)
        self.top_k = int(top_k)
        self.temperature = float(temperature)

    def query(self, question: str) -> RAGResult:
        start = time.perf_counter()
        resp = self.client.ask(question, top_k=self.top_k, temperature=self.temperature)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return RAGResult(
            answer=resp.answer,
            retrieved_docs=[d.text for d in resp.sources],
            metadata={
                "latency_ms": elapsed_ms,
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "model": resp.model,
                "top_k": self.top_k,
            },
        )
```

### What to put in `metadata`

The metrics registry reads these keys — populate what you can:

| Key | Type | Consumers |
|---|---|---|
| `latency_ms` | float | `latency_p50`, `latency_p95` (auto-filled by orchestrator if missing) |
| `prompt_tokens` | int | `token_cost` |
| `completion_tokens` | int | `token_cost` |
| `model` | str | `token_cost` (for pricing lookup) |
| `top_k` | int | reporters (appears in detail CSV) |

Anything extra you drop here is passed through to the CSV writer untouched, so it is a fine place to stash adapter-specific debugging info (cache hits, retrieval IDs, etc.).

## 2. Register it in the factory

Edit [ragharness/adapters/__init__.py](../ragharness/adapters/__init__.py) and add a branch:

```python
elif adapter_type == "myvendor":
    from ragharness.adapters.myvendor import MyVendorRAGSystem
    return MyVendorRAGSystem(**merged)
```

Keep the import inside the branch — this keeps `ragharness` importable when the optional dep is missing.

## 3. Whitelist the name in the config validator

Open [ragharness/config.py](../ragharness/config.py) and add `"myvendor"` to the adapter allowlist in `SystemConfig`. Configs with unknown adapter names fail `ragharness validate` with a clear error.

## 4. Declare the optional dependency

In [pyproject.toml](../pyproject.toml):

```toml
[project.optional-dependencies]
myvendor = ["myvendor-sdk>=1.2"]
```

And add `myvendor` to the `all` extra so `pip install ragharness[all]` pulls it in.

## 5. Add tests

`tests/test_adapters/test_myvendor.py`. Mock the SDK — don't hit the network:

```python
from unittest.mock import MagicMock
from ragharness.adapters.myvendor import MyVendorRAGSystem
from ragharness.protocol import RAGSystem


def test_protocol_conformance():
    adapter = MyVendorRAGSystem(api_url="http://x")
    assert isinstance(adapter, RAGSystem)


def test_query_populates_metadata():
    adapter = MyVendorRAGSystem(api_url="http://x")
    adapter.client = MagicMock()
    adapter.client.ask.return_value = ...  # fake response
    result = adapter.query("hi")
    assert result.metadata["model"] == "expected-model"
```

## 6. Ship examples

- `examples/myvendor_config.yaml` — minimal YAML config
- `examples/myvendor_python.py` — equivalent programmatic invocation

## Ad-hoc custom systems

If you don't need a PR to this repo, skip steps 2–6. Any class with a `query` method works with the orchestrator's Python API:

```python
from ragharness import RAGResult
from ragharness.config import load_config
from ragharness.orchestrator import run_sweep

class MySystem:
    def query(self, question):
        ...
        return RAGResult(answer="...", retrieved_docs=[...], metadata={...})

# Instead of loading a YAML, build the RagHarnessConfig in Python and
# supply an adapter factory via the 'raw' adapter's retriever argument,
# or call the metric functions directly on your own loop. See
# examples/programmatic_sweep.py and examples/custom_rag_system.py.
```
