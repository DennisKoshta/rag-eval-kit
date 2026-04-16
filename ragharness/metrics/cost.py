"""Cost aggregate metric computed from token counts and a pricing table."""

from __future__ import annotations

from typing import Any

from ragharness.protocol import RAGResult


def token_cost(results: list[RAGResult], *, pricing: dict[str, Any]) -> float:
    """Total cost in USD for all results, using per-model pricing.

    ``pricing`` maps model identifiers (as reported by adapters in
    ``metadata["model"]``) to dicts with ``input_per_1k`` and
    ``output_per_1k`` keys — the USD cost per 1 000 tokens. Example::

        {
            "gpt-4o":                    {"input_per_1k": 0.005, "output_per_1k": 0.015},
            "claude-sonnet-4-20250514":  {"input_per_1k": 0.003, "output_per_1k": 0.015},
        }

    Expected ``RAGResult.metadata`` keys:

    - ``model`` — string identifier used to look up pricing
    - ``prompt_tokens`` — input token count
    - ``completion_tokens`` — output token count

    If a result's ``model`` is not in the pricing table, or token counts
    are missing, that result contributes 0.0 to the total. Adapters in
    this package (``raw``, ``langchain``, ``llamaindex``, ``haystack``)
    populate these fields automatically from their provider SDKs; custom
    systems need to set them for cost accounting to work.
    """
    total = 0.0
    for r in results:
        model = r.metadata.get("model", "")
        model_pricing = pricing.get(model, {})
        input_rate = model_pricing.get("input_per_1k", 0.0)
        output_rate = model_pricing.get("output_per_1k", 0.0)
        prompt_tokens = r.metadata.get("prompt_tokens", 0)
        completion_tokens = r.metadata.get("completion_tokens", 0)
        total += (prompt_tokens / 1000) * input_rate
        total += (completion_tokens / 1000) * output_rate
    return total
