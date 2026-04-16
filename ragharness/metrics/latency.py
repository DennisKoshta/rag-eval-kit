"""Latency aggregate metrics computed from ``metadata["latency_ms"]``."""

from __future__ import annotations

import numpy as np

from ragharness.protocol import RAGResult


def latency_p50(results: list[RAGResult]) -> float:
    """Median latency (ms) across all results.

    Reads ``metadata["latency_ms"]`` from each ``RAGResult``. When a result
    does not populate this key the orchestrator fills it in from a
    ``time.perf_counter`` measurement around ``system.query(...)``, so this
    metric always has a value — custom adapters do not need to set it
    explicitly unless they want to report a finer-grained timing (e.g.
    server-side latency excluding network round-trip).

    Returns 0.0 for an empty result list.
    """
    if not results:
        return 0.0
    latencies = [r.metadata.get("latency_ms", 0.0) for r in results]
    return float(np.percentile(latencies, 50))


def latency_p95(results: list[RAGResult]) -> float:
    """95th-percentile latency (ms) across all results.

    Same semantics as :func:`latency_p50`. Useful for catching tail-latency
    regressions that median hides.
    """
    if not results:
        return 0.0
    latencies = [r.metadata.get("latency_ms", 0.0) for r in results]
    return float(np.percentile(latencies, 95))
