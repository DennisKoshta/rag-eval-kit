"""R2R adapter: default path and pre-built-client escape hatch.

R2R (SciPhi) runs retrieval + generation server-side. This example assumes
you have a running R2R instance at http://localhost:7272 with documents
already ingested. See https://r2r-docs.sciphi.ai for setup; the quickest
path is ``docker compose up`` from the R2R repo.

Usage:
    python examples/r2r_python.py
"""

from __future__ import annotations

from pathlib import Path

from r2r import R2RClient

from ragharness import EvalDataset
from ragharness.adapters.r2r import R2RRAGSystem
from ragharness.metrics import get_per_question_metric


def default_path() -> None:
    """YAML-equivalent path: adapter builds R2RClient from base_url."""
    system = R2RRAGSystem(
        base_url="http://localhost:7272",
        llm_model="openai/gpt-4o-mini",
        top_k=5,
    )
    _evaluate(system, label="default")


def escape_hatch() -> None:
    """Power-user path: build R2RClient yourself (e.g. with cloud auth)."""
    client = R2RClient(base_url="http://localhost:7272")
    # If you're using R2R Cloud with auth:
    #   client.users.login(email="...", password="...")
    system = R2RRAGSystem(client=client, llm_model="openai/gpt-4o-mini", top_k=5)
    _evaluate(system, label="escape-hatch")


def _evaluate(system: R2RRAGSystem, *, label: str) -> None:
    dataset = EvalDataset.from_jsonl(Path(__file__).parent / "sample_dataset.jsonl")
    exact_match = get_per_question_metric("exact_match")

    print(f"\n--- {label} ---")
    for item in list(dataset)[:3]:
        result = system.query(item.question)
        em = exact_match(item, result)
        print(f"  Q: {item.question}")
        print(f"  A: {result.answer!r}  (expected: {item.expected_answer!r})  em={em:.2f}")


def main() -> None:
    default_path()
    escape_hatch()


if __name__ == "__main__":
    main()
