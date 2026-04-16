"""Haystack 2.x adapter: pre-built pipeline escape hatch.

Haystack's strength is composable pipelines — retrievers, re-rankers,
prompt builders, generators, and agents wired together by name. Build
whatever you need in Python and pass it to ``HaystackRAGSystem`` via
``pipeline=...``; the adapter drives ``pipeline.run(...)`` per query.

Requires OPENAI_API_KEY.

Usage:
    python examples/haystack_python.py
"""

from __future__ import annotations

from pathlib import Path

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from ragharness import EvalDataset
from ragharness.adapters.haystack import HaystackRAGSystem
from ragharness.metrics import get_per_question_metric

SAMPLE_CORPUS = [
    "Paris is the capital of France.",
    "William Shakespeare wrote Romeo and Juliet.",
    "The speed of light in a vacuum is 299,792,458 meters per second.",
    "World War II ended in 1945.",
    "The chemical symbol for gold is Au.",
    "Leonardo da Vinci painted the Mona Lisa.",
    "Jupiter is the largest planet in our solar system.",
    "Python was created by Guido van Rossum.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Albert Einstein developed general relativity.",
]

PROMPT = """\
Answer the question using the provided documents. Respond with one short phrase.

Documents:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Question: {{ question }}
Answer:
"""


def build_pipeline() -> Pipeline:
    store = InMemoryDocumentStore()
    store.write_documents([Document(content=t) for t in SAMPLE_CORPUS])

    pipe = Pipeline()
    pipe.add_component("retriever", InMemoryBM25Retriever(store))
    pipe.add_component("prompt_builder", PromptBuilder(template=PROMPT))
    pipe.add_component(
        "generator",
        OpenAIGenerator(model="gpt-4o-mini", generation_kwargs={"temperature": 0.0}),
    )
    pipe.connect("retriever.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "generator.prompt")
    return pipe


def main() -> None:
    pipeline = build_pipeline()
    system = HaystackRAGSystem(pipeline=pipeline, llm_model="gpt-4o-mini", top_k=3)

    dataset = EvalDataset.from_jsonl(Path(__file__).parent / "sample_dataset.jsonl")
    exact_match = get_per_question_metric("exact_match")

    for item in list(dataset)[:3]:
        result = system.query(item.question)
        em = exact_match(item, result)
        print(f"  Q: {item.question}")
        print(f"  A: {result.answer!r}  (expected: {item.expected_answer!r})  em={em:.2f}")


if __name__ == "__main__":
    main()
