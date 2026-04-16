from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ── Known metric / adapter names ────────────────────────

KNOWN_METRICS = {
    "exact_match",
    "llm_judge",
    "precision_at_k",
    "latency_p50",
    "latency_p95",
    "token_cost",
}

KNOWN_ADAPTERS = {"raw", "langchain", "llamaindex", "r2r", "haystack"}


# ── Sub-models ──────────────────────────────────────────


class DatasetConfig(BaseModel):
    """Where to load evaluation data from."""

    source: Literal["jsonl", "csv", "huggingface"] = "jsonl"
    path: str | None = None
    name: str | None = None  # HuggingFace dataset name
    split: str = "validation"
    limit: int | None = None
    config_name: str | None = None
    question_field: str = "question"
    answer_field: str = "answer"
    docs_field: str | None = None
    trust_remote_code: bool = False

    @model_validator(mode="after")
    def _check_source_fields(self) -> DatasetConfig:
        if self.source in ("jsonl", "csv") and not self.path:
            raise ValueError(f"source={self.source!r} requires 'path' to be set")
        if self.source == "huggingface" and not self.name:
            raise ValueError("source='huggingface' requires 'name' to be set")
        return self


class SystemConfig(BaseModel):
    """Which RAG adapter to use and how to configure it."""

    adapter: str
    adapter_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("adapter")
    @classmethod
    def _validate_adapter(cls, v: str) -> str:
        if v not in KNOWN_ADAPTERS:
            raise ValueError(
                f"Unknown adapter: {v!r}. Supported: {sorted(KNOWN_ADAPTERS)}"
            )
        return v


class OutputConfig(BaseModel):
    """Where to write results."""

    csv: str | None = "./results/run_{timestamp}.csv"
    charts: str | None = "./results/charts/"


class RagHarnessConfig(BaseModel):
    """Root configuration model for a ragharness evaluation run."""

    dataset: DatasetConfig
    system: SystemConfig
    sweep: dict[str, list[Any]] = Field(default_factory=dict)
    metrics: list[str | dict[str, Any]] = Field(
        default_factory=lambda: list[str | dict[str, Any]](
            ["exact_match", "latency_p50", "latency_p95"]
        )
    )
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("metrics")
    @classmethod
    def _validate_metrics(cls, v: list[str | dict[str, Any]]) -> list[str | dict[str, Any]]:
        for entry in v:
            if isinstance(entry, str):
                name = entry
            elif isinstance(entry, dict):
                if len(entry) != 1:
                    raise ValueError(
                        f"Parameterised metric must have exactly one key, got: {entry}"
                    )
                name = next(iter(entry))
            else:
                raise ValueError(f"Metric entry must be a string or dict, got: {type(entry)}")
            if name not in KNOWN_METRICS:
                raise ValueError(
                    f"Unknown metric: {name!r}. Supported: {sorted(KNOWN_METRICS)}"
                )
        return v


def load_config(path: str | Path) -> RagHarnessConfig:
    """Load and validate a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping, got {type(raw).__name__}"
        )
    return RagHarnessConfig(**raw)
