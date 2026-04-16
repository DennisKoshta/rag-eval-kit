from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvalItem:
    """A single evaluation question with expected answer and optional metadata."""

    question: str
    expected_answer: str
    expected_docs: list[str] | None = None
    tags: dict[str, Any] | None = None


class EvalDataset:
    """Loads and iterates over evaluation items from JSONL or CSV files."""

    def __init__(self, items: list[EvalItem]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[EvalItem]:
        return iter(self._items)

    def __getitem__(self, idx: int) -> EvalItem:
        return self._items[idx]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> EvalDataset:
        """Load evaluation items from a JSONL file.

        Each line must be a JSON object with at least ``question`` and
        ``expected_answer`` keys.  Optional keys: ``expected_docs`` (list
        of strings) and ``tags`` (dict).
        """
        path = Path(path)
        items: list[EvalItem] = []
        with path.open() as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} of {path}: {e}"
                    ) from e
                items.append(
                    EvalItem(
                        question=obj["question"],
                        expected_answer=obj["expected_answer"],
                        expected_docs=obj.get("expected_docs"),
                        tags=obj.get("tags"),
                    )
                )
        return cls(items)

    @classmethod
    def from_csv(cls, path: str | Path) -> EvalDataset:
        """Load evaluation items from a CSV file.

        Required columns: ``question``, ``expected_answer``.
        Optional columns: ``expected_docs`` (JSON-encoded list),
        ``tags`` (JSON-encoded dict).
        """
        path = Path(path)
        items: list[EvalItem] = []
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                expected_docs = None
                if row.get("expected_docs"):
                    expected_docs = json.loads(row["expected_docs"])
                tags = None
                if row.get("tags"):
                    tags = json.loads(row["tags"])
                items.append(
                    EvalItem(
                        question=row["question"],
                        expected_answer=row["expected_answer"],
                        expected_docs=expected_docs,
                        tags=tags,
                    )
                )
        return cls(items)

    @classmethod
    def from_huggingface(
        cls,
        name: str,
        *,
        split: str = "validation",
        config_name: str | None = None,
        question_field: str = "question",
        answer_field: str = "answer",
        docs_field: str | None = None,
        trust_remote_code: bool = False,
    ) -> EvalDataset:
        """Load evaluation items from a HuggingFace dataset.

        Fields are extracted with dotted-path access so nested schemas
        (e.g. SQuAD's ``answers.text[0]``) can be mapped without custom
        code: use ``answer_field="answers.text.0"``.

        Parameters
        ----------
        name:
            Dataset repo id on the Hub (e.g. ``"rajpurkar/squad"``).
        split:
            Dataset split to load (default ``"validation"``).
        config_name:
            Optional sub-config name (e.g. ``"distractor"`` for
            ``hotpot_qa``).
        question_field, answer_field:
            Dotted paths to the question / answer fields.
        docs_field:
            Dotted path to a list-of-strings field used as
            ``expected_docs``. Optional.
        trust_remote_code:
            Forwarded to ``datasets.load_dataset``. Opt-in for safety.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets package required. "
                "Install with: pip install ragharness[huggingface]"
            ) from None

        ds = load_dataset(
            name, config_name, split=split, trust_remote_code=trust_remote_code
        )

        items: list[EvalItem] = []
        for idx, row in enumerate(ds):
            question = _dotted_get(row, question_field)
            if question is None:
                keys = list(row) if isinstance(row, dict) else "?"
                raise ValueError(
                    f"question_field {question_field!r} did not resolve on row {idx} "
                    f"of {name}:{split}. Available keys: {keys}"
                )
            answer = _dotted_get(row, answer_field)
            if answer is None:
                raise ValueError(
                    f"answer_field {answer_field!r} did not resolve on row {idx} "
                    f"of {name}:{split}."
                )

            expected_docs: list[str] | None = None
            if docs_field:
                raw_docs = _dotted_get(row, docs_field)
                if raw_docs is not None:
                    expected_docs = [str(d) for d in raw_docs]

            items.append(
                EvalItem(
                    question=str(question),
                    expected_answer=str(answer),
                    expected_docs=expected_docs,
                )
            )

        return cls(items)


def _dotted_get(obj: Any, path: str) -> Any:
    """Traverse ``obj`` by a dotted path. Numeric segments are list indices.

    Returns ``None`` if any segment is missing or the type doesn't support
    the access (instead of raising), so callers can distinguish "field is
    None/missing" from programming errors without a try/except dance.
    """
    current: Any = obj
    for segment in path.split("."):
        if current is None:
            return None
        try:
            if segment.isdigit():
                current = current[int(segment)]
            else:
                current = current[segment]
        except (KeyError, IndexError, TypeError):
            return None
    return current
