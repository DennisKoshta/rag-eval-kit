import json

import pytest

from ragharness.dataset import EvalDataset, EvalItem


def test_eval_item_defaults():
    item = EvalItem(question="What?", expected_answer="Yes")
    assert item.question == "What?"
    assert item.expected_answer == "Yes"
    assert item.expected_docs is None
    assert item.tags is None


def test_eval_item_with_all_fields():
    item = EvalItem(
        question="Capital of France?",
        expected_answer="Paris",
        expected_docs=["doc1"],
        tags={"category": "geography"},
    )
    assert item.expected_docs == ["doc1"]
    assert item.tags["category"] == "geography"


# ── JSONL loading ────────────────────────────────────────


def test_from_jsonl(tmp_path):
    data = [
        {"question": "Q1", "expected_answer": "A1"},
        {"question": "Q2", "expected_answer": "A2", "expected_docs": ["d1", "d2"]},
        {"question": "Q3", "expected_answer": "A3", "tags": {"topic": "science"}},
    ]
    path = tmp_path / "test.jsonl"
    path.write_text("\n".join(json.dumps(d) for d in data))

    ds = EvalDataset.from_jsonl(path)
    assert len(ds) == 3
    assert ds[0].question == "Q1"
    assert ds[0].expected_docs is None
    assert ds[1].expected_docs == ["d1", "d2"]
    assert ds[2].tags == {"topic": "science"}


def test_from_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"question": "Q1", "expected_answer": "A1"}\n'
        "\n"
        '{"question": "Q2", "expected_answer": "A2"}\n'
    )
    ds = EvalDataset.from_jsonl(path)
    assert len(ds) == 2


def test_from_jsonl_malformed_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question": "Q1", "expected_answer": "A1"}\nnot json\n')

    with pytest.raises(ValueError, match="line 2"):
        EvalDataset.from_jsonl(path)


# ── CSV loading ──────────────────────────────────────────


def test_from_csv(tmp_path):
    path = tmp_path / "test.csv"
    path.write_text(
        "question,expected_answer,expected_docs,tags\n"
        'What is 1+1?,2,["doc1"],{"topic": "math"}\n'
        "Capital of France?,Paris,,\n"
    )
    ds = EvalDataset.from_csv(path)
    assert len(ds) == 2
    assert ds[0].question == "What is 1+1?"
    assert ds[0].expected_docs == ["doc1"]
    assert ds[0].tags == {"topic": "math"}
    assert ds[1].expected_docs is None
    assert ds[1].tags is None


# ── Iteration ────────────────────────────────────────────


def test_iteration(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"question": "Q1", "expected_answer": "A1"}\n'
        '{"question": "Q2", "expected_answer": "A2"}\n'
    )
    ds = EvalDataset.from_jsonl(path)
    questions = [item.question for item in ds]
    assert questions == ["Q1", "Q2"]


# ── HuggingFace loading ─────────────────────────────────


def test_dotted_get_flat():
    from ragharness.dataset import _dotted_get

    assert _dotted_get({"a": 1}, "a") == 1
    assert _dotted_get({"a": 1}, "b") is None


def test_dotted_get_nested():
    from ragharness.dataset import _dotted_get

    obj = {"answers": {"text": ["Paris", "France"]}}
    assert _dotted_get(obj, "answers.text.0") == "Paris"
    assert _dotted_get(obj, "answers.text.5") is None
    assert _dotted_get(obj, "answers.missing") is None
    assert _dotted_get(obj, "answers.text.0.upper") is None  # no .upper on str via key


def _install_fake_load_dataset(monkeypatch, rows, calls=None):
    """Patch datasets.load_dataset to return `rows`, recording call kwargs."""
    pytest.importorskip("datasets")
    import datasets as _datasets

    def _fake(*args, **kwargs):
        if calls is not None:
            calls.append({"args": args, "kwargs": kwargs})
        return rows

    monkeypatch.setattr(_datasets, "load_dataset", _fake)


def test_from_huggingface_flat_schema(monkeypatch):
    _install_fake_load_dataset(
        monkeypatch,
        [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ],
    )

    ds = EvalDataset.from_huggingface("fake/ds")
    assert len(ds) == 2
    assert ds[0].question == "Q1"
    assert ds[1].expected_answer == "A2"
    assert ds[0].expected_docs is None


def test_from_huggingface_dotted_path(monkeypatch):
    _install_fake_load_dataset(
        monkeypatch,
        [{"question": "Capital of France?", "answers": {"text": ["Paris"]}}],
    )

    ds = EvalDataset.from_huggingface("fake/squad", answer_field="answers.text.0")
    assert ds[0].expected_answer == "Paris"


def test_from_huggingface_with_docs_field(monkeypatch):
    _install_fake_load_dataset(
        monkeypatch,
        [{"question": "Q", "answer": "A", "context": ["doc1", "doc2"]}],
    )

    ds = EvalDataset.from_huggingface("fake/ds", docs_field="context")
    assert ds[0].expected_docs == ["doc1", "doc2"]


def test_from_huggingface_missing_field_raises(monkeypatch):
    _install_fake_load_dataset(
        monkeypatch,
        [{"question": "Q"}],  # no answer field
    )

    with pytest.raises(ValueError, match="answer_field"):
        EvalDataset.from_huggingface("fake/ds")


def test_from_huggingface_trust_remote_code_default_false(monkeypatch):
    calls: list[dict] = []
    _install_fake_load_dataset(monkeypatch, [{"question": "Q", "answer": "A"}], calls=calls)

    EvalDataset.from_huggingface("fake/ds", split="train", config_name="distractor")
    assert calls[0]["args"] == ("fake/ds", "distractor")
    assert calls[0]["kwargs"]["split"] == "train"
    assert calls[0]["kwargs"]["trust_remote_code"] is False


def test_from_huggingface_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("No module named 'datasets'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"ragharness\[huggingface\]"):
        EvalDataset.from_huggingface("fake/ds")
