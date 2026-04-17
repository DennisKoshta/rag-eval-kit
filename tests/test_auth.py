from __future__ import annotations

import pytest

from ragbench.auth import (
    MissingAPIKeyError,
    check_api_key,
    infer_provider,
    load_dotenv,
)


class TestInferProvider:
    @pytest.mark.parametrize(
        "model, expected",
        [
            ("claude-sonnet-4-20250514", "anthropic"),
            ("Claude-3-opus", "anthropic"),
            ("gpt-4o", "openai"),
            ("gpt-3.5-turbo", "openai"),
            ("o1-mini", "openai"),
            ("o3-mini", "openai"),
            ("o4-preview", "openai"),
            ("mistral-large", None),
            ("", None),
        ],
    )
    def test_known_prefixes(self, model: str, expected: str | None) -> None:
        assert infer_provider(model) == expected


class TestLoadDotenv:
    def test_missing_file_no_op(self, tmp_path, monkeypatch):
        monkeypatch.delenv("RAGBENCH_TEST_KEY", raising=False)
        assert load_dotenv(tmp_path / "nope.env") is False
        assert "RAGBENCH_TEST_KEY" not in __import__("os").environ

    def test_parses_basic_pairs(self, tmp_path, monkeypatch):
        import os

        env_file = tmp_path / ".env"
        env_file.write_text(
            "# a comment\n"
            "RAGBENCH_TEST_A=foo\n"
            '\nRAGBENCH_TEST_B="bar baz"\n'
            "export RAGBENCH_TEST_C='quoted'\n"
            "no_equals_line\n"
        )
        monkeypatch.delenv("RAGBENCH_TEST_A", raising=False)
        monkeypatch.delenv("RAGBENCH_TEST_B", raising=False)
        monkeypatch.delenv("RAGBENCH_TEST_C", raising=False)

        assert load_dotenv(env_file) is True
        assert os.environ["RAGBENCH_TEST_A"] == "foo"
        assert os.environ["RAGBENCH_TEST_B"] == "bar baz"
        assert os.environ["RAGBENCH_TEST_C"] == "quoted"

    def test_existing_env_wins(self, tmp_path, monkeypatch):
        import os

        env_file = tmp_path / ".env"
        env_file.write_text("RAGBENCH_TEST_X=from_file\n")
        monkeypatch.setenv("RAGBENCH_TEST_X", "from_shell")

        load_dotenv(env_file)
        assert os.environ["RAGBENCH_TEST_X"] == "from_shell"


class TestCheckApiKey:
    def test_no_provider_or_model_is_noop(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        check_api_key({})  # should not raise

    def test_explicit_provider_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(MissingAPIKeyError, match="ANTHROPIC_API_KEY"):
            check_api_key({"llm_provider": "anthropic"})

    def test_inferred_provider_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
            check_api_key({"llm_model": "gpt-4o"})

    def test_key_present_passes(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        check_api_key({"llm_provider": "anthropic", "llm_model": "claude-sonnet-4-20250514"})

    def test_unknown_provider_is_noop(self, monkeypatch):
        check_api_key({"llm_provider": "mystery-cloud"})

    def test_error_message_includes_model(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(MissingAPIKeyError, match="claude-sonnet"):
            check_api_key({"llm_provider": "anthropic", "llm_model": "claude-sonnet-4-20250514"})
