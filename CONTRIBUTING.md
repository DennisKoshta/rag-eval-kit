# Contributing to ragharness

Thanks for your interest in contributing. This guide covers the local dev loop, how to extend the two most common integration points (adapters and metrics), and the release process.

## Dev setup

```bash
git clone https://github.com/DennisKoshta/ragharness.git
cd ragharness
uv venv
uv pip install -e ".[dev,all]"
pre-commit install
```

`pre-commit install` wires ruff + mypy + basic hygiene hooks to run on every commit. They run the same checks CI does, so clean commits locally mean green CI.

## Running checks

```bash
pytest                      # tests
ruff check .                # lint
ruff format --check .       # formatting (run without --check to fix)
mypy ragharness/            # types
pre-commit run --all-files  # everything above in one shot
```

All four must pass before a PR merges. The project is strict-mypy, so `Any` needs a justification in the adjacent code.

## Adding an adapter

See [docs/writing_adapters.md](docs/writing_adapters.md) for a step-by-step walkthrough. In short:

1. Create `ragharness/adapters/<name>.py` exposing a class with a `query(self, question: str) -> RAGResult` method.
2. Add a branch to `create_adapter` in [ragharness/adapters/__init__.py](ragharness/adapters/__init__.py).
3. Gate any third-party imports behind a new `[<name>]` extra in [pyproject.toml](pyproject.toml) and raise a helpful `ImportError` with the install hint if the extra is missing.
4. Register the adapter name in the config validator in [ragharness/config.py](ragharness/config.py) (`SystemConfig` validator).
5. Add tests under `tests/test_adapters/test_<name>.py` — mock the provider SDK rather than hitting the network.
6. Add an example under `examples/<name>_config.yaml` and `examples/<name>_python.py`.

## Adding a metric

See [docs/writing_metrics.md](docs/writing_metrics.md). In short:

- **Per-question metric:** function with signature `(EvalItem, RAGResult) -> float`. Register in `PER_QUESTION_REGISTRY` in [ragharness/metrics/__init__.py](ragharness/metrics/__init__.py).
- **Aggregate metric:** function with signature `(list[RAGResult], **kwargs) -> float`. Register in `AGGREGATE_REGISTRY`.
- Tests go under `tests/test_metrics/test_<name>.py`.
- Whitelist the name in `RagHarnessConfig.metrics` validator in [ragharness/config.py](ragharness/config.py) if it needs to be a valid config string.

## Commit & PR conventions

- Keep PRs focused — one adapter, one metric, one bug fix per PR.
- Write the "why" in the PR description, not the commit title.
- Don't touch `ragharness/_version.py` in feature PRs — version bumps happen at release time.
- Don't bundle doc fixes or unrelated cleanup with feature work.

## Release process (maintainers)

1. Ensure `main` is green.
2. Update `CHANGELOG.md`: rename `[Unreleased]` section to `[X.Y.Z] - YYYY-MM-DD` and add a fresh `[Unreleased]` on top.
3. Bump `ragharness/_version.py` to `X.Y.Z`.
4. Commit (`chore: release X.Y.Z`) and tag: `git tag vX.Y.Z && git push --tags`.
5. The `publish.yml` workflow builds, runs tests, and uploads to PyPI via OIDC trusted publishing. No API token is required.
6. Verify the release on https://pypi.org/project/ragharness/ and draft a GitHub release against the tag.
