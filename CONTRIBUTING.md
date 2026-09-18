# Contributing to Alaap

Thank you for your interest in contributing to Alaap. This guide covers development setup, code standards, and submission guidelines.

## Development Setup

### Using venv (recommended)

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

### Using uv

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Code Standards

### Type Hints and Linting

- Full type hints on all public APIs and module-level code.
- `ruff check src tests` must pass with no errors.
- `mypy src/alaap --strict` must pass (or at least have zero new errors).
- Target Python 3.11+ throughout.

### Testing

- Every module ships unit tests in `tests/`.
- Tests use fakes only: no network calls, no API keys, no GPU, no audio hardware.
- Use `pytest -q` to run the full suite.
- Provider implementations are tested against fake responses, never live services.

### Documentation

- Docstrings must explain the *why*, not just the *what*.
- Teaching-quality documentation; this is an open-source project others will learn from.

## Code Organization

All code follows the architecture laid out in `docs/ARCHITECTURE.md`:

- **Core** (`agent.py`, `events.py`, `session.py`, primitives, cost) depends only on stdlib and pydantic. Heavy dependencies (audio, ML, vendor SDKs) live behind lazy imports.
- **Providers** live in `providers/` and are optional extras. Each provider is tested with fakes.
- **No proprietary references** to prior systems, employers, or clients. This is a clean-room implementation.
- **Runtime interfaces** are stable contracts; switching runtimes is a config change, not a rewrite.

## Commits

Use conventional-ish commit messages:

- `feat: add X` for new features.
- `fix: resolve X` for bug fixes.
- `docs: update X` for documentation.
- `test: add tests for X` for test additions.
- `refactor: improve X` for code improvements without behavior change.

Example: `feat: add Deepgram STT provider with streaming support`

## Provider Additions

When adding a new provider (STT, LLM, TTS, S2S):

1. Implement the interface from `providers/base.py`.
2. **Add a rate-card entry** in `cost/rates.py` with the vendor's pricing model.
3. **Set the `estimated` flag honestly** on all usage: `True` for inferred units (e.g. character estimates), `False` for vendor-reported usage.
4. Test against fakes, not live services.
5. Document the provider's capabilities and any limitations (latency assumptions, concurrency limits, etc.).

The cost meter will use your rate card to report accurate call costs. Silent estimation is a bug.

## Submitting Work

1. Create a topic branch: `git checkout -b feat/my-feature`
2. Make changes and commit with conventional messages.
3. Push and open a pull request.
4. Ensure CI passes (ruff, mypy, pytest).
5. Include a description of what changed and why.

## Questions?

Open an issue or start a discussion. We are building this in the open.
