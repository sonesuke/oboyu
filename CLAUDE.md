# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important Git Guidelines

### Pre-commit Hooks
- NEVER use `--no-verify` option when committing
- Always fix lint errors and test failures before committing
- Pre-commit hooks are there to maintain code quality

## Build and Development Commands

### Dependency Management
- Install dependencies: `uv sync`
- Add a new dependency: `uv add package-name`

### Code Quality
- Lint and format code: `uv run ruff check --fix`
- Type checking: `uv run mypy`

### Testing
- Run fast tests (recommended for development): `uv run pytest -m "not slow" -k "not integration"`
- Run all tests with coverage: `uv run pytest --cov=src`
- Run a specific test: `uv run pytest tests/test_file.py::test_function`
- Run slow tests only: `uv run pytest -m "slow"`
- Run integration tests only: `uv run pytest -k "integration"`

**Note**: Slow tests include actual ML model loading and large dataset processing, taking 15+ seconds. Use fast test suite for regular development.

### Code Analysis with Repomix in Claude Code
When analyzing this codebase with Claude Code's repomix tool:
- Use the `mcp__repomix__pack_codebase` tool with `compress: true` option to reduce token count
- If the output is still too large, use `includePatterns` to filter specific files
- Example: `includePatterns: "src/**/*.py,tests/**/*.py,pyproject.toml"`

## Project Architecture

This is a Python package named "oboyu" with a modern structure:

1. Source code is in `src/oboyu/` with a `py.typed` marker for type checking
2. Tests are in the `tests/` directory
3. Documentation in `docs/` directory
4. Configuration in `repomix.config.json` for codebase analysis

The project uses:
- Python 3.13
- UV as package manager (faster alternative to pip)
- Ruff for linting and formatting with rules: C9, ANN, S, E, F, W, I, D
- MyPy with strict type checking
- Pytest for testing with coverage reporting
- Pre-commit hooks for code quality and conventional commits
- Repomix for code analysis and AI comprehension

## Code Organization Guidelines
- Keep source files under 500 lines to maintain readability
- Each module should have a single responsibility and focus on one specific task