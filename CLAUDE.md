# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

### Dependency Management
- Install dependencies: `uv sync`
- Add a new dependency: `uv add package-name`

### Code Quality
- Lint and format code: `uv run ruff check --fix`
- Type checking: `uv run mypy`

### Testing
- Run all tests with coverage: `uv run pytest --cov=src`
- Run a specific test: `uv run pytest tests/test_file.py::test_function`

## Project Architecture

This is a Python package named "oboyu" with a modern structure:

1. Source code is in `src/oboyu/` with a `py.typed` marker for type checking
2. Tests are in the `tests/` directory

The project uses:
- Python 3.13
- UV as package manager (faster alternative to pip)
- Ruff for linting and formatting with rules: C9, ANN, S, E, F, W, I, D
- MyPy with strict type checking
- Pytest for testing with coverage reporting
- Pre-commit hooks for code quality and conventional commits