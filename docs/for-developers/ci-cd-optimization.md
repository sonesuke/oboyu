# CI/CD Workflow Optimization

## Overview

This document describes the optimized CI/CD workflow structure designed to reduce execution time and eliminate redundancy.

## Workflow Structure

### 1. PR Validation (`.github/workflows/pr-validation.yml`)

**Trigger**: Pull requests to main branch

**Purpose**: Fast validation for PR review process

**Jobs**:
- `lint-and-test`: Runs pre-commit hooks only (includes ruff, mypy, pytest)
- `docs-build-validation`: Validates documentation builds

**Optimization**:
- Uses optimized pre-commit hooks that run only on changed files
- Eliminates duplicate tool execution
- Focuses on speed for developer feedback

### 2. Build and Publish (`.github/workflows/test-pypi.yml`)

**Trigger**: Push to main/develop branches only

**Purpose**: Comprehensive testing and package publishing

**Jobs**:
- `test`: Comprehensive test suite with coverage
- `build`: Package building and artifact creation
- `publish-test-pypi`: Publication to Test PyPI (main/develop only)

**Optimization**:
- Removed PR trigger to eliminate unnecessary package builds
- Runs full test suite for release validation
- Only publishes when code is merged

## Performance Improvements

### Before Optimization
- 2 workflows running on every PR
- Duplicate execution of ruff, mypy, pytest
- Unnecessary package building on PRs
- Estimated time: 6-10 minutes per PR

### After Optimization
- 1 workflow for PR validation
- 1 workflow for release builds (push only)
- Pre-commit handles all PR checks efficiently
- Estimated time: 2-4 minutes per PR

## Workflow Responsibilities

| Workflow | Trigger | Speed | Coverage | Purpose |
|----------|---------|--------|----------|---------|
| PR Validation | Pull Request | Fast | Essential checks | Developer feedback |
| Build & Publish | Push to main/develop | Comprehensive | Full validation | Release preparation |

## Developer Workflow

### For Pull Requests
1. Create PR → PR Validation runs
2. Fast feedback on code quality
3. Documentation build validation
4. Ready for review if checks pass

### For Releases
1. Merge to main → Build & Publish runs
2. Comprehensive testing with coverage
3. Package building and artifact creation
4. Publication to Test PyPI

## Local Development

Use the same pre-commit hooks locally for consistency:

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run all checks manually
uv run pre-commit run -a

# Run individual tools for debugging
uv run ruff check --fix
uv run mypy
uv run pytest -m "not slow" -k "not integration"
```

## Benefits

1. **Faster PR feedback**: Reduced CI time for developers
2. **Resource efficiency**: No duplicate tool execution
3. **Clear separation**: PR validation vs release preparation
4. **Consistency**: Same pre-commit hooks locally and in CI
5. **Cost optimization**: Fewer compute minutes used